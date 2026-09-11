"""Read-only response, sampling, capacity and fixed-camera parity controls."""
from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import trimesh

from forward_validation import load_benchmark_mesh, normalize_challenge_mesh
from pytorch3d_forward import (RendererConfig, PyTorch3DLightcurveModel, ObservationBatch,
                             challenge_view_directions, image_response, extract_lightcurves)
from shape_model import OctantoidShapeModel, TriangleMesh
from .calibration import Calibration, normalize, curve_metrics, phase_degrees, score_candidates
from .correction_campaign import write_json
from .datasets import load_lightcurves, public_mesh_path
from .provenance import file_sha256, fingerprint, code_identity
from .reference import ray_images
from render_safety import estimate_raster_memory

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/diagnosis_a2_20260907'


def endpoint_curves(endpoints, calibration):
    raw=[]
    for start in range(0,endpoints.shape[1],6):
        linear=torch.from_numpy(calibration.lambertian_fraction*endpoints[0,start:start+6]
                              +(1-calibration.lambertian_fraction)*endpoints[1,start:start+6])
        pixels=image_response(linear,calibration.response,calibration.radiance_gain)
        raw.append(extract_lightcurves(pixels.flatten(0,1),pixels.shape[0],pixels.shape[1],
                                      mode='exact',compute_binary=False).raw_intensity.numpy())
    return normalize(np.concatenate(raw))


def video_control(output):
    """Re-extract saved radiance endpoints with the production response helper.

    Independently rerender a phase subset to verify the archived endpoints,
    and record the source identities rather than silently trusting cache names.
    """
    records=[]
    for asteroid,resolution in ((2,256),(3,128)):
        path=OLD/f'rays_a{asteroid}_raw_{resolution}_s1_1c_smooth_dinf_endpoints.npz'
        video_path=OLD/f'video_a{asteroid}_Asteroid{asteroid}_videos_simulation_orthographic_225_bottom.npz'
        with np.load(path) as z:
            endpoints=z['endpoints'];angles=z['angles']
        video=np.load(video_path)['curve'][:,None]
        mesh=trimesh.load(public_mesh_path(asteroid),force='mesh',process=True)
        views=challenge_view_directions(('225_bottom',),device='cpu',dtype=torch.float64).numpy()
        check_indices=np.array([0,59,119,179,239,299])
        fresh=ray_images(mesh,angles[check_indices],views,resolution,math.inf)
        np.testing.assert_allclose(fresh,endpoints[:,check_indices],atol=2e-6,rtol=2e-5)
        identity=dict(endpoint_file=str(path),endpoint_sha256=file_sha256(path),
                      video_curve_sha256=file_sha256(video_path),mesh_sha256=file_sha256(public_mesh_path(asteroid)),
                      fresh_ray_phases=check_indices.tolist(),fresh_endpoint_max_error=float(np.max(abs(fresh-endpoints[:,check_indices]))),
                      projection='orthographic',alignment='first video frame = zero degree pose',
                      target='decoded orthographic video; never substituted for text')
        saved=dict(observed=video)
        for response in ('linear','srgb'):
            prediction=np.roll(endpoint_curves(endpoints,Calibration(response=response)),1,axis=0)
            metrics=curve_metrics(prediction,video,('225_bottom',))
            records.append(dict(asteroid=asteroid,response=response,metrics=metrics,identity=identity))
            saved[response]=prediction
        np.savez_compressed(output/f'video_a{asteroid}_curves.npz',**saved)
        write_json(output/'video_control.json',records)
    return records


def sampling_control(output):
    rows=[]
    for asteroid in (2,3):
        candidates=[dict(projection='perspective',camera_distance_radii=d,lambertian_fraction=.5,
                         response='linear',radiance_gain=1.,phase_convention='frame_label') for d in (3.5,16.)]
        scored=score_candidates(asteroid,candidates,np.arange(360),output/'sampling_cache')
        data=load_lightcurves(asteroid)
        for row in scored:
            raw=row['raw'];pred=normalize(raw)
            for count,offset in [(12,0),(360,0)]+[(72,i) for i in range(5)]:
                idx=np.arange(offset,360,360//count)
                subset=normalize(raw[idx]);target=normalize(data.curves[idx])
                j=data.columns.index('225_bottom')
                rows.append(dict(asteroid=asteroid,distance=row['candidate']['camera_distance_radii'],
                    count=count,offset=offset,subset_rmse=float(np.sqrt(np.mean((subset-target)**2))),
                    full_normalization_rmse=float(np.sqrt(np.mean((pred[idx]-data.curves[idx])**2))),
                    observed_mean_225_bottom=float(data.curves[idx,j].mean()),
                    model_mean_relative_error_225_bottom=float(raw[idx,j].mean()/raw[:,j].mean()-1),
                    metrics=curve_metrics(pred[idx],data.curves[idx],data.columns)))
        write_json(output/'sampling.json',rows)
    return rows


def geometry_fit(proxy,degree,steps=1200):
    """Fit all a,b,c fields to proxy vertices, with a global reporting scale.

    Public geometry is a diagnostic target only; these coefficients never initialize inversion.
    """
    vertices=proxy.vertices
    directions=vertices/vertices.norm(dim=-1,keepdim=True)
    shape=OctantoidShapeModel(directions,proxy.faces,lmax=degree,r0=1.,check_self_intersections=False)
    p=shape.zero_parameters().requires_grad_(True)
    log_scale=torch.zeros((),device=vertices.device,requires_grad=True)
    optimizer=torch.optim.Adam([p,log_scale],lr=.015)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        fitted=shape.mesh(p).vertices*log_scale.exp()
        loss=(fitted-vertices).square().mean()
        loss.backward()
        optimizer.step()
        if step%50==0:history.append(float(loss))
    fitted=TriangleMesh((shape.mesh(p).vertices*log_scale.exp()).detach(),proxy.faces)
    return fitted,dict(degree=degree,steps=steps,vertex_rmse=float(torch.sqrt((fitted.vertices-vertices).square().mean())),
                       parameters=p.detach().cpu().tolist(),scale=float(log_scale.exp()),loss_history=history)


def radial_geometry_error(mesh, reference, count=4096):
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    k=np.arange(count,dtype=float)+.5
    z=1-2*k/count;phi=k*np.pi*(3-np.sqrt(5))
    directions=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
    radii=[]
    for surface in (mesh,reference):
        hits,ids,_=RayMeshIntersector(surface).intersects_location(np.zeros_like(directions),directions,multiple_hits=True)
        values=np.zeros(count);np.maximum.at(values,ids,np.linalg.norm(hits,axis=1))
        radii.append(values)
    if min(r.min() for r in radii)<=0:
        return dict(missed_rays=int(sum(np.count_nonzero(r<=0) for r in radii)))
    error=np.log(radii[0]/radii[1])
    return dict(radial_log_rms=float(np.sqrt(np.mean(error**2))),radial_log_max=float(np.max(abs(error))),sample_directions=count)


@torch.no_grad()
def render_mesh(mesh,calibration,data,*,reference_radius=None,image_size=96,shadow_size=None,bias=.002,softness=.002):
    if len(mesh.faces)>5120:
        raise ValueError('GPU controls are bounded to 5,120 faces; raw references must use CPU rays')
    renderer=PyTorch3DLightcurveModel(calibration.renderer_config(replace(RendererConfig.optimization(),
        image_size=image_size,shadow_size=shadow_size or image_size,batch_size=16,
        face_limit=len(mesh.faces),allow_large_mesh=len(mesh.faces)>1280,reference_radius=reference_radius,
        shadow_bias_radii=bias,shadow_softness_radii=softness))).cuda()
    raw=[]
    for start in range(0,360,6):
        idx=np.arange(start,min(start+6,360))
        obs=ObservationBatch.challenge(torch.tensor(np.radians(phase_degrees(data,idx,calibration.phase_convention)),
                                         device='cuda',dtype=torch.float32),data.columns)
        raw.append(renderer(mesh,obs,extraction='exact',compute_binary=False).raw_intensity.cpu().numpy())
    return normalize(np.concatenate(raw))


def geometry_control(output,capacity=True):
    rows=[]
    for asteroid in (2,3):
        data=load_lightcurves(asteroid)
        calibration=Calibration.load(output.parent/f'calibration/asteroid{asteroid:02d}/calibration.json')
        raw=normalize_challenge_mesh(trimesh.load(public_mesh_path(asteroid),force='mesh',process=True))
        reference_radius=float(np.linalg.norm(raw.vertices,axis=1).max())
        variants=[]
        preflight = estimate_raster_memory(16, 96, 4, requires_grad=False)
        if preflight.mebibytes > 512:
            raise RuntimeError('geometry diagnostic exceeds raster memory ceiling')
        for subdivisions,degree in ((3,6),(3,10),(4,10)) if capacity else ((3,6),):
            proxy=load_benchmark_mesh(asteroid,20*4**subdivisions,torch.device('cuda'),output/'mesh_cache',allow_large_mesh=subdivisions>3)
            if subdivisions==3 and degree==6:variants.append(('public_proxy',proxy,{}))
            start=time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            fitted,metrics=geometry_fit(proxy,degree)
            metrics.update(runtime_seconds=time.perf_counter()-start,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            variants.append((f'geometry_degree{degree}_faces{len(proxy.faces)}',fitted,metrics))
        for label,mesh,geometry in variants:
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            # Common physical camera gauge for capacity/visibility comparisons.
            pred=render_mesh(mesh,calibration,data,reference_radius=reference_radius)
            metrics=curve_metrics(pred,data.curves,data.columns)
            tag=f'a{asteroid}_{label}'
            surface=trimesh.Trimesh(mesh.vertices.cpu().numpy(),mesh.faces.cpu().numpy(),process=False)
            geometry['dense_surface_error']=radial_geometry_error(surface,raw)
            np.savez_compressed(output/f'{tag}.npz',prediction=pred,observed=data.curves,
                                vertices=mesh.vertices.cpu().numpy(),faces=mesh.faces.cpu().numpy())
            trimesh.Trimesh(mesh.vertices.cpu().numpy(),mesh.faces.cpu().numpy(),process=False).export(output/f'{tag}.stl')
            rows.append(dict(asteroid=asteroid,label=label,metrics=metrics,geometry=geometry,
                             reference_radius=reference_radius,calibration=asdict(calibration),
                             render_seconds=time.perf_counter()-start,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20))
            write_json(output/'geometry.json',rows)
        # Independent raw surface: CPU only, same fixed camera gauge as proxies.
        start=time.perf_counter()
        views=challenge_view_directions(data.columns,device='cpu',dtype=torch.float64).numpy()
        endpoints=ray_images(raw,phase_degrees(data,np.arange(360),calibration.phase_convention),
                             views,96,math.inf if calibration.projection=='orthographic' else calibration.camera_distance_radii,
                             reference_radius=reference_radius)
        pred=endpoint_curves(endpoints,calibration)
        np.savez_compressed(output/f'a{asteroid}_raw_reference.npz',prediction=pred,observed=data.curves)
        rows.append(dict(asteroid=asteroid,label='raw_reference_hard_rays',
                         metrics=curve_metrics(pred,data.curves,data.columns),reference_radius=reference_radius,
                         runtime_seconds=time.perf_counter()-start,mesh_sha256=file_sha256(public_mesh_path(asteroid)),
                         note='CPU first-hit rays; camera/framing held fixed; compare proxy hard rays to isolate visibility'))
        write_json(output/'geometry.json',rows)
        proxy_surface=trimesh.Trimesh(variants[0][1].vertices.cpu().numpy(),variants[0][1].faces.cpu().numpy(),process=False)
        proxy_endpoints=ray_images(proxy_surface,phase_degrees(data,np.arange(360),calibration.phase_convention),
                             views,96,math.inf if calibration.projection=='orthographic' else calibration.camera_distance_radii,
                             reference_radius=reference_radius)
        proxy_pred=endpoint_curves(proxy_endpoints,calibration)
        np.savez_compressed(output/f'a{asteroid}_proxy_hard_rays.npz',prediction=proxy_pred,observed=data.curves)
        rows.append(dict(asteroid=asteroid,label='public_proxy_hard_rays',
                         metrics=curve_metrics(proxy_pred,data.curves,data.columns),reference_radius=reference_radius))
        write_json(output/'geometry.json',rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('video','sampling','geometry','parity','common','soft_exact'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--defer-capacity',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.action=='common':common_evaluation(args.output_dir)
    elif args.action=='geometry':geometry_control(args.output_dir,not args.defer_capacity)
    else:globals()[args.action+'_control'](args.output_dir)

def parity_control(output):
    """Fixed-camera hard-ray/shadow-map comparisons; no production retuning."""
    records=[]
    degrees=np.array([.1,15.1,45.1,89.1,123.1,180.1,225.1,300.1])
    columns=('0_top','225_bottom')
    views=challenge_view_directions(columns,device='cpu',dtype=torch.float64).numpy()
    proxy=load_benchmark_mesh(2,1280,torch.device('cpu'),output/'mesh_cache')
    shapes=dict(sphere=trimesh.creation.icosphere(subdivisions=2),
                box=trimesh.creation.box(),
                sawed_cube_proxy=trimesh.Trimesh(proxy.vertices.numpy(),proxy.faces.numpy(),process=False))
    calibration=Calibration.load(output.parent/'calibration/asteroid02/calibration.json')
    for label,source in shapes.items():
        radius=float(np.linalg.norm(source.vertices,axis=1).max())
        mesh=TriangleMesh(torch.tensor(np.asarray(source.vertices),device='cuda',dtype=torch.float32),
                          torch.tensor(np.asarray(source.faces),device='cuda',dtype=torch.long))
        reference={}
        for resolution,smooth in ((96,True),(96,False),(192,True)):
            endpoints=ray_images(source,degrees,views,resolution,
                    math.inf if calibration.projection=='orthographic' else calibration.camera_distance_radii,
                    smooth=smooth,reference_radius=radius)
            reference[(resolution,smooth)]=endpoint_curves(endpoints,calibration)
        for resolution,shadow,bias,softness in ((96,96,.002,.002),(192,96,.002,.002),
                  (96,192,.002,.002),(96,96,.001,.002),(96,96,.004,.002),
                  (96,96,.002,.001),(96,96,.002,.004)):
            config=calibration.renderer_config(replace(RendererConfig.optimization(),
                image_size=resolution,shadow_size=shadow,reference_radius=radius,batch_size=8,
                shadow_bias_radii=bias,shadow_softness_radii=softness))
            renderer=PyTorch3DLightcurveModel(config).cuda()
            observations=ObservationBatch.challenge(torch.tensor(np.radians(degrees),device='cuda',dtype=torch.float32),columns)
            with torch.no_grad():
                pred=renderer(mesh,observations,extraction='exact',compute_binary=False).intensity.cpu().numpy()
            target=reference[(resolution,True)]
            records.append(dict(mesh=label,image_size=resolution,shadow_size=shadow,bias=bias,softness=softness,
                reference_radius=radius,metrics=curve_metrics(pred,target,columns),
                sphere_max_rotation_deviation=float(abs(pred-1).max()) if label=='sphere' else None,
                rotations=degrees.tolist(),normalization='complete eight-phase diagnostic set',
                note='Hard ray smooth normals vs production smooth shadow maps; observer discretization also contributes'))
        records.append(dict(mesh=label,normal_comparison='hard-ray flat vs smooth',
                            metrics=curve_metrics(reference[(96,False)],reference[(96,True)],columns)))
        write_json(output/'parity.json',records)
    return records


def common_evaluation(output):
    """Rescore historical and paired inverse shapes under one fixed model per body."""
    from .reporting import align_and_score_public_meshes
    rows=[]
    for asteroid in (2,3):
        data=load_lightcurves(asteroid)
        cal=Calibration.load(output.parent/f'calibration/asteroid{asteroid:02d}/calibration.json')
        raw=normalize_challenge_mesh(trimesh.load(public_mesh_path(asteroid),force='mesh',process=True))
        radius=float(np.linalg.norm(raw.vertices,axis=1).max())
        proxy=load_benchmark_mesh(asteroid,1280,torch.device('cuda'),output/'mesh_cache')
        reference=(proxy.vertices.cpu().numpy(),proxy.faces.cpu().numpy())
        paths=[ROOT/f'artifacts/tuning_a2_a3_20260907/winner/asteroid{asteroid:02d}/mesh_final.stl']
        paths+=sorted((output.parent/f'fits/asteroid{asteroid:02d}').glob('*/mesh_final.stl'))
        for path in paths:
            source=trimesh.load(path,force='mesh',process=True)
            _,vertices,geometry=align_and_score_public_meshes(np.asarray(source.vertices),np.asarray(source.vertices),np.asarray(source.faces),reference)
            mesh=TriangleMesh(torch.tensor(vertices,device='cuda',dtype=torch.float32),torch.tensor(source.faces,device='cuda',dtype=torch.long))
            pred=render_mesh(mesh,cal,data,reference_radius=radius)
            label='historical' if 'tuning_a2_a3' in str(path) else path.parent.name
            np.savez_compressed(output/f'common_a{asteroid}_{label}.npz',prediction=pred,observed=data.curves)
            # Also score the old mesh with its legacy forward settings, using the new full-turn scoring convention.
            stage_metrics=None
            if label=='historical':
                legacy=Calibration.load(ROOT/'artifacts/tuning_a2_a3_20260907/calibration/calibration.json')
                original=TriangleMesh(torch.tensor(np.asarray(source.vertices),device='cuda',dtype=torch.float32),mesh.faces)
                stage_pred=render_mesh(original,legacy,data)
                stage_metrics=curve_metrics(stage_pred,data.curves,data.columns)
            rows.append(dict(asteroid=asteroid,label=label,mesh_sha256=file_sha256(path),
                             fixed_model_metrics=curve_metrics(pred,data.curves,data.columns),
                             legacy_model_metrics=stage_metrics,geometry=geometry['fitted'],reference_radius=radius))
            write_json(output/'common_evaluation.json',rows)
    return rows


@torch.no_grad()
def soft_exact_control(output):
    rows=[]
    for asteroid in (2,3):
        data=load_lightcurves(asteroid)
        cal=Calibration.load(output.parent/f'calibration/asteroid{asteroid:02d}/calibration.json')
        mesh=load_benchmark_mesh(asteroid,1280,torch.device('cuda'),output/'mesh_cache')
        config=cal.renderer_config(replace(RendererConfig.optimization(),batch_size=16,intensity_slope=8.))
        renderer=PyTorch3DLightcurveModel(config).cuda()
        soft,exact=[],[]
        for start in range(0,360,6):
            idx=np.arange(start,start+6)
            obs=ObservationBatch.challenge(torch.tensor(np.radians(phase_degrees(data,idx,cal.phase_convention)),device='cuda',dtype=torch.float32),data.columns)
            result=renderer(mesh,obs,compute_binary=False,return_images=True)
            soft.append(result.raw_intensity.cpu().numpy())
            reduced=extract_lightcurves(result.images.flatten(0,1),6,len(data.columns),mode='exact',compute_binary=False)
            exact.append(reduced.raw_intensity.cpu().numpy())
        soft,exact=normalize(np.concatenate(soft)),normalize(np.concatenate(exact))
        np.savez_compressed(output/f'soft_exact_a{asteroid}.npz',soft=soft,exact=exact,observed=data.curves)
        rows.append(dict(asteroid=asteroid,slope=8.,soft_exact=curve_metrics(soft,exact,data.columns),
                         soft_observed=curve_metrics(soft,data.curves,data.columns),
                         exact_observed=curve_metrics(exact,data.curves,data.columns),
                         note='Same encoded images; full-turn normalization; production soft surrogate retained'))
        write_json(output/'soft_exact.json',rows)
    return rows


if __name__=='__main__':main()
