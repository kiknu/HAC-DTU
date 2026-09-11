"""Controlled 225_bottom experiments using the maintained fitter.

python -m hac_asteroids.correction_campaign benchmark|calibrate|fits|report --output-dir ...
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .calibration import Calibration, calibrate_public, curve_metrics
from .datasets import load_lightcurves
from .inversion import (FitConfig, _build_model, _evaluate, _initial_shape, _normalized,
                        optimization_state, run_inversion, two_pass_backward)
from .provenance import code_identity, fingerprint
from pytorch3d_forward import ObservationBatch

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'artifacts/tuning_a2_a3_20260907'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def legacy_calibration():
    return Calibration.load(HISTORY/'calibration/calibration.json')


def fit_config(asteroid, variant, start):
    values = json.loads((HISTORY/'best_config.json').read_text())['fit_config']
    values.update(asteroid=asteroid, frame_count=12 if variant == 'legacy' else 72,
                  validation_frame_count=72, validation_indices=tuple(range(2, 360, 5)),
                  training_block_frames=6, seed=12345 if start else 0,
                  initialization_noise=.01 if start else 0.)
    return FitConfig(**values)


def initial_coefficients(asteroid, config):
    shape, _ = _build_model(config, torch.device('cuda'))
    # The same historical twelve-frame initializer is used for every paired variant.
    return _initial_shape(shape, load_lightcurves(asteroid), np.arange(0, 360, 30)).detach()


def benchmark(output):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rows = []
    for variant in ('legacy', 'combined'):
        config = fit_config(2, variant, 0)
        calibration = replace(legacy_calibration(), response='srgb') if variant == 'combined' else legacy_calibration()
        shape, model = _build_model(config, torch.device('cuda'), calibration)
        p = initial_coefficients(2, config).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([p])
        data = load_lightcurves(2)
        indices = np.arange(0, 360, 30 if variant == 'legacy' else 5)
        angles = torch.tensor(np.radians(data.frames[indices]), dtype=torch.float32, device='cuda')
        target = torch.tensor(_normalized(data.curves[indices]), dtype=torch.float32, device='cuda')
        blocks = [slice(i,i+config.training_block_frames) for i in range(0,len(indices),config.training_block_frames)]
        def render(block):
            return model(p, ObservationBatch.challenge(angles[block], data.columns), compute_binary=False).raw_intensity
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in [0, 1, 2, 3, 4, 40, 41, 42, 43, 44]:
            start = time.perf_counter()
            state = optimization_state(config, step)
            optimizer.param_groups[0]['lr'] = state.learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss = two_pass_backward(render, blocks, target)
            prior = config.regularization_weight*state.regularization_multiplier*shape.prior_residuals(p).square().mean()
            prior.backward()
            if not torch.isfinite(p.grad).all():
                raise RuntimeError('nonfinite benchmark gradient')
            if state.maximum_degree:
                p.grad[shape.parameter_degrees() > state.maximum_degree] = 0
            torch.nn.utils.clip_grad_norm_([p], config.gradient_clip)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            print(f'[benchmark {variant}] step={step} seconds={times[-1]:.3f} loss={float(loss):.6f}', flush=True)
        start=time.perf_counter()
        _evaluate(model, p, data, calibration)
        validation_seconds=time.perf_counter()-start
        rows.append(dict(variant=variant, step_seconds=times, mean_step_seconds=float(np.mean(times)),
                         full_turn_seconds=validation_seconds,
                         projected_fit_seconds=300*np.mean(times)+33*validation_seconds,
                         peak_memory_mib=torch.cuda.max_memory_allocated()/2**20))
        write_json(output/'benchmark.json', rows)
    projected = 4*rows[0]['projected_fit_seconds'] + 6*rows[1]['projected_fit_seconds']
    write_json(output/'runtime_projection.json', dict(ten_fit_seconds=projected,
               total_with_four_hours_controls_seconds=projected+4*3600,
               target_compute_seconds=8*3600, hard_limit_seconds=9.5*3600))


def fits(output, max_seconds):
    torch.set_num_threads(4)
    freeze_path = output/'frozen_configuration.json'
    calibrations = {a: Calibration.load(output/f'calibration/asteroid{a:02d}/calibration.json') for a in (2,3)}
    frozen = dict(calibrations={str(a):asdict(c) for a,c in calibrations.items()},
                  code=code_identity(), fit_configs={f'{a}_{v}_{s}':asdict(fit_config(a,v,s))
                       for s in (0,1) for a in (2,3) for v in ('legacy','coverage','combined') if s==0 or v!='coverage'})
    if freeze_path.exists():
        if fingerprint(json.loads(freeze_path.read_text())) != fingerprint(frozen):
            raise ValueError('frozen campaign configuration changed')
    else:
        write_json(freeze_path, frozen)
    deadline = time.time()+max_seconds
    initial = {}
    for asteroid in (2, 3):
        path = output / f'initial_coefficients_a{asteroid}.npy'
        if not path.exists():
            np.save(path, initial_coefficients(asteroid, fit_config(asteroid, 'legacy', 0)).cpu().numpy())
        initial[asteroid] = np.load(path)
    runs = []
    order = [(a,v,0) for a in (2,3) for v in ('legacy','coverage','combined')]
    order += [(a,v,1) for a in (2,3) for v in ('legacy','combined')]
    for asteroid,variant,start in order:
        directory = output/f'fits/asteroid{asteroid:02d}/{variant}_start{start}'
        if (directory/'run.json').exists():
            summary=json.loads((directory/'run.json').read_text())['summary']
        else:
            if time.time() >= deadline-180:
                write_json(output/'budget_stop.json', dict(next_run=str(directory), reason='checkpointed compute deadline'))
                break
            config=fit_config(asteroid,variant,start)
            calibration=calibrations[asteroid] if variant=='combined' else legacy_calibration()
            resume=directory/'checkpoint.pt'
            try:
                summary=run_inversion(config,calibration,directory,
                     resume=resume if resume.exists() else None,
                     initial_parameters=initial[asteroid], deadline=deadline-120)
            except RuntimeError as error:
                write_json(output/'numerical_failure.json',dict(run=str(directory),error=str(error)))
                raise
        runs.append(dict(asteroid=asteroid,variant=variant,start=start,summary=summary))
        write_json(output/'fit_progress.json',runs)
        if summary['status']!='complete':
            break
    report(output)


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for asteroid in (2,3):
        for start in (0,1):
            fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
            plotted=False
            for variant in ('legacy','coverage','combined'):
                directory=output/f'fits/asteroid{asteroid:02d}/{variant}_start{start}'
                if not (directory/'run.json').exists():
                    continue
                run=json.loads((directory/'run.json').read_text())
                if run['summary'].get('status') != 'complete':
                    continue
                metric=json.loads((directory/'full_metrics.json').read_text())
                difficult=next(r for r in metric['cameras'] if r['camera']=='225_bottom')
                with np.load(directory/'curves.npz') as curves:
                    remaining_mask = ~np.isin(curves['indices'] % 5, (0,2))
                    common_remaining = float(np.sqrt(np.mean((curves['fitted'][remaining_mask]-curves['observed'][remaining_mask])**2)))
                records.append(dict(asteroid=asteroid,start=start,variant=variant,
                    pooled_rmse=metric['pooled_rmse'],rmse_225_bottom=difficult['rmse'],
                    correlation=difficult['correlation'],dark_mean=difficult['dark_mean'],
                    peak_delta_samples=difficult['peak_delta_samples'],peak_amplitude=difficult['peak_amplitude'],
                    validation_rmse=run['summary']['best_validation_rmse'],
                    remaining_rmse=common_remaining,
                    unused_by_actual_training_rmse=run['summary']['remaining_rotation_rmse'],
                    runtime_seconds=run['summary']['runtime_seconds'],memory_mib=run['summary']['peak_memory_mib']))
                with np.load(directory/'curves.npz') as z:
                    j=list(z['columns']).index('225_bottom')
                    if not plotted:
                        axes[0].plot(z['indices'],z['observed'][:,j],'k',label='Original text target',lw=1.5)
                    axes[0].plot(z['indices'],z['fitted'][:,j],label=variant,alpha=.85)
                    axes[1].plot(z['indices'],z['fitted'][:,j]-z['observed'][:,j],label=variant,alpha=.85)
                plotted=True
            if plotted:
                axes[0].set_title(f'Asteroid {asteroid}, start {start}: 225_bottom, all 360 samples')
                axes[0].set_ylabel('Mean-normalized brightness');axes[1].set_ylabel('Residual');axes[1].set_xlabel('Rotation index')
                axes[0].legend();fig.tight_layout()
                fig.savefig(output/f'curves_a{asteroid}_start{start}.png',dpi=160)
            plt.close(fig)
    write_json(output/'results.json',records)
    if records:
        with (output/'results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    required=[(a,v,s) for a in (2,3) for v in ('legacy','coverage','combined') for s in (0,1)
              if s==0 or v!='coverage']
    complete=all(any((r['asteroid'],r['variant'],r['start'])==key for r in records) for key in required)
    means={}
    if complete:
        for a in (2,3):
            for v in ('legacy','combined'):
                rows=[r for r in records if r['asteroid']==a and r['variant']==v]
                means[f'{a}_{v}']={k:float(np.mean([r[k] for r in rows])) for k in ('pooled_rmse','rmse_225_bottom')}
        gates=dict(a2_channel=means['2_combined']['rmse_225_bottom'] <= .8*means['2_legacy']['rmse_225_bottom'],
                   a2_pooled=means['2_combined']['pooled_rmse'] < means['2_legacy']['pooled_rmse'],
                   a3_regression=means['3_combined']['pooled_rmse'] <= 1.05*means['3_legacy']['pooled_rmse'],
                   numerical_finiteness=not (output/'numerical_failure.json').exists())
        verdict='PASS' if all(gates.values()) else 'FAIL'
    else:
        gates={};verdict='INCONCLUSIVE'
    write_json(output/'verdict.json',dict(verdict=verdict,paired_means=means,gates=gates,
               experimental=verdict!='PASS',conditional_validation=True))
    text=[f'# 225_bottom correction campaign: {verdict}', '',
          'Original text targets are unchanged. Validation and remaining rotations are conditional on calibration using public full-turn data.', '',
          'Checkpoint selection uses indices 2,7,...,357 with full-turn normalization. The second start perturbs only degree-1/2 coefficients with seeded Gaussian noise of standard deviation 0.01.', '',
          'Remaining-rotation comparisons use the same 216 indices outside the 72/72 partition for every variant. Run manifests additionally report rows unused by that run\'s actual training set.', '',
          '| Body | Start | Variant | 225_bottom RMSE | Pooled RMSE | Validation | Remaining rotations |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for r in records:
        text.append(f"| {r['asteroid']} | {r['start']} | {r['variant']} | {r['rmse_225_bottom']:.5f} | {r['pooled_rmse']:.5f} | {r['validation_rmse']:.5f} | {r['remaining_rmse']:.5f} |")
    text+=['',f'Gate results: `{json.dumps(gates)}`.', '',
           'See results.csv, verdict.json, frozen_configuration.json, each fit/full_metrics.json (all cameras, peaks and dark intervals), curves.npz, mesh exports, run.json and checkpoints.', '',
           'Calibration is empirical; projection and response are not recovered acquisition settings. Geometry controls and independent orthographic-video checks are reported separately.']
    if means:
        text += ['', '## Paired-start means', '',
                 '| Body / metric | Legacy | Combined | Change |',
                 '| --- | ---: | ---: | ---: |']
        for a in (2,3):
            for key,label in [('rmse_225_bottom','225_bottom RMSE'),('pooled_rmse','pooled RMSE')]:
                before,after=means[f'{a}_legacy'][key],means[f'{a}_combined'][key]
                text.append(f'| A{a} {label} | {before:.6f} | {after:.6f} | {100*(after/before-1):+.2f}% |')
        text += ['', ('The correction remains experimental because the agreed promotion gate failed.'
                     if verdict=='FAIL' else 'The paired results meet the agreed promotion gate.'),
                 'These are two deterministic starts with one shared perturbation; no statistical population claim is implied.']
    if records:
        text += ['', '## Runtime and artifacts', '',
                 f"Completed fit runtime totals {sum(r['runtime_seconds'] for r in records)/3600:.3f} hours; "
                 f"peak allocated memory is {max(r['memory_mib'] for r in records):.1f} MiB.", '',
                 'Before/after curves: [A2 start 0](curves_a2_start0.png), [A2 start 1](curves_a2_start1.png), '
                 '[A3 start 0](curves_a3_start0.png), [A3 start 1](curves_a3_start1.png).', '',
                 'Detailed exports: [fit camera metrics](fit_camera_metrics.csv), [local peak metrics](peak_metrics.csv), '
                 '[saved-state audit](audit_as_run.json), [controls](controls/), '
                 '[implementation and interpretation](../../docs/forward-model-correction-results.md).']
    common_path=output/'controls/common_evaluation.json'
    if common_path.exists():
        text += ['', '## Saved shapes under a common forward model', '',
                 'Each body uses its selected response/projection and the raw reference framing radius. '
                 'Shapes are aligned by one global scale to the public proxy. These fixed-camera errors '
                 'are separate from the radius-relative production scores used by the gate.', '',
                 '| Body | Saved shape | Pooled RMSE | 225_bottom RMSE | Radial log RMS |',
                 '| --- | --- | ---: | ---: | ---: |']
        for row in json.loads(common_path.read_text()):
            channel=next(c for c in row['fixed_model_metrics']['cameras'] if c['camera']=='225_bottom')
            text.append(f"| {row['asteroid']} | {row['label']} | {row['fixed_model_metrics']['pooled_rmse']:.5f} | "
                        f"{channel['rmse']:.5f} | {row['geometry']['radial_log_rms']:.5f} |")
    budget_path=output/'compute_budget.json'
    if budget_path.exists():
        budget=json.loads(budget_path.read_text())
        if 'envelope_elapsed_seconds' in budget:
            text += ['', f"The conservative wall-time envelope from the first benchmark through verification is "
                     f"{budget['envelope_elapsed_seconds']/3600:.2f} hours, including intervening engineering time. "
                     'See compute_budget.json for the eight-hour target and 9.5-hour hard stop.']
    (output/'report.md').write_text('\n'.join(text)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('benchmark','calibrate','fits','report'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=6*3600)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.action=='calibrate':
        calibrate_public((2,3),args.output_dir/'calibration')
    elif args.action=='fits':
        fits(args.output_dir,args.max_seconds)
    else:
        globals()[args.action](args.output_dir)

if __name__=='__main__':main()
