"""Standalone figures and detailed peak diagnostics from saved campaign arrays."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-corrections')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .correction_campaign import write_json


def peaks(prediction,observed):
    """Compare each observed peak in a shared circular +/-10 degree window."""
    candidates=np.flatnonzero((observed>np.roll(observed,1)) & (observed>=np.roll(observed,-1)))
    indices=[]
    for index in sorted(candidates,key=lambda i:-observed[i]):
        left=observed[(index-np.arange(1,16))%len(observed)].min()
        right=observed[(index+np.arange(1,16))%len(observed)].min()
        if observed[index]-max(left,right)<.15:
            continue
        if all(min(abs(index-old),len(observed)-abs(index-old))>=15 for old in indices):
            indices.append(index)
    indices.sort()
    rows=[]
    for index in indices:
        offsets=np.arange(-10,11)
        local=prediction[(index+offsets)%len(prediction)]
        offset=int(offsets[np.argmax(local)])
        rows.append(dict(observed_index=int(index),predicted_index=int((index+offset)%len(prediction)),
                         offset_degrees=offset,observed_amplitude=float(observed[index]),
                         predicted_amplitude=float(local.max()),window_boundary=abs(offset)==10))
    return rows


def render(output):
    controls=output/'controls';figdir=output/'figures';figdir.mkdir(exist_ok=True)
    peak_rows=[]
    fit_camera_rows=[]
    for asteroid in (2,3):
        path=controls/f'video_a{asteroid}_curves.npz'
        if path.exists():
            with np.load(path) as z:
                fig,ax=plt.subplots(figsize=(11,3.5))
                for key,label in [('observed','Orthographic video'),('linear','Linear response'),('srgb','sRGB response')]:
                    ax.plot(z[key].ravel(),label=label,lw=1.3)
                ax.set(title=f'Asteroid {asteroid}: known orthographic video control',xlabel='Frame index',ylabel='Normalized brightness')
                ax.legend();fig.tight_layout();fig.savefig(figdir/f'video_a{asteroid}.png',dpi=170);plt.close(fig)
        for run in sorted((output/f'fits/asteroid{asteroid:02d}').glob('*/curves.npz')):
            metrics=json.loads((run.parent/'full_metrics.json').read_text())
            fit_camera_rows.extend(dict(asteroid=asteroid,variant=run.parent.name,**camera)
                                   for camera in metrics['cameras'])
            with np.load(run) as z:
                j=list(z['columns']).index('225_bottom')
                for row in peaks(z['fitted'][:,j],z['observed'][:,j]):
                    peak_rows.append(dict(asteroid=asteroid,variant=run.parent.name,**row))
        geometry=list(controls.glob(f'a{asteroid}_geometry_degree*_faces*.npz'))
        proxy=controls/f'a{asteroid}_public_proxy.npz';raw=controls/f'a{asteroid}_raw_reference.npz'
        if geometry and raw.exists():
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
            paths=[raw,proxy]+geometry
            for path in paths:
                with np.load(path) as z:
                    from .datasets import load_lightcurves
                    j=load_lightcurves(asteroid).columns.index('225_bottom')
                    if path==raw:axes[0].plot(z['observed'][:,j],'k',label='Text target',lw=1.3)
                    label=path.stem.replace(f'a{asteroid}_','')
                    axes[0].plot(z['prediction'][:,j],label=label,alpha=.8)
                    axes[1].plot(z['prediction'][:,j]-z['observed'][:,j],label=label,alpha=.8)
            axes[0].set_title(f'Asteroid {asteroid}: geometry controls, fixed camera and framing')
            axes[0].set_ylabel('Normalized brightness');axes[1].set_ylabel('Residual');axes[1].set_xlabel('Frame index')
            axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/f'geometry_a{asteroid}.png',dpi=170);plt.close(fig)
    write_json(output/'peak_metrics.json',peak_rows)
    if fit_camera_rows:
        with (output/'fit_camera_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fit_camera_rows[0]);writer.writeheader();writer.writerows(fit_camera_rows)
    if peak_rows:
        with (output/'peak_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=peak_rows[0]);writer.writeheader();writer.writerows(peak_rows)
    path=controls/'sampling.json'
    if path.exists():
        rows=json.loads(path.read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,asteroid in zip(axes,(2,3)):
            for distance in (3.5,16.):
                selected=sorted([r for r in rows if r['asteroid']==asteroid and r['count']==72 and r['distance']==distance],key=lambda r:r['offset'])
                ax.plot([r['offset'] for r in selected],[r['subset_rmse'] for r in selected],'o-',label=f'{distance:g} radii, subset norm')
                ax.plot([r['offset'] for r in selected],[r['full_normalization_rmse'] for r in selected],'x--',label=f'{distance:g} radii, full norm')
            ax.set(title=f'Asteroid {asteroid}: five offsets',xlabel='Offset of 5-degree grid',ylabel='Pooled RMSE');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(figdir/'sampling_offsets.png',dpi=170);plt.close(fig)
    rows=[]
    for path in sorted((output/'calibration').glob('asteroid*/finalists.json')):
        for row in json.loads(path.read_text()):
            for camera in row['metrics']['cameras']:
                rows.append(dict(asteroid=int(path.parent.name[-2:]),**row['candidate'],**camera))
    if rows:
        with (output/'calibration_camera_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output_dir',type=Path)
    render(parser.parse_args().output_dir)

if __name__=='__main__':main()
