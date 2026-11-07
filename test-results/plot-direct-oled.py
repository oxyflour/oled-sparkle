"""Compare optical-grid and 50 um collected brightness on the same 1 mm field."""
from pathlib import Path
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'oled-1mm-height1um-50um'
fig,axes=plt.subplots(2,4,figsize=(16,8.2),layout='constrained')
for column,name in enumerate(('torch','torchoptics','chromatix','mitsuba')):
    stats=json.loads((DATA/name/'stats.json').read_text())
    context=stats.get('execution_context',{})
    device='CUDA' if (context.get('device')=='cuda' or context.get('variant','').startswith('cuda')) else 'CPU'
    extent=(-stats['field_width_um']/2,stats['field_width_um']/2,
            -stats['field_height_um']/2,stats['field_height_um']/2)
    for row,filename in enumerate(('brightness.npy','analysis_brightness.npy')):
        ax=axes[row,column]; image=np.load(DATA/name/filename)
        image_extent=extent if row==0 else tuple(stats['analysis_extent_um_xyxy'][i] for i in (0,2,1,3))
        m=ax.imshow(image,origin='lower',extent=image_extent,cmap='gray',vmin=0,
                    vmax=float(image.max()),interpolation='nearest')
        if row==0:
            ax.set_title(f'{name} ({device})\nOptical grid: {stats["height_pitch_um"]:g} um\nSolver: {stats["elapsed_seconds"]:.3f} s',fontsize=11)
        else:
            ax.set_title(f'50 um pixels: 20 x 20\nCV = {stats["sparkle_cv_percent"]:.6f}%')
        ax.set(xlabel='x (um)',ylabel='y (um)',xticks=[-500,0,500],yticks=[-500,0,500])
        fig.colorbar(m,ax=ax,shrink=.8,label='Relative linear brightness')
fig.suptitle('1 mm x 1 mm OLED field: height sampling 1 um x 1 um',fontsize=15)
fig.supxlabel('Each 50 x 50 um pixel averages 50 x 50 optical samples; all 400 pixels enter the statistic.\n'
              'CV = 100 x std(I_50um, ddof=0) / mean(I_50um), from floating-point brightness.',fontsize=11)
output=ROOT/'brightness-comparison-height1um-50um.png'
fig.savefig(output,dpi=180)
shutil.copy2(output,ROOT/'brightness-comparison.png')
print(output)
