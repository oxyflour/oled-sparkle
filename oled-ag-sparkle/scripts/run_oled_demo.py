#!/usr/bin/env python3
"""Run the sourced OLED example; raw brightness CV on 50 um measurement pixels.

Default: etched AG, 1 um optical grid, all four backends. SMS-separated sparkle
uses run_sms_reference.py instead. build_command is shared by that workflow.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def build_command(*, height_surface='etched-ag', pitch_um=1., emission_layout='periodic',
                  backend='all', device='auto', backend_workers=3,
                  mitsuba_variant='cuda_ad_rgb', out=None, dry_run=False,
                  skip_unavailable=False, solver_options=None, extra=()):
    """Prepare selected inputs and build one solver invocation with shared defaults.

    solver_options lets the SMS workflow set its green input and camera geometry;
    it does not change how the solver computes raw brightness CV.
    """
    if height_surface=='original' and pitch_um!=1.:
        raise ValueError('The original interpolated surface is available only at 1 um')
    if height_surface=='etched-ag':
        from prepare_etched_ag import DEFAULT_CASE, DEFAULT_INPUTS, prepare
        inputs=DEFAULT_INPUTS[pitch_um]
        case=DEFAULT_CASE+'-periodic-50um'
    elif height_surface=='random-pits':
        from prepare_random_pits import prepare
        inputs=ROOT/'assets/oled-1mm-pits3to5um'
        case='oled-1mm-pits3to5um-periodic-50um'
    else:
        from prepare_oled_1mm import prepare
        inputs=ROOT/'assets/oled-1mm-height1um'
        case='oled-1mm-original-periodic-50um'
    if pitch_um!=1.:
        if height_surface!='etched-ag':
            inputs=inputs.with_name(inputs.name+'-pitch0p5')
        case+='-pitch0p5'
    if emission_layout!='periodic':
        case=case.replace('-periodic-',f'-{emission_layout}-')
    if not (inputs/'heightfield.png').exists():
        inputs=prepare(inputs, pitch_um=pitch_um) if height_surface!='original' else prepare()
    metadata=json.loads((inputs/'heightfield.source.json').read_text(encoding='utf-8'))
    height_pitch=metadata['assigned_texel_um']
    height_range=metadata['height_range_um']
    if height_pitch!=pitch_um:
        raise ValueError(f'Height sidecar pitch {height_pitch} differs from requested {pitch_um}')
    out=Path(out).expanduser().resolve() if out else ROOT/'outputs'/case
    if emission_layout=='periodic':
        emission=ROOT/'assets/oled-1mm-periodic/emission.png'
        if not emission.exists():
            from prepare_tiled_emission import prepare as prepare_emission
            emission=prepare_emission()/'emission.png'
    elif emission_layout=='legacy-whole-tile':
        emission=ROOT/'assets/oled-1mm-tiled/emission.png'
    else:
        emission=ROOT/'assets/oled-1mm/emission.png'
    options={'height':inputs/'heightfield.png', 'height_pitch_um':height_pitch,
             'height_range_um':height_range, 'emission':emission, 'emission_pitch_um':2,
             'gamma_decode':True, 'wavelength_nm':530, 'gap_um':500, 'n_stack':1.5,
             'n_out':1, 'na':.08, 'source_samples':1, 'max_sources':100000,
             'pad_px':round(140/pitch_um), 'batch':16 if pitch_um==1. else 8,
             'device':device, 'backend':backend, 'backend_workers':backend_workers,
             'mitsuba_variant':mitsuba_variant, 'ray_radial':24, 'ray_angular':64,
             'analysis_pitch_um':50, 'roi_margin_um':0, 'progress_every':200,
             'out':out, 'dry_run':dry_run, 'skip_unavailable':skip_unavailable}
    options.update(solver_options or {})
    command=[sys.executable,str(ROOT/'scripts/sparkle_multibackend.py')]
    for key,value in options.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value: command.append(flag)
        else:
            command.extend([flag,str(value)])
    command.extend(extra)
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',help='Output directory; defaults to skill outputs/<case>')
    parser.add_argument('--height-surface',default='etched-ag',choices=['etched-ag','random-pits','original'])
    parser.add_argument('--pitch-um',type=float,default=1.,choices=(1.,.5),
                        help='Optical/height pitch; .5 retains geometry and physical padding')
    parser.add_argument('--emission-layout',default='periodic',choices=['periodic','legacy-whole-tile','legacy-stretched'],
                        help='Current repeat cell, or an explicitly selected historical input')
    parser.add_argument('--backend',default='all',choices=['all','torch','torchoptics','chromatix','mitsuba'])
    parser.add_argument('--mitsuba-variant',default='cuda_ad_rgb')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--backend-workers',type=int,default=3)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--skip-unavailable',action='store_true')
    parser.add_argument('--extra',nargs=argparse.REMAINDER,default=[],help='Additional solver flags; place last')
    args=parser.parse_args()
    if args.height_surface=='original' and args.pitch_um!=1.:
        parser.error('--height-surface original supports only --pitch-um 1')
    command=build_command(**vars(args))
    subprocess.run(command,check=True)


if __name__=='__main__': main()
