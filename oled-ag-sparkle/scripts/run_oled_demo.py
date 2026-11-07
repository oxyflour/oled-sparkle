#!/usr/bin/env python3
"""Run a 1 mm x 1 mm OLED field; calculate brightness CV on 50 um pixels."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'outputs/oled-1mm-pits3to5um-periodic-50um'))
    parser.add_argument('--height-surface',default='random-pits',choices=['random-pits','original'])
    parser.add_argument('--emission-layout',default='periodic',choices=['periodic','tiled','legacy-whole-tile','legacy-stretched'],
                        help='Crop a repeat cell first; other layouts only reproduce superseded inputs')
    parser.add_argument('--backend',default='all',choices=['all','torch','torchoptics','chromatix','mitsuba'])
    parser.add_argument('--mitsuba-variant',default='cuda_ad_rgb')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--backend-workers',type=int,default=3)
    args=parser.parse_args()
    if args.height_surface=='random-pits':
        from prepare_random_pits import prepare
        inputs=ROOT/'assets/oled-1mm-pits3to5um'
    else:
        from prepare_oled_1mm import prepare
        inputs=ROOT/'assets/oled-1mm-height1um'
    if not (inputs/'heightfield.png').exists():
        inputs=prepare()
    if args.emission_layout=='periodic':
        from prepare_tiled_emission import prepare as prepare_emission
        emission=prepare_emission()/'emission.png'
    elif args.emission_layout in ('tiled','legacy-whole-tile'):
        emission=ROOT/'assets/oled-1mm-tiled/emission.png'
    else:
        emission=ROOT/'assets/oled-1mm/emission.png'
    command=[sys.executable,str(ROOT/'scripts/sparkle_multibackend.py'),
        '--height',str(inputs/'heightfield.png'),'--height-pitch-um','1','--height-range-um','.5',
        '--emission',str(emission),'--emission-pitch-um','2','--gamma-decode',
        '--wavelength-nm','530','--gap-um','500','--n-stack','1.5','--n-out','1','--na','.08',
        '--source-samples','1','--max-sources','100000','--pad-px','140','--batch','16','--device',args.device,'--backend',args.backend,
        '--backend-workers',str(args.backend_workers),
        '--mitsuba-variant',args.mitsuba_variant,'--ray-radial','24','--ray-angular','64',
        '--analysis-pitch-um','50','--roi-margin-um','0','--progress-every','200','--out',args.out]
    subprocess.run(command,check=True)


if __name__=='__main__': main()
