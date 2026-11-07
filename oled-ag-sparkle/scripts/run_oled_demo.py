#!/usr/bin/env python3
"""Run a 1 mm x 1 mm OLED field; calculate brightness CV on 50 um pixels."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'outputs/oled-1mm-height1um-50um'))
    parser.add_argument('--backend',default='all',choices=['all','torch','torchoptics','chromatix','mitsuba'])
    parser.add_argument('--mitsuba-variant',default='cuda_ad_rgb')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--backend-workers',type=int,default=3)
    args=parser.parse_args()
    from prepare_oled_1mm import prepare
    inputs=ROOT/'assets/oled-1mm-height1um'
    if not (inputs/'heightfield.png').exists():
        inputs=prepare()
    command=[sys.executable,str(ROOT/'scripts/sparkle_multibackend.py'),
        '--height',str(inputs/'heightfield.png'),'--height-pitch-um','1','--height-range-um','.5',
        '--emission',str(ROOT/'assets/oled-1mm/emission.png'),'--emission-pitch-um','2','--gamma-decode',
        '--wavelength-nm','530','--gap-um','500','--n-stack','1.5','--n-out','1','--na','.08',
        '--source-samples','1','--max-sources','100000','--pad-px','140','--batch','16','--device',args.device,'--backend',args.backend,
        '--backend-workers',str(args.backend_workers),
        '--mitsuba-variant',args.mitsuba_variant,'--ray-radial','24','--ray-angular','64',
        '--analysis-pitch-um','50','--roi-margin-um','0','--progress-every','200','--out',args.out]
    subprocess.run(command,check=True)


if __name__=='__main__': main()
