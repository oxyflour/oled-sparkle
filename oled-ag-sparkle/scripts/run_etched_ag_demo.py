#!/usr/bin/env python3
"""Generate an etched-AG proxy and run the existing OLED brightness solvers."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

from prepare_etched_ag import prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pitch-um', type=float, default=1., choices=(1., .5))
    parser.add_argument('--backend', default='all', choices=('all', 'torch', 'torchoptics', 'chromatix', 'mitsuba'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    label = '1um' if args.pitch_um == 1. else '0p5um'
    asset = ROOT / 'assets' / ('oled-1mm-etched-ag' if label == '1um' else 'oled-1mm-etched-ag-pitch0p5')
    prepare(asset, pitch_um=args.pitch_um)
    metadata = json.loads((asset / 'heightfield.source.json').read_text(encoding='utf-8'))
    out = args.out or ROOT / 'outputs/oled-etched-ag' / label
    command = [sys.executable, str(ROOT / 'scripts/sparkle_multibackend.py'),
               '--height', str(asset / 'heightfield.png'), '--height-pitch-um', str(args.pitch_um),
               '--height-range-um', str(metadata['height_range_um']),
               '--emission', str(ROOT / 'assets/oled-1mm-periodic/emission.png'),
               '--emission-pitch-um', '2', '--gamma-decode',
               '--wavelength-nm', '530', '--gap-um', '500', '--n-stack', '1.5', '--n-out', '1',
               '--na', '.08', '--source-samples', '1', '--max-sources', '100000',
               '--pad-px', str(round(140 / args.pitch_um)),
               '--batch', '16' if args.pitch_um == 1. else '8',
               '--device', 'cuda', '--backend', args.backend, '--backend-workers', '3',
               '--mitsuba-variant', 'cuda_ad_rgb', '--ray-radial', '24', '--ray-angular', '64',
               '--analysis-pitch-um', '50', '--roi-margin-um', '0', '--progress-every', '400',
               '--out', str(out)]
    if args.dry_run:
        command.append('--dry-run')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
