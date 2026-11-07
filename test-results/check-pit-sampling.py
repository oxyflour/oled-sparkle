"""Repeat the surface comparison on a 0.5 um optical grid using Torch CUDA."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT/'oled-ag-sparkle'
sys.path.insert(0, str(SKILL/'scripts'))
from prepare_random_pits import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-stretched', action='store_true', help='Reproduce the superseded stretched-input experiment')
    parser.add_argument('--legacy-whole-tile', action='store_true')
    args = parser.parse_args()
    emission = SKILL/'assets/oled-1mm/emission.png'
    sampling_name = 'pit-sampling-0p5um'
    if args.legacy_whole_tile:
        emission = SKILL/'assets/oled-1mm-tiled/emission.png'
        sampling_name = 'pit-sampling-tiled-0p5um'
    elif not args.legacy_stretched:
        from prepare_tiled_emission import prepare as prepare_emission
        emission = prepare_emission()/'emission.png'
        sampling_name = 'pit-sampling-periodic-0p5um'
    pits = SKILL/'assets/oled-1mm-pits3to5um-pitch0p5'
    if not (pits/'heightfield.png').exists():
        prepare(pits, pitch_um=.5)
    original = SKILL/'assets/oled-1mm-original-pitch0p5'
    original.mkdir(exist_ok=True)
    source = SKILL/'assets/oled-1mm-height1um/heightfield.png'
    values = np.asarray(Image.open(source), dtype=np.float32)
    values = np.asarray(Image.fromarray(values).resize((2000, 2000), Image.Resampling.BILINEAR))
    Image.fromarray(np.rint(values).astype(np.uint16)).save(original/'heightfield.png')
    (original/'heightfield.source.json').write_text(json.dumps({
        'source': 'existing assets/oled-1mm-height1um/heightfield.png',
        'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'resampling': 'bilinear interpolation of existing 1um samples; no new measured detail',
        'assigned_field_um': [1000, 1000], 'assigned_texel_um': .5,
        'height_range_um': .5}, indent=2), encoding='utf-8')
    for label, folder in (('original', original), ('pits', pits)):
        command = [sys.executable, str(SKILL/'scripts/sparkle_multibackend.py'),
            '--height', str(folder/'heightfield.png'), '--height-pitch-um', '.5',
            '--height-range-um', '.5', '--emission', str(emission),
            '--emission-pitch-um', '2', '--gamma-decode', '--wavelength-nm', '530',
            '--gap-um', '500', '--n-stack', '1.5', '--n-out', '1', '--na', '.08',
            '--source-samples', '1', '--max-sources', '100000', '--pad-px', '280',
            '--batch', '8', '--device', 'cuda', '--backend', 'torch',
            '--ray-radial', '24', '--ray-angular', '64', '--mitsuba-variant', 'cuda_ad_rgb',
            '--analysis-pitch-um', '50', '--roi-margin-um', '0', '--progress-every', '400',
            '--out', str(ROOT/f'test-results/{sampling_name}/{label}')]
        print('RUN', label, flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
