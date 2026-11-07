#!/usr/bin/env python3
"""Run full-green/no-LDF SMS reference stages, separately from raw brightness CV.

Use --stage all for preparation, six optical runs and the separated-sparkle report.
Use generate_report.py --kind sms-reference to reprocess saved runs without optics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from PIL import Image
from run_oled_demo import build_command

ROOT = Path(__file__).resolve().parents[1]
CAMERA = {'focal_length_mm': 50., 'f_number': 8., 'object_distance_mm': 400.,
          'sensor_pitch_um': 3.75, 'object_pitch_um': 26.25,
          'magnification': 1/7, 'object_NA': math.sin(math.atan(50/8/2/400)),
          'geometric_green_pitch_um': 70., 'green_pixel_ISR': 70/26.25}


def prepare_green():
    src = ROOT / 'assets/oled-1mm-periodic/emission.png'
    folder = ROOT / 'assets/oled-1mm-periodic-green'
    folder.mkdir(exist_ok=True)
    rgb = np.asarray(Image.open(src).convert('RGB'))
    green = np.zeros_like(rgb)
    green[..., 1] = rgb[..., 1]
    path = folder / 'emission.png'
    Image.fromarray(green).save(path)
    meta = json.loads(src.with_suffix('.source.json').read_text(encoding='utf-8'))
    meta.update({'intensity_model': 'Original green channel only, sRGB-decoded to linear green intensity at 530 nm; R/B zero. Image is an illustrative emitter proxy, not calibrated panel radiometry.',
                 'parent_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'processing': 'G unchanged byte for byte; R=B=0; no resizing, repainting, edge blending or lattice changes',
                 'full_emission_supercell_um': 140.,
                 'green_centers_geometric_pitch_um': 70.,
                 'green_subcell_caveat': 'The four green footprints within the full supercell are not identical. The exact intensity repeat is 140 um, not 70 um.'})
    path.with_suffix('.source.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def command_for(out, pitch=1., backend='all', na=None, dry=False, backend_workers=1):
    return build_command(pitch_um=pitch, backend=backend, device='cuda', out=out, dry_run=dry,
                         backend_workers=backend_workers,
                         solver_options={
                             'emission': ROOT / 'assets/oled-1mm-periodic-green/emission.png',
                             'emission_channel': 'green',
                             'na': CAMERA['object_NA'] if na is None else na,
                             'analysis_pitch_um': CAMERA['object_pitch_um'],
                             'roi_margin_um': 220, 'progress_every': 400})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/oled-sms-reference',
                        help='Root for protocol, run directories and reports')
    parser.add_argument('--stage', choices=('prepare', 'main', 'fine', 'old-na', 'all'), required=True,
                        help='prepare: inputs/protocol/dry-run; main/fine/old-na: optical runs; all: complete workflow')
    parser.add_argument('--backend-workers', type=int, default=1,
                        help='Concurrent backend workers; default 1 for sequential timing on one GPU')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare', 'all'):
        from prepare_etched_ag import prepare
        from prepare_tiled_emission import prepare as prepare_emission
        for pitch, name in ((1., 'oled-1mm-etched-ag'), (.5, 'oled-1mm-etched-ag-pitch0p5')):
            prepare(ROOT / 'assets' / name, pitch_um=pitch)
        prepare_emission()
        prepare_green()
        spec = {'status': 'predeclared before computing green optical results',
                'execution': {'batch_id': uuid4().hex,
                              'created_at_utc': datetime.now(timezone.utc).isoformat(),
                              'backend_workers': args.backend_workers,
                              'solver_sha256': hashlib.sha256((ROOT / 'scripts/sparkle_multibackend.py').read_bytes()).hexdigest(),
                              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                'camera': CAMERA, 'height': 'Existing etched-AG proxy unchanged, Sa = 0.10 um',
                'includes_LDF': False, 'roi_width_um': 560.,
                'primary_filter': {'method': 'Fourier lattice notch, DC retained', 'full_supercell_um': 140.,
                                   'notch_guard_bins': 0, 'radial_cutoff_cycles_per_um': 1/70.},
                'sensitivity_cases': ['notch guard 1 bin', '0.8x band cutoff', 'three full-supercell 140 um MWA passes', 'original NA 0.08', '0.5 um optical grid'],
                'reference_comparison_percent': [4.8, 6.0],
                'comparison_status': 'Scale check only. Cheng 2025 did not publish instrument settings; its Ra=0.25-0.35 um surfaces and panel are different.',
                'sources': ['https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf',
                            'https://www.display-messtechnik.de/fileadmin/template/main/docs/SMS1000-05.2023-7.pdf',
                            'https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf']}
        (args.out / 'protocol.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
        subprocess.run(command_for(args.out / 'dry-run', dry=True, backend_workers=args.backend_workers), check=True)
        if args.stage == 'prepare':
            return
    elif not (args.out / 'protocol.json').is_file():
        parser.error('Missing protocol.json; run --stage prepare before optical stages')
    settings = {'main': ('1um', 1., 'all', None),
                'fine': ('0p5um', .5, 'torch', None),
                'old-na': ('old-na-0p08', 1., 'torch', .08)}
    for stage in settings if args.stage == 'all' else (args.stage,):
        name, pitch, backend, na = settings[stage]
        subprocess.run(command_for(args.out / name, pitch, backend, na, backend_workers=args.backend_workers), check=True)
    if args.stage == 'all':
        from reporting import write_sms_report
        print('HTML report:', write_sms_report(args.out))


if __name__ == '__main__':
    main()
