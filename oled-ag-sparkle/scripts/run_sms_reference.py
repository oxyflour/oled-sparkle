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
import sys
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from PIL import Image
from run_oled_demo import build_command
from prepare_etched_ag import DEFAULT_CASE, DEFAULT_INPUTS

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


def command_for(out, pitch=1., backend='all', na=None, dry=False, backend_workers=1,
                emission=None, emission_pitch_um=2., emission_scale=1., device='cuda',
                emission_threshold=.001, height=None):
    height_options = {}
    if height is not None:
        height = Path(height).resolve()
        meta = json.loads(height.with_suffix('.source.json').read_text(encoding='utf-8'))
        if meta['assigned_texel_um'] != pitch:
            raise ValueError('Custom height pitch does not match the optical grid')
        height_options = {'height': height, 'height_pitch_um': pitch,
                          'height_range_um': meta['height_range_um']}
    return build_command(pitch_um=pitch, backend=backend, device=device, out=out, dry_run=dry,
                         backend_workers=backend_workers,
                         solver_options={
                             'emission': emission or ROOT / 'assets/oled-1mm-periodic-green/emission.png',
                             'emission_pitch_um': emission_pitch_um, 'emission_scale': emission_scale,
                             'emission_threshold': emission_threshold,
                             'emission_channel': 'green',
                             'na': CAMERA['object_NA'] if na is None else na,
                             'analysis_pitch_um': CAMERA['object_pitch_um'],
                             'roi_margin_um': 220, 'progress_every': 400, **height_options})


def run_layout_comparison(args, *, heights=None, computed_layouts=None, coarse_only=False):
    """Use the same runner/solver with a frozen multi-layout measurement plan."""
    from prepare_tiled_emission import prepare_layout
    from sparkle_multibackend import ALL_BACKENDS
    from sms_reference import common_mwa_plan, sampled_camera_plan
    protocol_path = args.out/'protocol.json'
    if args.stage == 'old-na':
        raise ValueError('The layout comparison supports prepare/main/fine/all stages')
    if args.stage in ('prepare', 'all'):
        if protocol_path.exists():
            raise ValueError('Comparison protocol already exists; use main/fine or a new --out directory')
        if args.reuse_optics:
            if args.measurement != 'sampled-camera':
                raise ValueError('--reuse-optics requires --measurement sampled-camera')
            source = args.reuse_optics.resolve()
            if source == args.out:
                raise ValueError('Use a separate camera result directory to preserve historical analysis')
            original_path = source/'protocol.json'
            spec = json.loads(original_path.read_text(encoding='utf-8'))
            if (spec['kind'] != 'oled-layout-comparison' or set(args.layouts) != set(spec['layouts'])
                    or spec['nominal_pixel_pitch_um'] != args.pixel_pitch_um
                    or any(m['assigned_emission_texel_um'] != args.emission_texel_um for m in spec['layouts'].values())
                    or spec['camera'] != {k: v for k, v in CAMERA.items()
                                          if k not in ('geometric_green_pitch_um', 'green_pixel_ISR')}):
                raise ValueError('Reused optics must match the requested layouts, pitch and camera')
            if spec['execution']['solver_sha256'] != hashlib.sha256((ROOT/'scripts/sparkle_multibackend.py').read_bytes()).hexdigest():
                raise ValueError('Reused optical solver differs from the current implementation')
            periods = sorted({tuple(m['green_period_um_xy']) for m in spec['layouts'].values()})
            spec['analysis'] = sampled_camera_plan(periods, spec['camera'])
            spec['optical_root'] = str(source)
            spec['optical_source_protocol_sha256'] = hashlib.sha256(original_path.read_bytes()).hexdigest()
            spec['status'] = 'Camera capture and common spectral mask declared before applying them; previously computed ideal-pupil optical fields reused'
            spec['measurement_execution'] = {'batch_id': uuid4().hex,
                                             'declared_at_utc': datetime.now(timezone.utc).isoformat()}
            spec['reused_files_sha256'] = {}
            for run in spec['runs']:
                for relative in ['run.json']+[f'{b}/{f}' for b in run['backends'] if b != 'mitsuba'
                                               for f in ('brightness.npy', 'stats.json')]:
                    path = source/run['output']/relative
                    spec['reused_files_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            protocol_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
            if args.stage == 'all':
                from reporting import write_sms_report
                print('HTML report:', write_sms_report(args.out))
            else:
                print('Camera protocol frozen; reprocess with generate_report.py --kind sms-reference')
            return
        layouts = {}
        for layout in args.layouts:
            folder = prepare_layout(layout, pixel_pitch_um=args.pixel_pitch_um, texel_um=args.emission_texel_um)
            meta = json.loads((folder/'emission-green.source.json').read_text(encoding='utf-8'))
            layouts[layout] = {**meta, 'emission_path': str(folder/'emission-green.png'),
                               'rgb_path': str(folder/'emission.png')}
        # A common rectangular ROI contains an integer number of ALL exact
        # intensity periods. The union mask is identical for every layout.
        period_texels = [round(v/args.emission_texel_um) for meta in layouts.values()
                         for v in meta['green_period_um_xy']]
        common_period_um = math.lcm(*period_texels)*args.emission_texel_um
        roi_um = math.floor(560/common_period_um)*common_period_um
        if roi_um <= 0:
            raise ValueError('No common complete-period ROI fits within 560 um; reduce pixel pitch')
        periods = sorted({tuple(meta['green_period_um_xy']) for meta in layouts.values()})
        runs = []
        for grid, pitch, backend in (('1um', 1., args.backend), ('0p5um', .5, 'torch')):
            if coarse_only and grid == '0p5um':
                continue
            for layout, meta in layouts.items():
                if computed_layouts is not None and layout not in computed_layouts:
                    continue
                relative = f'{grid}/{layout}'
                command = command_for(args.out/relative, pitch=pitch, backend=backend,
                                      backend_workers=args.backend_workers, device=args.device,
                                      emission=meta['emission_path'], emission_pitch_um=args.emission_texel_um,
                                      emission_scale=meta['linear_emission_scale'], emission_threshold=0.,
                                      height=heights[pitch] if heights else None)
                runs.append({'grid': grid, 'layout': layout, 'pitch_um': pitch, 'output': relative,
                             'backends': list(ALL_BACKENDS) if backend == 'all' else [backend], 'argv': command[2:]})
        height_paths = ({str(path) for path in heights.values()} if heights else
                        {str(folder/'heightfield.png') for folder in DEFAULT_INPUTS.values()})
        analysis = (sampled_camera_plan(periods, CAMERA) if args.measurement == 'sampled-camera' else
                    {'roi_width_um': roi_um, 'camera_pitch_um': CAMERA['object_pitch_um'],
                     'cutoff_um_inv': 1/args.pixel_pitch_um, 'notch_periods_um': periods,
                     'mwa_passes': 1,
                     'primary': 'Identical period-aware three-pass MWA for every layout',
                     'common_mwa': common_mwa_plan(periods, args.pixel_pitch_um),
                     'Fourier': 'Original common reciprocal-lattice mask retained as a diagnostic',
                     'MWA': 'One exact rectangular green-period pass per layout; diagnostic only'})
        spec = {'kind': 'oled-layout-comparison', 'includes_LDF': False,
                'status': 'predeclared before any comparison optical computation',
                'execution': {'batch_id': uuid4().hex, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                              'backend_workers': args.backend_workers,
                              'solver_sha256': hashlib.sha256((ROOT/'scripts/sparkle_multibackend.py').read_bytes()).hexdigest()},
                'camera': {k: v for k, v in CAMERA.items() if k not in ('geometric_green_pitch_um', 'green_pixel_ISR')},
                'nominal_pixel_pitch_um': args.pixel_pitch_um,
                'target_linear_green_mean': 1., 'emission_channel': 'green',
                'height_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in height_paths},
                'layouts': layouts, 'runs': runs,
                'analysis': analysis,
                'limitations': ['Representative aperture polygons, not calibrated commercial panels',
                                'P is nominal logical-address pitch, not equal nearest-neighbor emitter spacing',
                                'One AG realization and one declared relative emitter phase; not a phone-quality ranking'],
                'code_sha256': {name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in
                                ('prepare_tiled_emission.py', 'run_sms_reference.py', 'sms_reference.py', 'reporting.py')}}
        protocol_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.stage == 'prepare':
            for run in runs:
                if run['grid'] == '1um':
                    subprocess.run([sys.executable, str(ROOT/'scripts/sparkle_multibackend.py'),
                                    *run['argv'], '--dry-run'], check=True)
            return
    elif not protocol_path.is_file():
        raise ValueError('Missing comparison protocol; run --stage prepare first')
    spec = json.loads(protocol_path.read_text(encoding='utf-8'))
    if 'optical_root' in spec:
        raise ValueError('This protocol reuses frozen optics; use generate_report.py --kind sms-reference instead of rerunning optical stages')
    if spec['kind'] != 'oled-layout-comparison' or set(args.layouts) != set(spec['layouts']):
        raise ValueError('Requested layouts do not match the saved comparison protocol')
    if spec['execution']['solver_sha256'] != hashlib.sha256((ROOT/'scripts/sparkle_multibackend.py').read_bytes()).hexdigest():
        raise ValueError('Solver changed after the protocol was declared; use a new --out directory')
    for run in spec['runs']:
        if args.stage == 'all' or run['grid'] == {'main': '1um', 'fine': '0p5um'}[args.stage]:
            subprocess.run([sys.executable, str(ROOT/'scripts/sparkle_multibackend.py'), *run['argv']], check=True)
    if args.stage == 'all':
        from reporting import write_sms_report
        print('HTML report:', write_sms_report(args.out))


def surface_cases(args):
    """Declare the two surface experiments without changing their camera metric."""
    seeds = args.study_seeds or ([20261107] if args.surface_study == 'width' else
                                 [20261107, 20261108, 20261109])
    values = ((args.widths_um or [4.8, 6.9, 10., 15., 25., 40.]) if args.surface_study == 'width'
              else (args.min_separation_ratios or [0., .3, .5, .7]))
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Study seeds must be distinct nonnegative integers')
    if len(set(values)) != len(values) or any(not math.isfinite(v) or
            not (0 < v <= 250 if args.surface_study == 'width' else 0 <= v < 1) for v in values):
        raise ValueError('Study values must be distinct; widths in (0,250] um, exclusion ratios in [0,1)')
    return [{'output': f'{args.surface_study}-{value}-seed{seed}'.replace('.', 'p'),
             'spacing_um': value if args.surface_study == 'width' else 6.9,
             'min_separation_ratio': .3 if args.surface_study == 'width' else value,
             'seed': seed, 'fine': seed == seeds[0]}
            for seed in seeds for value in values]


def run_surface_study(args):
    """Reuse height preparation, layout protocols, native solves and SMS reporting."""
    from argparse import Namespace
    from prepare_etched_ag import prepare
    from prepare_tiled_emission import LAYOUTS
    protocol_path = args.out/'protocol.json'
    if args.stage in ('prepare', 'all'):
        if protocol_path.exists():
            raise ValueError('Study protocol already exists; use main/fine or a new --out directory')
        cases = surface_cases(args)
        for case in cases:
            folder = args.out/case['output']
            if folder.exists():
                raise ValueError(f'Study case already exists: {folder}')
        for case in cases:
            folder = args.out/case['output']
            folder.mkdir(parents=True)
            heights = {}
            for pitch, grid in ((1., '1um'), (.5, '0p5um')):
                if pitch == .5 and not case['fine']:
                    continue
                inputs = prepare(folder/'inputs'/grid, pitch_um=pitch,
                                 spacing_um=case['spacing_um'], target_sz_um=1.,
                                 min_separation_ratio=case['min_separation_ratio'], seed=case['seed'])
                heights[pitch] = inputs/'heightfield.png'
            child = Namespace(**{**vars(args), 'out': folder, 'stage': 'prepare',
                                 'layouts': list(LAYOUTS), 'reuse_optics': None})
            run_layout_comparison(child, heights=heights, computed_layouts=['diamond'],
                                  coarse_only=not case['fine'])
            case['protocol_sha256'] = hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()
        spec = {'kind': 'oled-surface-study', 'study': args.surface_study,
                'declared_at_utc': datetime.now(timezone.utc).isoformat(), 'cases': cases,
                'generator_sha256': hashlib.sha256((ROOT/'scripts/prepare_etched_ag.py').read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fixed': '1mm square, reference Sz=1um, rounding sigma=1.2um, profile power=2; full-green Diamond; same camera and four-layout common notch mask',
                'limitations': ['Nominal spacing is a density scale, not actual nearest-neighbor mean',
                                'Sa/Sq, spatial correlations and individual random depths can vary',
                                'Fine grid checks the first seed; source quadrature stays at 2um',
                                'Seed ranges are sample ranges, not confidence intervals or calibrated SMS1000 readings']}
        protocol_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        if args.stage == 'prepare':
            return
    spec = json.loads(protocol_path.read_text(encoding='utf-8'))
    if spec['kind'] != 'oled-surface-study' or spec['study'] != args.surface_study:
        raise ValueError('Requested surface study does not match the saved protocol')
    for case in spec['cases']:
        folder = args.out/case['output']
        if hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest() != case['protocol_sha256']:
            raise ValueError('Case protocol changed after study preparation')
        child = Namespace(**{**vars(args), 'out': folder, 'layouts': list(LAYOUTS)})
        for stage in (('main', 'fine') if args.stage == 'all' else (args.stage,)):
            if stage == 'fine' and not case['fine']:
                continue
            child.stage = stage
            run_layout_comparison(child)
    if args.stage == 'all':
        from reporting import write_sms_report
        print('HTML report:', write_sms_report(args.out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        help='Root for protocol, run directories and reports')
    parser.add_argument('--stage', choices=('prepare', 'main', 'fine', 'old-na', 'all'), required=True,
                        help='prepare: inputs/protocol/dry-run; main/fine/old-na: optical runs; all: complete workflow')
    parser.add_argument('--backend-workers', type=int, default=1,
                        help='Concurrent backend workers; default 1 for sequential timing on one GPU')
    from prepare_tiled_emission import LAYOUTS
    parser.add_argument('--layouts', nargs='+', choices=LAYOUTS,
                        help='Compare equal-pitch layouts through the shared SMS runner; omitted preserves the existing six-run reference')
    parser.add_argument('--pixel-pitch-um', type=float, default=70., help='Comparison nominal logical-address pitch P')
    parser.add_argument('--emission-texel-um', type=float, default=2.)
    parser.add_argument('--measurement', choices=('sampled-camera', 'optical-mwa'), default='sampled-camera',
                        help='Layout comparison: capture camera pixels before documented spectral masking (default); optical-mwa reproduces the historical custom filter')
    parser.add_argument('--reuse-optics', type=Path,
                        help='Layout comparison: verified existing ideal-pupil optical runs; write the new camera analysis to a separate --out')
    parser.add_argument('--backend', choices=('all', 'torch', 'torchoptics', 'chromatix', 'mitsuba'), default='all',
                        help='Comparison main-grid backends; fine-grid check always uses Torch')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='Comparison Torch device')
    parser.add_argument('--surface-study', choices=('width', 'spacing-distribution'),
                        help='Compare AG widths or spacing distributions through the existing sampled-camera workflow')
    parser.add_argument('--widths-um', type=float, nargs='+', help='Width study nominal spacings (default 4.8 6.9 10 15 25 40)')
    parser.add_argument('--min-separation-ratios', type=float, nargs='+', help='Distribution study exclusions (default 0 .3 .5 .7)')
    parser.add_argument('--study-seeds', type=int, nargs='+', help='Width default: 20261107; distribution: 20261107 20261108 20261109')
    args = parser.parse_args()
    if ((args.widths_um and args.surface_study != 'width') or
            (args.min_separation_ratios and args.surface_study != 'spacing-distribution') or
            (args.study_seeds and not args.surface_study)):
        parser.error('Study value/seed options require their corresponding --surface-study')
    if args.surface_study and (args.layouts or args.reuse_optics or args.stage == 'old-na' or
                               args.measurement != 'sampled-camera' or args.backend == 'mitsuba'):
        parser.error('Surface studies require sampled-camera, a wave backend and prepare/main/fine/all; omit --layouts and --reuse-optics')
    if args.reuse_optics and not args.layouts:
        parser.error('--reuse-optics requires --layouts')
    args.out = (args.out or ROOT/'outputs'/('ag-'+args.surface_study if args.surface_study else
                 DEFAULT_CASE+('-sms-layouts' if args.layouts else '-sms-reference'))).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.surface_study:
        run_surface_study(args)
        return
    if args.layouts:
        run_layout_comparison(args)
        return
    if args.stage in ('prepare', 'all'):
        from prepare_etched_ag import prepare
        from prepare_tiled_emission import prepare as prepare_emission
        for pitch, folder in DEFAULT_INPUTS.items():
            if not (folder/'heightfield.png').exists():
                prepare(folder, pitch_um=pitch)
        prepare_emission()
        prepare_green()
        height_meta = json.loads((DEFAULT_INPUTS[1.]/'heightfield.source.json').read_text(encoding='utf-8'))
        spec = {'status': 'predeclared before computing green optical results',
                'execution': {'batch_id': uuid4().hex,
                              'created_at_utc': datetime.now(timezone.utc).isoformat(),
                              'backend_workers': args.backend_workers,
                              'solver_sha256': hashlib.sha256((ROOT / 'scripts/sparkle_multibackend.py').read_bytes()).hexdigest(),
                              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                'camera': CAMERA,
                'height': {'path': str(DEFAULT_INPUTS[1.]/'heightfield.png'),
                           'nominal_site_spacing_um': height_meta['nominal_site_spacing_um'],
                           'calibration': height_meta['calibration']},
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
