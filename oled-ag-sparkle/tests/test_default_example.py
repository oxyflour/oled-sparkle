"""Exercise default entrypoints and retained legacy cases as real CLI runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


class DefaultHeightfieldTests(unittest.TestCase):
    def run_dry(self, script, *extra):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'result'
            command = [sys.executable, str(ROOT / 'scripts' / script),
                       '--device', 'cpu', '--backend', 'torch', '--dry-run',
                       '--out', str(out), *extra]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(run['status'], 'dry_run')
            self.assertTrue((out / 'report.html').is_file())
            return run

    def test_default_entrypoint_uses_etched_height_and_physical_range(self):
        height = ROOT / 'assets/oled-1mm-etched-ag-spacing6p9-sz1um/heightfield.png'
        run = self.run_dry('run_oled_demo.py')
        self.assertEqual(Path(run['parameters']['height']), height)
        self.assertEqual(run['input_files']['height']['sha256'], hashlib.sha256(height.read_bytes()).hexdigest())
        self.assertEqual(run['input_files']['height']['png_bits_per_channel'], 16)
        self.assertEqual(run['effective_model']['height_grid'], [1000, 1000])
        self.assertAlmostEqual(run['parameters']['height_range_um'], 1.04)
        meta = run['input_files']['height']['provenance']
        self.assertEqual(meta['nominal_site_spacing_um'], 6.9)
        self.assertEqual(meta['calibration']['target_Sz_um'], 1.)
        self.assertAlmostEqual(run['effective_model']['height_max_um']-run['effective_model']['height_min_um'],
                               .9975812, places=6)

    def test_finer_grid_preserves_field_height_scale_emission_and_measurement_pixels(self):
        run = self.run_dry('run_oled_demo.py', '--pitch-um', '.5')
        self.assertEqual(run['effective_model']['height_grid'], [2000, 2000])
        self.assertEqual(run['effective_model']['field_width_um'], 1000)
        self.assertAlmostEqual(run['parameters']['height_range_um'], 1.04)
        self.assertIn('spacing6p9-sz1um-pitch0p5', run['parameters']['height'])
        self.assertEqual(run['parameters']['emission_pitch_um'], 2)
        self.assertEqual(run['parameters']['analysis_pitch_um'], 50)
        self.assertEqual(run['parameters']['pad_px'], 280)

    def test_haze_default_uses_the_same_new_surface_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, str(ROOT/'scripts/calculate_haze.py'), '--out', folder]
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            record = json.loads((Path(folder)/'haze.json').read_text('utf-8'))['records'][0]
            self.assertIn('spacing6p9-sz1um', record['input']['path'])
            self.assertEqual(record['input']['height_range_um'], 1.04)
            self.assertAlmostEqual(record['haze_percent'], 41.20671869857012, places=5)

    def test_sms_layout_prepare_records_the_new_default_inputs(self):
        from run_sms_reference import run_layout_comparison
        from prepare_etched_ag import DEFAULT_INPUTS
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(out=Path(folder), stage='prepare', layouts=['diamond', 'rgb-stripe', 'rgb-delta', 'windmill'],
                                   reuse_optics=None, measurement='sampled-camera', pixel_pitch_um=70.,
                                   emission_texel_um=2., backend='torch', backend_workers=1, device='cpu')
            with patch('prepare_tiled_emission.prepare_layout',
                       side_effect=lambda name, **kwargs: ROOT/'assets/oled-layouts'/name), \
                 patch('run_sms_reference.subprocess.run'):
                run_layout_comparison(args)
            spec = json.loads((Path(folder)/'protocol.json').read_text('utf-8'))
            self.assertEqual(set(spec['height_sha256']), {str(p/'heightfield.png') for p in DEFAULT_INPUTS.values()})
            for run in spec['runs']:
                argv = run['argv']
                self.assertEqual(argv[argv.index('--height')+1], str(DEFAULT_INPUTS[run['pitch_um']]/'heightfield.png'))
                self.assertEqual(float(argv[argv.index('--height-range-um')+1]), 1.04)
            self.assertIsNone(spec['analysis']['additional_lowpass'])

    def test_sms_reference_prepare_preserves_existing_height_assets(self):
        from run_sms_reference import main
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sys, 'argv', ['run_sms_reference.py', '--stage', 'prepare', '--out', folder]), \
                 patch('prepare_etched_ag.prepare') as generate, \
                 patch('prepare_tiled_emission.prepare'), patch('run_sms_reference.prepare_green'), \
                 patch('run_sms_reference.subprocess.run'):
                main()
            generate.assert_not_called()
            height = json.loads((Path(folder)/'protocol.json').read_text('utf-8'))['height']
            self.assertEqual(height['nominal_site_spacing_um'], 6.9)
            self.assertEqual(height['calibration']['target_Sz_um'], 1.)

    def test_circular_pits_remain_an_explicit_legacy_option(self):
        run = self.run_dry('run_oled_demo.py', '--height-surface', 'random-pits')
        self.assertIn('oled-1mm-pits3to5um', run['parameters']['height'])
        self.assertEqual(run['parameters']['height_range_um'], .5)

    def test_quick_fixture_remains_explicit(self):
        run = self.run_dry('run_legacy_demo.py', '--example', 'quick')
        self.assertIn('demo_quick', run['parameters']['height'])
        self.assertEqual(run['effective_model']['height_grid'], [128, 128])

    def test_historical_runner_requires_an_explicit_fixture(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_legacy_demo.py'), '--dry-run'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--example', result.stderr)


if __name__ == '__main__':
    unittest.main()
