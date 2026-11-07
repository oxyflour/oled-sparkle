"""Exercise default entrypoints and retained legacy cases as real CLI runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


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
        height = ROOT / 'assets/oled-1mm-etched-ag/heightfield.png'
        run = self.run_dry('run_oled_demo.py')
        self.assertEqual(Path(run['parameters']['height']), height)
        self.assertEqual(run['input_files']['height']['sha256'], hashlib.sha256(height.read_bytes()).hexdigest())
        self.assertEqual(run['input_files']['height']['png_bits_per_channel'], 16)
        self.assertEqual(run['effective_model']['height_grid'], [1000, 1000])
        self.assertAlmostEqual(run['parameters']['height_range_um'], .5361845861372564)
        self.assertAlmostEqual(run['effective_model']['height_max_um']-run['effective_model']['height_min_um'],
                               .5150186515322807, places=6)

    def test_finer_grid_preserves_field_height_scale_emission_and_measurement_pixels(self):
        run = self.run_dry('run_oled_demo.py', '--pitch-um', '.5')
        self.assertEqual(run['effective_model']['height_grid'], [2000, 2000])
        self.assertEqual(run['effective_model']['field_width_um'], 1000)
        self.assertAlmostEqual(run['parameters']['height_range_um'], .5361845861372564)
        self.assertEqual(run['parameters']['emission_pitch_um'], 2)
        self.assertEqual(run['parameters']['analysis_pitch_um'], 50)
        self.assertEqual(run['parameters']['pad_px'], 280)

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
