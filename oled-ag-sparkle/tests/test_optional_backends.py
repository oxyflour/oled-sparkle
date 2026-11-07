"""Opt-in integration checks; require all four backend dependencies.

Set SPARKLE_TEST_OPTIONAL=1 and, if needed, SPARKLE_TEST_MITSUBA_VARIANT
(for example cuda_ad_rgb) before running unittest discovery.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('SPARKLE_TEST_OPTIONAL') == '1',
                     'set SPARKLE_TEST_OPTIONAL=1 to run optional backend integration')
class OptionalBackendsTest(unittest.TestCase):
    def run_all(self, out):
        command = [sys.executable, str(ROOT / 'scripts/run_legacy_demo.py'),
                   '--example', 'quick', '--backend', 'all', '--device', os.environ.get('SPARKLE_TEST_TORCH_DEVICE','cpu'), '--out', str(out),
                   '--extra', '--analysis-pitch-um', '32', '--roi-margin-um', '64',
                   '--backend-workers','2',
                   '--mitsuba-variant', os.environ.get('SPARKLE_TEST_MITSUBA_VARIANT', 'llvm_ad_rgb'),
                   '--progress-every', '0']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wave_backends_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.run_all(out)
            reference = np.load(out / 'torch/analysis_brightness.npy')
            mask = np.load(out / 'torch/roi_mask.npy')
            for backend in ('torchoptics', 'chromatix'):
                with self.subTest(backend=backend):
                    other_mask = np.load(out / backend / 'roi_mask.npy')
                    np.testing.assert_array_equal(mask, other_mask)
                    ratio = np.load(out / backend / 'analysis_brightness.npy')
                    np.testing.assert_allclose(ratio[mask], reference[mask], rtol=1e-3, atol=1e-6)

    def test_all_metrics_equal_the_saved_brightness_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.run_all(out)
            for backend in ('torch', 'torchoptics', 'chromatix', 'mitsuba'):
                with self.subTest(backend=backend):
                    stats = json.loads((out / backend / 'stats.json').read_text(encoding='utf-8'))
                    image=np.load(out / backend / 'analysis_brightness.npy')
                    fine=np.load(out / backend / 'brightness.npy').astype(np.float64)
                    expected=fine[32:96,32:96].reshape(4,16,4,16).mean(axis=(1,3))
                    np.testing.assert_allclose(image,expected,rtol=1e-14)
                    mask=np.load(out / backend / 'roi_mask.npy')
                    values=image[mask].astype(np.float64)
                    self.assertAlmostEqual(stats['sparkle_cv_percent'],100*values.std()/values.mean(),places=10)


if __name__ == '__main__':
    unittest.main()
