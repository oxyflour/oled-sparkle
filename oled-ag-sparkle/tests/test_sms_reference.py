"""Validate periodic separation without tuning against a measured target."""
from pathlib import Path
import sys
import unittest
import hashlib
import json
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sms_reference import (analyze, box_kernel, flat_wave_control, fourier_separate,
                           measurement_statistics, moving_average, analyze_saved_runs)
from sparkle_multibackend import model_from_args, parse_args, simulate_torch_wave
from run_sms_reference import CAMERA, command_for
from run_oled_demo import build_command


class SMSReferenceTests(unittest.TestCase):
    def test_saved_report_rejects_a_run_before_its_declared_batch(self):
        solver_sha = hashlib.sha256((ROOT / 'scripts/sparkle_multibackend.py').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / '1um').mkdir()
            (base / 'protocol.json').write_text(json.dumps({
                'includes_LDF': False,
                'execution': {'created_at_utc': '2026-11-07T12:00:00+00:00',
                              'solver_sha256': solver_sha, 'backend_workers': 1}}), encoding='utf-8')
            (base / '1um/run.json').write_text(json.dumps({
                'status': 'complete', 'parameters': {'emission_channel': 'green', 'backend_workers': 1},
                'solver': {'sha256': solver_sha}, 'started_at_utc': '2026-11-07T11:00:00+00:00'}), encoding='utf-8')
            with self.assertRaises(AssertionError):
                analyze_saved_runs(base)

    def test_saved_report_rejects_a_different_solver_version(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / 'protocol.json').write_text(json.dumps({
                'includes_LDF': False,
                'execution': {'created_at_utc': '2026-11-07T12:00:00+00:00',
                              'solver_sha256': 'old-solver-version'}}), encoding='utf-8')
            with self.assertRaises(AssertionError):
                analyze_saved_runs(base)

    def test_shared_runner_preserves_distinct_green_measurement_conditions(self):
        raw = parse_args(build_command(device='cpu')[2:])
        for pitch, pad, batch in ((1., 140, 16), (.5, 280, 8)):
            with self.subTest(pitch=pitch):
                green = parse_args(command_for(ROOT / 'outputs/check', pitch=pitch)[2:])
                self.assertEqual(green.emission_channel, 'green')
                self.assertEqual(green.analysis_pitch_um, 26.25)
                self.assertEqual(green.roi_margin_um, 220)
                self.assertEqual(green.na, CAMERA['object_NA'])
                self.assertEqual(green.height_pitch_um, pitch)
                self.assertEqual(green.pad_px, pad)
                self.assertEqual(green.batch, batch)
                for key in ('gap_um', 'wavelength_nm', 'n_stack', 'n_out', 'emission_pitch_um',
                            'height_range_um', 'ray_radial', 'ray_angular'):
                    self.assertEqual(getattr(raw, key), getattr(green, key), key)
        self.assertEqual(raw.analysis_pitch_um, 50)
        self.assertEqual(raw.na, .08)
        self.assertEqual(raw.emission_channel, 'luma')

    def test_lattice_rejected_and_random_off_lattice_retained(self):
        n = 560
        xx = np.arange(n)[None, :]
        yy = np.arange(n)[:, None]
        random_signal = .04 * np.cos(2*np.pi*(3*xx+2*yy)/n)
        lattice = .6 * np.cos(2*np.pi*xx/140) + .2 * np.sin(2*np.pi*yy/140)
        image = 1 + lattice + random_signal
        result, mask, meta = fourier_separate(image, 1.)
        np.testing.assert_allclose(result, 1 + random_signal, atol=1e-13)
        self.assertTrue(mask[0, 0])
        self.assertAlmostEqual(result.mean(), image.mean(), places=14)
        self.assertEqual(len(meta['notched_lattice_bins_yx']), 8)

    def test_constant_zero_and_scale_invariance(self):
        full = np.ones((1000, 1000))
        result = analyze(full)
        self.assertAlmostEqual(result['summary']['primary']['sparkle_contrast_percent'], 0, places=10)
        self.assertEqual(result['measurement_pixels'].shape, (21, 21))
        rng = np.random.default_rng(31)
        signal = 1 + .01 * rng.normal(size=(560, 560))
        filtered, _, _ = fourier_separate(signal, 1.)
        _, original = measurement_statistics(filtered, 1., 26.25)
        _, scaled = measurement_statistics(filtered * 17, 1., 26.25)
        self.assertAlmostEqual(original['sparkle_contrast_percent'], scaled['sparkle_contrast_percent'], places=12)
        signed = np.array([[-1., 3.], [-1., 3.]])
        pixels, stats = measurement_statistics(signed, 1., 1.)
        np.testing.assert_allclose(pixels, signed)
        self.assertAlmostEqual(stats['sparkle_contrast_percent'], 200.)
        self.assertEqual(stats['negative_filtered_measurement_pixels'], 2)

    def test_mwa_removes_exact_supercell(self):
        n = 1000
        line = 1 + .4 * np.cos(2*np.pi*(np.arange(n)+.5)/140)
        values = np.broadcast_to(line, (n, n))
        out = moving_average(values, 1.)[220:780, 220:780]
        np.testing.assert_allclose(out, 1., atol=1e-13)
        for width in (1, 3, 3.5, 140):
            kernel = box_kernel(width)
            self.assertAlmostEqual(kernel.sum(), 1.)
            np.testing.assert_allclose(kernel, kernel[::-1])

    def test_reject_noninteger_cell_crop(self):
        with self.assertRaises(ValueError):
            fourier_separate(np.ones((500, 500)), 1.)

    def test_green_input_preserves_original_G(self):
        original = np.asarray(Image.open(ROOT / 'assets/oled-1mm-periodic/emission.png'))
        green = np.asarray(Image.open(ROOT / 'assets/oled-1mm-periodic-green/emission.png'))
        np.testing.assert_array_equal(original[..., 1], green[..., 1])
        self.assertEqual(np.count_nonzero(green[..., (0, 2)]), 0)

    def test_fast_flat_control_matches_wave_propagation(self):
        args = parse_args(['--height', str(ROOT / 'assets/demo_quick/height_flat.png'),
                           '--height-pitch-um', '2', '--height-range-um', '.5',
                           '--emission', str(ROOT / 'assets/demo_quick/emission.png'),
                           '--emission-pitch-um', '20', '--device', 'cpu',
                           '--pad-px', '12', '--batch', '4', '--progress-every', '0'])
        model = model_from_args(args)
        model.h_m = np.zeros((24, 24), dtype=np.float32)
        model.emission = np.array([[0., .3], [.5, 1.]], dtype=np.float32)
        model.em_dx_m = 10e-6
        computed, _ = simulate_torch_wave(model, 'torch', 10)
        control = flat_wave_control(model)
        error = np.linalg.norm(computed-control) / np.linalg.norm(control)
        self.assertLess(error, 3e-5)


if __name__ == '__main__':
    unittest.main()
