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
                           measurement_statistics, moving_average, analyze_saved_runs,
                           common_mwa_plan, measure_mwa, sampled_camera_plan,
                           sample_camera, sampled_periodic_mask, analyze_sampled_camera,
                           sampled_statistics, circular_pupil_mtf, periodic_camera_control)
from sparkle_multibackend import model_from_args, parse_args, simulate_torch_wave
from run_sms_reference import CAMERA, command_for
from run_oled_demo import build_command


class SMSReferenceTests(unittest.TestCase):
    def test_sampled_camera_removes_periodic_aliases_after_capture(self):
        plan = sampled_camera_plan([(70, 70), (140, 140), (210, 140)], CAMERA)
        yy, xx = np.indices((1000, 1000))
        # 2/70 exceeds detector Nyquist but is transmitted by the lens.
        signal = 1+.3*np.cos(2*np.pi*(xx+.5)*2/70)+.2*np.sin(2*np.pi*(yy+.5)/140)
        result = analyze_sampled_camera(signal, 1., plan)
        self.assertGreater(result['summary']['direct_unseparated_camera_CV_percent'], 1.)
        # Piecewise-constant reconstruction on the 1 um optical grid adds tiny
        # sampling replicas; the ideal periodic null is tested separately.
        self.assertLess(result['summary']['primary']['sparkle_contrast_percent'], .002)
        self.assertEqual(result['camera_pixels'].shape, (32, 32))
        self.assertTrue(any(h['aliased'] for h in result['summary']['primary_filter']['harmonics']))
        self.assertAlmostEqual(result['summary']['primary']['mean_preservation_error'], 0., places=14)

    def test_sampled_notches_retain_random_band_and_have_hermitian_symmetry(self):
        plan = sampled_camera_plan([(70, 70), (140, 140), (210, 140)], CAMERA)
        mask, meta = sampled_periodic_mask((32, 32), 26.25, plan['notch_periods_um_xy'],
                                           plan['optical_intensity_cutoff_cycles_per_um'])
        yy, xx = np.indices(mask.shape)
        np.testing.assert_array_equal(mask, mask[(-yy) % 32, (-xx) % 32])
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[0, (-8) % 32])  # aliased 2/70 cycles/um
        image = 1+.2*np.cos(2*np.pi*(xx+yy)/32)
        np.testing.assert_allclose(np.fft.ifft2(np.fft.fft2(image)*mask).real, image, atol=1e-14)
        self.assertEqual(meta['DC_aliases_unidentifiable'], [])
        with self.assertRaises(ValueError):
            sampled_periodic_mask((21, 21), 26.25, [(210, 140)], .03)
        with self.assertRaises(ValueError):
            sampled_camera_plan([(150, 150)], CAMERA)

    def test_camera_pixel_aperture_matches_sinc_including_above_nyquist(self):
        pitch, frequency, amplitude = .5, 2/70, .4
        x = (np.arange(2000)+.5)*pitch
        image = np.broadcast_to(1+amplitude*np.cos(2*np.pi*frequency*x), (2000, 2000))
        pixels, geom = sample_camera(image, pitch, 840., 26.25)
        centers = 80+(np.arange(32)+.5)*26.25
        expected = 1+amplitude*np.sinc(frequency*26.25)*np.cos(2*np.pi*frequency*centers)
        quadrature_bound = amplitude*(np.pi*frequency*pitch)**2/6
        np.testing.assert_allclose(pixels, np.broadcast_to(expected, pixels.shape), atol=quadrature_bound)
        self.assertEqual(geom['analysis_extent_um_xyxy'], [-420., -420., 420., 420.])
        _, shifted = sample_camera(image, pitch, 840., 26.25, (7., 0.))
        self.assertEqual(shifted['analysis_extent_um_xyxy'], [-413., -420., 427., 420.])
        with self.assertRaises(ValueError):
            sample_camera(image, pitch, 840., 26.25, (.1, 0.))

    def test_camera_no_second_integration_or_signed_signal_clipping(self):
        stats = sampled_statistics(np.array([[-1., 3.], [-1., 3.]]))
        self.assertEqual(stats['num_pixels'], 4)
        self.assertEqual(stats['negative_filtered_measurement_pixels'], 2)
        self.assertAlmostEqual(stats['sparkle_contrast_percent'], 200.)
        np.testing.assert_allclose(circular_pupil_mtf(np.array([0., .5, 1., 2.]), 1.),
                                   [1., .3910022189557706, 0., 0.], atol=1e-14)

    def test_periodic_flat_camera_control_has_zero_post_notch_contrast(self):
        from types import SimpleNamespace
        yy, xx = np.indices((500, 500))
        emission = 1+.4*np.cos(2*np.pi*xx*2/210)+.2*np.sin(2*np.pi*yy*2/140)
        model = SimpleNamespace(emission=emission, em_dx_m=2e-6)
        plan = sampled_camera_plan([(70, 70), (140, 140), (210, 140)], CAMERA)
        pixels, stats = periodic_camera_control(model, plan)
        self.assertGreater(pixels.std(), .01)
        self.assertLess(stats['sparkle_contrast_percent'], 1e-11)

    def test_common_three_pass_mwa_cancels_all_periods_with_legacy_support(self):
        plan = common_mwa_plan([(70, 70), (140, 140), (210, 140)])
        self.assertEqual(plan['windows_um_xy'], [[70, 140], [140, 140], [210, 140]])
        self.assertEqual(plan['total_support_width_um_xy'], [420, 420])
        self.assertEqual(plan['roi_width_um'], 560)
        yy, xx = np.indices((1000, 1000))
        signal = 1+.3*np.cos(2*np.pi*xx/210)+.2*np.sin(2*np.pi*yy/140)+.1*np.cos(2*np.pi*xx/140)
        _, pixels, stats = measure_mwa(signal, 1., windows_um_xy=plan['windows_um_xy'])
        np.testing.assert_allclose(pixels, 1., atol=1e-13)
        self.assertEqual(stats['analysis_grid'], [21, 21])

    def test_explicit_three_identical_windows_preserve_legacy_mwa(self):
        rng = np.random.default_rng(22)
        signal = 1+rng.random((128, 128))
        expected = moving_average(signal, 1., width_um=14., passes=3)
        actual = moving_average(signal, 1., windows_um_xy=[[14., 14.]]*3)
        np.testing.assert_array_equal(actual, expected)

    def test_common_union_mask_removes_distinct_rectangular_lattices(self):
        yy, xx = np.indices((420, 420))
        signal = .04*np.cos(2*np.pi*(xx+yy)/420)
        lattice = .5*np.cos(2*np.pi*xx/140)+.2*np.cos(2*np.pi*xx/210)+.3*np.sin(2*np.pi*yy/140)
        previous = None
        for period in ((70, 70), (140, 140), (210, 140)):
            result, mask, _ = fourier_separate(1+signal+lattice, 1., supercell_um=period,
                                               notch_periods_um=[(70, 70), (140, 140), (210, 140)])
            np.testing.assert_allclose(result, 1+signal, atol=1e-13)
            if previous is not None:
                np.testing.assert_array_equal(mask, previous)
            previous = mask
        with self.assertRaises(ValueError):
            fourier_separate(np.ones((560, 560)), 1., notch_periods_um=[(210, 140)])

    def test_rectangular_mwa_uses_each_axis_period(self):
        yy, xx = np.indices((840, 840))
        image = 1+.4*np.cos(2*np.pi*xx/210)+.2*np.sin(2*np.pi*yy/140)
        result = moving_average(image, 1., width_um=(210, 140), passes=1)
        np.testing.assert_allclose(result[210:630, 210:630], 1., atol=1e-13)

    def test_solver_applies_gain_after_linear_decoding(self):
        args = parse_args(build_command(device='cpu')[2:])
        base = model_from_args(args).emission
        args.emission_scale = 7.
        scaled = model_from_args(args).emission
        np.testing.assert_allclose(scaled, base*7, rtol=1e-7)
        self.assertGreater(scaled.max(), 1.)
        args.emission_scale = -1.
        with self.assertRaises(ValueError):
            model_from_args(args)

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
