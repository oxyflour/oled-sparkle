"""Physical invariants and analytical diffraction checks for AG haze."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.special import jv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from calculate_haze import load_surface, transmission_haze, solve_wave_haze, solve_mitsuba_haze


class HazeTests(unittest.TestCase):
    def test_flat_and_index_matched_surfaces_have_zero_haze(self):
        rng = np.random.default_rng(41)
        for h, n_in in ((np.full((64, 64), 2e-6), 1.5),
                         (rng.normal(0, 1e-7, (64, 64)), 1.)):
            r, curve = transmission_haze(h, 1e-6, n_in=n_in)
            self.assertLess(r['haze_percent'], 1e-25)
            self.assertLess(r['parseval_relative_error'], 1e-14)
            self.assertAlmostEqual(curve[-1]['within_cone_percent'], 100.)

    def test_blazed_phase_grating_is_classified_by_its_physical_angle(self):
        x = np.arange(256)*.25e-6
        for period, expected in ((4e-6, 100.), (32e-6, 0.)):
            h = (530e-9/.5)*(x % period)/period
            r, _ = transmission_haze(np.broadcast_to(h, (64, 256)), .25e-6)
            self.assertAlmostEqual(r['haze_percent'], expected, places=10)

    def test_sinusoidal_phase_grating_matches_bessel_order_flux(self):
        wavelength, beta, period = 530e-9, .7, 16e-6
        x = np.arange(512)*.25e-6
        h = beta*wavelength/(2*np.pi*.5)*np.cos(2*np.pi*x/period)
        r, curve = transmission_haze(np.broadcast_to(h, (64, 512)), .25e-6)
        orders = np.arange(-30, 31)
        sine = wavelength*orders/period
        weights = jv(orders, beta)**2*np.sqrt(np.maximum(0., 1-sine*sine))
        expected = 100*weights[np.abs(sine) > np.sin(np.radians(2.5))].sum()/weights.sum()
        self.assertAlmostEqual(r['haze_percent'], expected, places=11)
        row = next(c for c in curve if c['half_angle_deg'] == 2.5)
        self.assertAlmostEqual(row['outside_cone_percent'], r['haze_percent'], places=11)
        self.assertTrue(all(a['outside_cone_percent'] >= b['outside_cone_percent']-1e-12
                            for a, b in zip(curve, curve[1:])))

    def test_same_roughness_at_different_lateral_periods_has_different_haze(self):
        x = np.arange(512)*.25e-6
        signals = [np.broadcast_to(80e-9*np.cos(2*np.pi*x/p), (64, 512)) for p in (8e-6, 32e-6)]
        self.assertAlmostEqual(signals[0].std(), signals[1].std(), places=18)
        values = [transmission_haze(s, .25e-6)[0]['haze_percent'] for s in signals]
        self.assertGreater(values[0], 100*values[1])

    def test_height_offset_sign_and_periodic_translation_leave_haze_unchanged(self):
        rng = np.random.default_rng(4)
        h = rng.normal(0, 80e-9, (64, 64))
        reference = transmission_haze(h, 1e-6)[0]['haze_percent']
        for other in (h+20e-6, -h, np.roll(h, (7, 3), (0, 1))):
            value = transmission_haze(other, 1e-6)[0]['haze_percent']
            self.assertAlmostEqual(value, reference, places=10)

    def test_reject_unresolved_cone_or_invalid_parameters(self):
        for options in ({'pitch_m': 10e-6}, {'acceptance_deg': 0},
                         {'wavelength_m': -1}, {'n_out': float('nan')}):
            args = {'pitch_m': 1e-6, **options}
            with self.assertRaises(ValueError):
                transmission_haze(np.zeros((32, 32)), **args)

    def test_loader_requires_calibration_periodicity_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'height.png'
            Image.fromarray(np.zeros((16, 16), dtype=np.uint16)).save(p)
            with self.assertRaises(ValueError):
                load_surface(p)
            with self.assertRaises(ValueError):
                load_surface(p, 1., .5)
            height, pitch, _ = load_surface(p, 1., .5, True)
            self.assertEqual(height.shape, (16, 16))
            self.assertEqual(pitch, 1.)
            meta = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                    'assigned_texel_um': 1., 'height_range_um': .5, 'periodic_domain_um': 16.}
            p.with_suffix('.source.json').write_text(json.dumps(meta), encoding='utf-8')
            load_surface(p)
            with self.assertRaises(ValueError):
                load_surface(p, height_range_um=.6)
            meta['sha256'] = 'incorrect'
            p.with_suffix('.source.json').write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_surface(p)


@unittest.skipUnless(os.environ.get('SPARKLE_TEST_HAZE') == '1',
                     'set SPARKLE_TEST_HAZE=1 for native four-backend haze validation')
class NativeHazeTests(unittest.TestCase):
    def test_native_wave_libraries_match_analytic_diffraction_orders(self):
        wavelength, beta, period = 530e-9, .7, 16e-6
        x = np.arange(128)*.5e-6
        h = np.broadcast_to(beta*wavelength/(2*np.pi*.5)*np.cos(2*np.pi*x/period), (32, 128)).copy()
        orders = np.arange(-30, 31)
        sine = wavelength*orders/period
        weights = jv(orders, beta)**2*np.sqrt(np.maximum(0., 1-sine*sine))
        expected = 100*weights[np.abs(sine) > np.sin(np.radians(2.5))].sum()/weights.sum()
        for backend in ('torch', 'torchoptics', 'chromatix'):
            with self.subTest(backend=backend):
                power, context = solve_wave_haze(h, .5e-6, wavelength, 1.5, 1., backend)
                r, _ = transmission_haze(h, .5e-6, angular_power=power)
                self.assertAlmostEqual(r['haze_percent'], expected, places=10)
                self.assertIn('native_operations', context)
                if backend == 'chromatix':
                    self.assertEqual(context['default_backend'], 'gpu')

    def test_angular_power_is_invariant_under_native_output_propagation(self):
        x = np.arange(64)*1e-6
        h = np.broadcast_to(100e-9*np.cos(2*np.pi*x/16e-6), (32, 64)).copy()
        for backend in ('torch', 'torchoptics'):
            a, _ = solve_wave_haze(h, 1e-6, 530e-9, 1.5, 1., backend, propagation_um=25.)
            b, _ = solve_wave_haze(h, 1e-6, 530e-9, 1.5, 1., backend, propagation_um=100.)
            np.testing.assert_allclose(a, b, atol=1e-14)

    def test_mitsuba_matches_projected_facet_snell_fresnel_integral(self):
        x = np.arange(64)*1e-6
        profile = .5e-6*np.cos(2*np.pi*x/16e-6)
        h = np.broadcast_to(profile, (32, 64)).copy()
        slopes = (np.roll(profile, -1)-profile)/1e-6
        incident = np.arctan(np.abs(slopes))
        transmitted = np.arcsin(1.5*np.sin(incident))
        angles = transmitted-incident
        ci, ct = np.cos(incident), np.cos(transmitted)
        rs = ((1.5*ci-ct)/(1.5*ci+ct))**2
        rp = ((ci-1.5*ct)/(ci+1.5*ct))**2
        weights = 1-(rs+rp)/2
        expected = 100*weights[angles > np.radians(2.5)].sum()/weights.sum()
        r, _, _ = solve_mitsuba_haze(h, 1e-6, 530e-9, 1.5, 1., 2.5)
        self.assertAlmostEqual(r['haze_percent'], expected, places=5)
        self.assertLess(r['energy_balance_error'], 1e-6)
        flat, _, _ = solve_mitsuba_haze(np.zeros_like(h), 1e-6, 530e-9, 1.5, 1., 2.5)
        self.assertAlmostEqual(flat['transmission_fraction'], .96, places=6)
        self.assertEqual(flat['haze_percent'], 0.)


if __name__ == '__main__':
    unittest.main()
