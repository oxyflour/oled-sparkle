"""Check physical calibration and reproducibility of the etched surface proxy."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_etched_ag import geometry, prepare, raw_surface


class EtchedAGTests(unittest.TestCase):
    def test_same_geometry_and_scale_at_both_pitches(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = [prepare(Path(folder) / str(p), pitch_um=p, field_um=100., spacing_um=20., target_sa_um=.1)
                       for p in (1., .5)]
            metadata = [json.loads((p / 'heightfield.source.json').read_text()) for p in outputs]
            self.assertEqual((outputs[0] / 'sites.csv').read_bytes(), (outputs[1] / 'sites.csv').read_bytes())
            self.assertEqual(metadata[0]['calibration'], metadata[1]['calibration'])
            self.assertEqual(metadata[0]['height_range_um'], metadata[1]['height_range_um'])
            self.assertAlmostEqual(metadata[1]['measured_surface']['Sa_um'], .1, places=5)
            self.assertLess(abs(metadata[0]['measured_surface']['Sa_um'] - .1), .002)

    def test_png_physical_roundtrip_and_determinism(self):
        with tempfile.TemporaryDirectory() as folder:
            output = prepare(folder, field_um=100., spacing_um=20.)
            png = (output / 'heightfield.png').read_bytes()
            meta = json.loads((output / 'heightfield.source.json').read_text())
            values = np.asarray(Image.open(output / 'heightfield.png'))
            self.assertEqual(values.dtype, np.uint16)
            self.assertEqual(meta['sha256'], hashlib.sha256(png).hexdigest())
            np.testing.assert_allclose(np.load(output / 'height_um.npy'),
                                       values.astype(float) / 65535 * meta['height_range_um'], atol=1e-15)
            self.assertLessEqual(meta['maximum_quantization_error_um'], meta['height_range_um'] / 65535 / 2 + 1e-12)
            prepare(folder, field_um=100., spacing_um=20.)
            self.assertEqual(png, (output / 'heightfield.png').read_bytes())

    def test_invalid_geometry(self):
        for field, spacing in ((100., 0.), (100., 30.), (float('nan'), 20.)):
            with self.assertRaises(ValueError):
                geometry(field, spacing, 1)
        for ratio in (-.1, 1., float('nan')):
            with self.assertRaises(ValueError):
                geometry(100., 20., 1, ratio)

    def test_exclusion_changes_distribution_without_changing_density(self):
        for ratio in (0., .3, .7):
            sites, depths = geometry(100., 20., 20261107, ratio)
            self.assertEqual(len(sites), 25)
            self.assertTrue(np.all((depths >= .7) & (depths <= 1.3)))
            delta = np.abs(sites[:, None] - sites[None, :])
            delta = np.minimum(delta, 100. - delta)
            distance = np.sqrt(np.sum(delta**2, axis=-1))
            np.fill_diagonal(distance, np.inf)
            self.assertGreaterEqual(distance.min(), ratio * 20.)
        with tempfile.TemporaryDirectory() as folder:
            outputs = [prepare(Path(folder)/str(p), pitch_um=p, field_um=100.,
                               spacing_um=20., min_separation_ratio=.7) for p in (1., .5)]
            self.assertEqual((outputs[0]/'sites.csv').read_bytes(), (outputs[1]/'sites.csv').read_bytes())
            meta = json.loads((outputs[1]/'heightfield.source.json').read_text())
            self.assertEqual(meta['minimum_site_separation_um'], 14.)
            self.assertEqual(meta['density_equivalent_spacing_um'], 20.)
            self.assertGreaterEqual(meta['nearest_neighbor_spacing_um']['minimum'], 14.)
            self.assertAlmostEqual(meta['measured_surface']['Sz_um'], 1., places=4)

    def test_sz_normalization_uses_one_post_rounding_reference_for_both_grids(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = [prepare(Path(folder)/str(p), pitch_um=p, field_um=100.,
                               spacing_um=6.9, target_sz_um=1.) for p in (1., .5)]
            metadata = [json.loads((p/'heightfield.source.json').read_text()) for p in outputs]
            self.assertEqual(metadata[0]['calibration'], metadata[1]['calibration'])
            self.assertEqual(metadata[0]['height_range_um'], metadata[1]['height_range_um'])
            self.assertEqual(metadata[1]['calibration']['target_Sz_um'], 1.)
            self.assertNotIn('target_Sa_um', metadata[1]['calibration'])
            self.assertAlmostEqual(metadata[1]['measured_surface']['Sz_um'], 1., places=4)
            self.assertGreater(metadata[1]['height_range_um'], metadata[1]['measured_surface']['Sz_um'])
            self.assertEqual((outputs[0]/'sites.csv').read_bytes(), (outputs[1]/'sites.csv').read_bytes())
            for invalid in (0., -1., float('nan')):
                with self.assertRaises(ValueError):
                    prepare(folder, field_um=100., spacing_um=20., target_sz_um=invalid)
            with self.assertRaises(ValueError):
                prepare(folder, field_um=100., spacing_um=20., target_sa_um=.1, target_sz_um=1.)

    def test_profile_power_changes_bowl_shape_with_the_same_sites_and_depths(self):
        sites, depths = geometry(100., 20., 20261107)
        original, _ = raw_surface(sites, depths, 100., .5, 1.2)
        explicit, _ = raw_surface(sites, depths, 100., .5, 1.2, 2.)
        np.testing.assert_array_equal(original, explicit)
        narrower, _ = raw_surface(sites, depths, 100., .5, 1.2, 4.)
        self.assertGreater(narrower.mean(), original.mean())
        self.assertLess(np.mean(narrower < -.2), np.mean(original < -.2))
        for power in (0., .5, float('nan')):
            with self.assertRaises(ValueError):
                raw_surface(sites, depths, 100., .5, 1.2, power)


if __name__ == '__main__':
    unittest.main()
