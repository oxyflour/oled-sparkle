"""Physical geometry and polarity checks for the replacement height surface."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_random_pits import random_pits, rasterize


class RandomPitTest(unittest.TestCase):
    def test_target_coverage_diameters_boundaries_and_nonoverlap(self):
        pits, _ = random_pits(field_um=100., coverage=.5, seed=20261107)
        radii = pits[:, 2]/2
        self.assertTrue(np.all((pits[:, 2] >= 3) & (pits[:, 2] <= 5)))
        self.assertTrue(np.all(pits[:, :2] >= radii[:, None]))
        self.assertTrue(np.all(pits[:, :2]+radii[:, None] <= 100))
        area_fraction = np.pi*np.sum(radii**2)/100**2
        self.assertGreaterEqual(area_fraction, .5)
        self.assertLess(area_fraction, .5+np.pi*2.5**2/100**2)
        distances = np.linalg.norm(pits[:, None, :2]-pits[None, :, :2], axis=-1)
        np.fill_diagonal(distances, np.inf)
        self.assertTrue(np.all(distances >= radii[:, None]+radii[None, :]))
        again, _ = random_pits(field_um=100., coverage=.5, seed=20261107)
        np.testing.assert_array_equal(pits, again)

    def test_pit_is_a_depression_with_fixed_absolute_depth_and_smooth_rim(self):
        # Center lands on a sample, with samples exactly on and outside the rim.
        image = rasterize(np.array([[4.5, 4.5, 4.]]), field_um=10., pitch_um=1.)
        self.assertEqual(image.dtype, np.uint16)
        self.assertEqual(image[4, 4], 0)
        self.assertEqual(image[4, 6], 65535)
        self.assertEqual(image[4, 7], 65535)
        self.assertAlmostEqual(image[4, 5]/65535, 1-(1-.25)**2, delta=1/65535)
        np.testing.assert_array_equal(image[4, 2:7], image[2:7, 4])


if __name__ == '__main__':
    unittest.main()
