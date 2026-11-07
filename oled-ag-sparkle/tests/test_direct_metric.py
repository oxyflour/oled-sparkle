"""Closed-form tests for the user-specified brightness-pixel CV."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sparkle_multibackend import brightness_statistics, collect_analysis_pixels


class DirectBrightnessMetricTest(unittest.TestCase):
    def test_dark_pixels_and_scale_invariance(self):
        image=np.array([[0.,0.],[0.,4.]])
        stats,mask=brightness_statistics(image,0)
        self.assertTrue(mask.all())
        self.assertEqual(stats['num_roi_pixels'],4)
        self.assertEqual(stats['brightness_mean'],1)
        self.assertAlmostEqual(stats['sparkle_cv_percent'],100*np.sqrt(3),places=12)
        scaled,_=brightness_statistics(image*17,0)
        self.assertAlmostEqual(scaled['sparkle_cv_percent'],stats['sparkle_cv_percent'],places=12)

    def test_roi_depends_only_on_geometry(self):
        image=np.pad(np.array([[0.,1.],[1.,2.]]),1,constant_values=100)
        stats,mask=brightness_statistics(image,1)
        self.assertEqual(stats['num_roi_pixels'],4)
        self.assertEqual(stats['roi_bounds_xyxy'],[1,1,3,3])
        self.assertAlmostEqual(stats['sparkle_cv_percent'],100*np.sqrt(.5),places=12)
        self.assertTrue(mask[1,1])
        with self.assertRaises(ValueError): brightness_statistics(np.zeros((4,4)),0)

    def test_1mm_field_collects_exactly_400_pixels_of_50um(self):
        image=np.arange(500*500,dtype=np.float64).reshape(500,500)
        collected,geometry=collect_analysis_pixels(image,2,50)
        np.testing.assert_allclose(collected,image.reshape(20,25,20,25).mean(axis=(1,3)),rtol=1e-14)
        self.assertEqual(collected.shape,(20,20))
        self.assertEqual(geometry['analysis_extent_um_xyxy'],[-500,-500,500,500])
        self.assertEqual(geometry['analysis_source_bounds_px_xyxy'],[0,0,500,500])
        self.assertEqual(geometry['analysis_pixel_area_um2'],2500)
        self.assertAlmostEqual(collected.mean(),image.mean(),places=9)

    def test_detector_pixel_averages_brightness_before_cv(self):
        image=np.tile([[0.,2.],[2.,0.]],(2,2))
        collected,_=collect_analysis_pixels(image,25,50)
        np.testing.assert_array_equal(collected,np.ones((2,2)))
        self.assertEqual(brightness_statistics(collected,0)[0]['sparkle_cv_percent'],0)
        self.assertEqual(brightness_statistics(image,0)[0]['sparkle_cv_percent'],100)

    def test_1um_height_grid_collects_50_by_50_samples_per_pixel(self):
        image=np.arange(1000*1000,dtype=np.float64).reshape(1000,1000)
        collected,geometry=collect_analysis_pixels(image,1,50)
        np.testing.assert_allclose(collected,image.reshape(20,50,20,50).mean(axis=(1,3)),rtol=1e-14)
        self.assertEqual(collected.shape,(20,20))
        self.assertEqual(geometry['analysis_extent_um_xyxy'],[-500,-500,500,500])
        self.assertAlmostEqual(collected.mean(),image.mean(),places=8)

    def test_fractional_overlap_and_centered_full_pixels(self):
        image=np.tile([0.,2.,4.,6.],(4,1))
        collected,geometry=collect_analysis_pixels(image,2,3)
        np.testing.assert_allclose(collected,[[4/3,14/3],[4/3,14/3]],rtol=1e-14)
        self.assertEqual(geometry['analysis_source_bounds_px_xyxy'],[.5,.5,3.5,3.5])
        for pitch,margin in ((0,0),(50,0),(3,-1),(float('nan'),0)):
            with self.assertRaises(ValueError): collect_analysis_pixels(image,2,pitch,margin)
