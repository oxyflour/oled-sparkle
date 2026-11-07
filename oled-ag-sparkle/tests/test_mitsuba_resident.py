"""Opt-in CUDA regression against the original host-side Mitsuba integrator.

Run with SPARKLE_TEST_MITSUBA=1, or alongside SPARKLE_TEST_OPTIONAL=1.
"""
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import sparkle_multibackend as s


@unittest.skipUnless(os.environ.get('SPARKLE_TEST_MITSUBA')=='1' or
                     os.environ.get('SPARKLE_TEST_OPTIONAL')=='1',
                     'set SPARKLE_TEST_MITSUBA=1 for CUDA Mitsuba regression')
class MitsubaResidentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mitsuba as mi
        import drjit as dr
        variant=os.environ.get('SPARKLE_TEST_MITSUBA_VARIANT','cuda_ad_rgb')
        if not variant.startswith('cuda'):
            raise unittest.SkipTest('resident integrator requires a CUDA variant')
        mi.set_variant(variant)
        cls.mi,cls.dr,cls.variant=mi,dr,variant

    def model(self,**changes):
        y,x=np.mgrid[:32,:40]
        height=(.2*np.sin(x*.37)*np.cos(y*.29)).astype(np.float32)*1e-6
        emission=np.random.default_rng(197).uniform(.1,1.,(10,12)).astype(np.float32)
        m=s.Model(h_m=height,emission=emission,dx_m=2e-6,em_dx_m=8e-6,
                  wavelength_m=530e-9,gap_m=20e-6,n_stack=1.5,n_out=1.,na=.08,
                  source_samples=1,margin_m=0.,offset_x_m=0.,offset_y_m=0.,
                  threshold=.001,pad=0,batch=2,display_pitch_m=16e-6,
                  roi_margin_m=0.,device='cuda',ray_radial=3,ray_angular=8,
                  ray_chunk=257,mesh_stride=1,mitsuba_variant=self.variant,progress_every=0)
        return replace(m,**changes)

    def compare_original(self,m):
        mesh,_,_=s.create_mitsuba_mesh(self.mi,m)
        scene=self.mi.load_dict({'type':'scene','ag_surface':mesh})
        reference,nref=s._simulate_mitsuba_numpy(self.mi,scene,m)
        image,n=s._simulate_mitsuba_resident(self.mi,self.dr,scene,m)
        self.assertEqual(n,nref)
        self.assertEqual(n,np.prod(m.shape)*m.ray_radial*m.ray_angular)
        self.assertEqual(image.dtype,np.float32)
        self.assertEqual(image.shape,m.shape)
        self.assertTrue(np.isfinite(image).all())
        self.assertGreaterEqual(image.min(),0)
        self.assertGreater(np.count_nonzero(reference),0)
        np.testing.assert_allclose(image,reference,rtol=3e-5,atol=1e-6)
        return image

    def test_flat_uniform_matches_fresnel_transmission(self):
        m=self.model(h_m=np.zeros((32,40),np.float32),emission=np.ones((10,12),np.float32))
        image=self.compare_original(m)
        transmission=[]
        for tx,ty in s.pupil_nodes(m.ray_radial,m.ray_angular,m.na,m.n_out):
            ci=math.sqrt(1-tx*tx-ty*ty)
            ct=math.sqrt(1-(m.n_out/m.n_stack)**2*(tx*tx+ty*ty))
            rs=((m.n_out*ci-m.n_stack*ct)/(m.n_out*ci+m.n_stack*ct))**2
            rp=((m.n_stack*ci-m.n_out*ct)/(m.n_stack*ci+m.n_out*ct))**2
            transmission.append(1-.5*(rs+rp))
        np.testing.assert_allclose(image[2:-2,2:-2],np.mean(transmission),rtol=2e-6)

    def test_rough_surface_and_piecewise_constant_emission(self):
        self.compare_original(self.model())

    def test_outside_emission_and_total_internal_reflection(self):
        y,x=np.mgrid[:32,:40]
        m=self.model(h_m=(4*np.sin(x*.7)*np.cos(y*.8)).astype(np.float32)*1e-6,
                     emission=np.ones((6,6),np.float32),em_dx_m=5e-6,
                     n_stack=1.,n_out=1.5,na=.45)
        image=self.compare_original(m)
        self.assertGreater(np.count_nonzero(image==0),0)

    def test_zero_emission_and_missed_surface_stay_finite(self):
        m=self.model(emission=np.zeros((10,12),np.float32),gap_m=500e-6)
        image,n=s.simulate_mitsuba(m)
        np.testing.assert_array_equal(image,np.zeros(m.shape,np.float32))
        self.assertEqual(n,np.prod(m.shape)*24)

    def test_chunk_size_and_mesh_stride_preserve_sampling(self):
        m=self.model(mesh_stride=3)
        expected=self.compare_original(m)
        for chunk in (41,65536):
            with self.subTest(chunk=chunk):
                actual,n=s.simulate_mitsuba(replace(m,ray_chunk=chunk))
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(n,np.prod(m.shape)*24)


if __name__=='__main__':
    unittest.main()
