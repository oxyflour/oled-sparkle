from pathlib import Path
import json
import unittest
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

class PackageAssetsTest(unittest.TestCase):
    def test_skill_header_and_scripts(self):
        content=(ROOT/'SKILL.md').read_text(encoding='utf-8')
        self.assertIn('name: oled-ag-sparkle',content)
        self.assertTrue((ROOT/'scripts/run_legacy_demo.py').is_file())
        self.assertTrue((ROOT/'scripts/sparkle_multibackend.py').is_file())
        self.assertTrue((ROOT/'scripts/reporting.py').is_file())
        self.assertTrue((ROOT/'scripts/generate_report.py').is_file())

    def test_1mm_geometry_and_encoding(self):
        base=ROOT/'assets/demo_1mm'
        meta=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
        height_path=base/'heightfield.png'
        h=np.array(Image.open(height_path))
        self.assertEqual(h.shape,(2000,2000))
        # Pillow 10 reads 16-bit PNGs as mode I / int32; newer versions
        # preserve uint16. Check the actual PNG IHDR encoding instead.
        header=height_path.read_bytes()[:26]
        self.assertEqual(header[:8],b'\x89PNG\r\n\x1a\n')
        self.assertEqual(header[24:26],bytes([16,0]))  # 16-bit grayscale
        self.assertTrue(np.issubdtype(h.dtype,np.integer))
        self.assertGreaterEqual(h.min(),0)
        self.assertLessEqual(h.max(),65535)
        self.assertAlmostEqual(h.shape[0]*meta['heightfield']['image_pitch_um']/1000,1.0)
        em=np.array(Image.open(base/'emission.png'))
        self.assertEqual(em.shape,(20,20,3))
        self.assertEqual(em[...,1].sum(),20*20*255)
        detail=np.array(Image.open(base/'emission_subpixel.png'))
        self.assertEqual(detail.shape,(100,100,3))

    def test_sourced_oled_1mm_inputs(self):
        base=ROOT/'assets/oled-1mm'
        for name in ('heightfield','emission'):
            with self.subTest(input=name):
                path=base/(name+'.png')
                with Image.open(path) as image: self.assertEqual(image.size,(500,500))
                meta=json.loads((base/(name+'.source.json')).read_text(encoding='utf-8'))
                self.assertEqual(meta['assigned_field_um'],[1000,1000])
        self.assertEqual((base/'heightfield.png').read_bytes()[24:26],bytes([16,0]))

    def test_refined_height_keeps_1mm_extent_and_existing_height_data(self):
        base=ROOT/'assets/oled-1mm-height1um'
        path=base/'heightfield.png'
        with Image.open(path) as image: self.assertEqual(image.size,(1000,1000))
        self.assertEqual(path.read_bytes()[24:26],bytes([16,0]))
        metadata=json.loads((base/'heightfield.source.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['assigned_field_um'],[1000,1000])
        self.assertEqual(metadata['assigned_texel_um'],1)
        import hashlib
        self.assertEqual(metadata['original_sha256'],hashlib.sha256(
            (ROOT/'assets/oled-1mm/heightfield.png').read_bytes()).hexdigest())

if __name__=='__main__': unittest.main()
