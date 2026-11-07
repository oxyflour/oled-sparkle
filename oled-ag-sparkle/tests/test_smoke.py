"""Dependency-free unittest smoke tests for the native torch reference backend."""
from __future__ import annotations
import json
import hashlib
from html.parser import HTMLParser
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

SCRIPT=Path(__file__).resolve().parents[1]/'sparkle_multibackend.py'


class ReportDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside=False
        self.payload=[]

    def handle_starttag(self,tag,attrs):
        if tag=='script' and dict(attrs).get('id')=='report-data':
            self.inside=True

    def handle_endtag(self,tag):
        if tag=='script': self.inside=False

    def handle_data(self,data):
        if self.inside: self.payload.append(data)


class SmokeTest(unittest.TestCase):
    def run_sample(self,flat=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rng=np.random.default_rng(44)
            h=np.full((128,128),128,dtype=np.uint8)
            if not flat:
                h=np.clip(128+25*rng.standard_normal((128,128)),0,255).astype(np.uint8)
            em=np.full((4,4,3),255,dtype=np.uint8)
            Image.fromarray(h).save(root/'heightfield.png')
            Image.fromarray(em).save(root/'emission.png')
            cmd=[sys.executable,str(SCRIPT),'--height',str(root/'heightfield.png'),
                 '--height-pitch-um','2','--height-range-um','0.5',
                 '--emission',str(root/'emission.png'),'--emission-pitch-um','64',
                 '--analysis-pitch-um','64','--gap-um','500','--pad-px','64',
                 '--roi-margin-um','0','--backend','torch','--device','cpu',
                 '--progress-every','0','--out',str(root/'results')]
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            stats=json.loads((root/'results'/'torch'/'stats.json').read_text())
            html=(root/'results'/'report.html').read_text(encoding='utf-8')
            parser=ReportDataParser(); parser.feed(html)
            report=json.loads(''.join(parser.payload))
            group=report['groups'][0]
            self.assertEqual(group['records'][0]['stats'],stats)
            self.assertEqual(group['run']['parameters']['emission_channel'],'luma')
            self.assertFalse(group['run']['parameters']['gamma_decode'])
            self.assertEqual(group['run']['parameters']['height_pitch_um'],2)
            self.assertEqual(group['run']['parameters']['source_samples'],1)
            self.assertEqual(group['run']['input_files']['height']['sha256'],
                             hashlib.sha256((root/'heightfield.png').read_bytes()).hexdigest())
            self.assertGreaterEqual(group['run']['wall_elapsed_seconds'],stats['elapsed_seconds'])
            self.assertGreaterEqual(stats['postprocess_elapsed_seconds'],0)
            self.assertIn('data:image/png;base64,',html)
            for f in ['brightness.npy','analysis_brightness.npy','analysis_brightness.png','roi_mask.npy']:
                self.assertTrue((root/'results'/'torch'/f).exists())
            self.assertEqual(np.load(root/'results'/'torch'/'brightness.npy').shape,(128,128))
            brightness=np.load(root/'results/torch/analysis_brightness.npy')
            fine=np.load(root/'results/torch/brightness.npy')
            np.testing.assert_allclose(brightness,fine.astype(np.float64).reshape(4,32,4,32).mean(axis=(1,3)),rtol=1e-14)
            self.assertEqual(brightness.shape,(4,4))
            self.assertEqual(stats['analysis_pixel_pitch_um'],64)
            mask=np.load(root/'results/torch/roi_mask.npy')
            self.assertTrue(mask.all())
            values=brightness[mask].astype(np.float64)
            self.assertAlmostEqual(stats['sparkle_cv_percent'],100*values.std()/values.mean(),places=10)
            self.assertEqual(stats['num_roi_pixels'],values.size)
            self.assertNotIn('reference_normalized_cv_percent',stats)
            return stats['sparkle_cv_percent']

    def test_constant_height_uses_direct_image_metric(self):
        self.run_sample(flat=True)

    def test_random_height_uses_direct_image_metric(self):
        self.assertGreater(self.run_sample(flat=False),0.01)


if __name__=='__main__':
    unittest.main()
