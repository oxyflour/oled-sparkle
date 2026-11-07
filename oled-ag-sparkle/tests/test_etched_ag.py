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
from prepare_etched_ag import geometry, prepare


class EtchedAGTests(unittest.TestCase):
    def test_same_geometry_and_scale_at_both_pitches(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = [prepare(Path(folder) / str(p), pitch_um=p, field_um=100., spacing_um=20.)
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


if __name__ == '__main__':
    unittest.main()
