"""Field expansion preserves emitter shape, texel pitch and periodicity."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_tiled_emission import (prepare, tile_to_field, crop_periodic_cell, boundary_statistics,
                                    LAYOUTS, layout_cell, prepare_layout)


class TiledEmissionTest(unittest.TestCase):
    def test_equal_address_pitch_preserves_different_exact_green_periods(self):
        expected = ((140, 140), (70, 70), (210, 140), (140, 140))
        for layout, period in zip(LAYOUTS, expected):
            with self.subTest(layout=layout):
                cell, meta = layout_cell(layout)
                self.assertEqual(meta['green_period_um_xy'], list(period))
                self.assertEqual(cell.shape[1]*2*cell.shape[0]*2/meta['nominal_addresses_per_cell'], 70**2)
        cell, _ = layout_cell('diamond')
        source = np.asarray(Image.open(ROOT/'assets/oled-diamond/emission.png'))
        np.testing.assert_array_equal(cell, source[58:128, 26:96])
        with self.assertRaises(ValueError):
            layout_cell('windmill', pixel_pitch_um=70.1)

    def test_linear_field_normalization_does_not_clip_or_discard_emitters(self):
        from sparkle_multibackend import read_emission
        with tempfile.TemporaryDirectory() as directory:
            for layout in LAYOUTS:
                folder = prepare_layout(layout, Path(directory)/layout)
                meta = json.loads((folder/'emission-green.source.json').read_text(encoding='utf-8'))
                linear = read_emission(folder/'emission-green.png', 'green', True)
                normalized = linear*meta['linear_emission_scale']
                self.assertAlmostEqual(float(normalized.mean(dtype=np.float64)), 1., places=7)
                self.assertGreater(normalized.max(), 1.)
                self.assertEqual(np.count_nonzero(normalized), np.count_nonzero(linear))
                rgb = np.asarray(Image.open(folder/'emission.png'))
                green = np.asarray(Image.open(folder/'emission-green.png'))
                np.testing.assert_array_equal(rgb[..., 1], green[..., 1])
                self.assertEqual(np.count_nonzero(green[..., (0, 2)]), 0)

    def test_partial_edge_tiles_preserve_all_source_texels(self):
        tile = np.arange(3*4*3, dtype=np.uint8).reshape(3, 4, 3)
        actual, repeats = tile_to_field(tile, field_um=20., pitch_um=2.)
        expected = tile[np.arange(10)[:, None] % 3, np.arange(10)[None, :] % 4]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(repeats, (4, 3))

    def test_bundled_1mm_field_keeps_texel_pitch_after_period_crop(self):
        source = np.asarray(Image.open(ROOT/'assets/oled-diamond/emission.png'))
        tile = source[58:128, 26:96]
        with tempfile.TemporaryDirectory() as directory:
            path = prepare(directory)
            output = np.asarray(Image.open(path/'emission.png'))
            metadata = json.loads((path/'emission.source.json').read_text(encoding='utf-8'))
        self.assertEqual(output.shape[:2], (500, 500))
        np.testing.assert_array_equal(output[:70, :70], tile)
        np.testing.assert_array_equal(output[70:, :], output[:-70, :])
        np.testing.assert_array_equal(output[:, 70:], output[:, :-70])
        np.testing.assert_array_equal(output[-1, -1], tile[499 % 70, 499 % 70])
        self.assertEqual(metadata['assigned_field_um'], [1000, 1000])
        self.assertEqual(metadata['source_texture_field_um'], [256, 256])
        self.assertEqual(metadata['resized_to_px'], [128, 128])
        self.assertEqual(metadata['source_tile_field_um'], [140, 140])
        self.assertEqual(metadata['cell_crop_xyxy_in_source_texture'], [26, 58, 96, 128])
        self.assertEqual(metadata['assigned_emission_texel_um'], 2)
        self.assertFalse(metadata['pixel_scale_changed'])

    def test_cell_contains_complete_emitter_bodies_and_continuous_joins(self):
        source = np.asarray(Image.open(ROOT/'assets/oled-diamond/emission.png'))
        cell, crop, periods, stats = crop_periodic_cell(source)
        self.assertEqual(periods['x']['period_texels'], 70)
        self.assertEqual(periods['y']['period_texels'], 70)
        for axis in ('x', 'y'):
            self.assertLess(stats[axis+'_wrap_mse'], stats[axis+'_interior_neighbor_mse'])
        # This condition failed for the previous uncropped 128x128 patch.
        old = boundary_statistics(source)
        self.assertGreater(old['x_wrap_mse'], old['x_interior_neighbor_mse'])
        rgb = cell[..., :3].astype(float)/255
        for channel, expected in ((0, 2), (1, 4), (2, 2)):
            mask = (rgb[..., channel] > .4) & (rgb[..., channel] >
                    1.8*np.max(rgb[..., [c for c in range(3) if c != channel]], axis=-1))
            self.assertFalse(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
            # Independent 4-connected segmentation counts complete R/G/B bodies.
            remaining = set(map(tuple, np.argwhere(mask)))
            count = 0
            while remaining:
                stack = [remaining.pop()]
                count += 1
                while stack:
                    y, x = stack.pop()
                    for neighbor in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)
            self.assertEqual(count, expected)


if __name__ == '__main__':
    unittest.main()
