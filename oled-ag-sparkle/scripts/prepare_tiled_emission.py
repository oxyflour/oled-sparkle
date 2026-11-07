#!/usr/bin/env python3
"""Crop a complete OLED repeat cell before expanding the field by integer tiling."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def boundary_statistics(tile):
    rgb = tile[..., :3].astype(np.float64)/255
    edges = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    x_wrap = float(np.mean((rgb[:, 0]-rgb[:, -1])**2))
    y_wrap = float(np.mean((rgb[0]-rgb[-1])**2))
    return {'x_wrap_mse': x_wrap, 'y_wrap_mse': y_wrap,
            'x_interior_neighbor_mse': float(np.mean(np.diff(rgb, axis=1)**2)),
            'y_interior_neighbor_mse': float(np.mean(np.diff(rgb, axis=0)**2)),
            'edge_rgb_energy': float(np.mean(edges**2)),
            'edge_max_srgb': float(edges.max())}


def crop_periodic_cell(texture):
    """Find integer RGB repeat periods, then choose low-energy matching boundaries."""
    rgb = texture[..., :3].astype(np.float64)/255
    periods = {}
    for axis, name in ((1, 'x'), (0, 'y')):
        size = rgb.shape[axis]
        scores = []
        for period in range(max(2, size//4), 3*size//4+1):
            first = np.take(rgb, range(period, size), axis=axis)
            second = np.take(rgb, range(size-period), axis=axis)
            scores.append((float(np.mean((first-second)**2)), period))
        score, period = min(scores)
        periods[name] = {'period_texels': period, 'translation_mse': score,
                         'search_range_inclusive': [max(2, size//4), 3*size//4]}
    width, height = periods['x']['period_texels'], periods['y']['period_texels']
    best = None
    for y in range(texture.shape[0]-height+1):
        for x in range(texture.shape[1]-width+1):
            cell = texture[y:y+height, x:x+width]
            stats = boundary_statistics(cell)
            cost = stats['edge_rgb_energy']+(stats['x_wrap_mse']+stats['y_wrap_mse'])/2
            if best is None or cost < best[0]:
                best = (cost, x, y, stats)
    _, x, y, stats = best
    return texture[y:y+height, x:x+width].copy(), [x, y, x+width, y+height], periods, stats


def tile_to_field(tile, field_um, pitch_um):
    if not math.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError('Texel pitch must be positive and finite')
    size = round(field_um/pitch_um)
    if size < 1 or not math.isclose(size*pitch_um, field_um):
        raise ValueError('Field must contain an integer number of texels')
    ny, nx = tile.shape[:2]
    repeats = (math.ceil(size/ny), math.ceil(size/nx))
    repetitions = repeats + (1,)*(tile.ndim-2)
    return np.tile(tile, repetitions)[:size, :size].copy(), repeats


def prepare(output=None):
    output = Path(output) if output else ROOT/'assets/oled-1mm-periodic'
    source = ROOT/'assets/oled-diamond/emission.png'
    metadata = json.loads(source.with_suffix('.source.json').read_text(encoding='utf-8'))
    pitch = float(metadata['assigned_emission_texel_um'])
    with Image.open(source) as image:
        texture = np.asarray(image)
    tile, crop, periods, boundary = crop_periodic_cell(texture)
    expanded, repeats = tile_to_field(tile, field_um=1000., pitch_um=pitch)
    output.mkdir(parents=True, exist_ok=True)
    cell_path = output/'periodic-cell.png'
    Image.fromarray(tile).save(cell_path)
    target = output/'emission.png'
    Image.fromarray(expanded).save(target)
    metadata.update({
        'source_texture': 'assets/oled-diamond/emission.png',
        'source_texture_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_texture_size_px': [texture.shape[1], texture.shape[0]],
        'source_texture_field_um': [texture.shape[1]*pitch, texture.shape[0]*pitch],
        'cell_crop_xyxy_in_source_texture': crop,
        'period_detection': periods,
        'crop_selection': 'Minimize boundary RGB energy plus opposite-edge neighbor jump at the detected periods',
        'cell_boundary_statistics': boundary,
        'source_tile': 'assets/oled-1mm-periodic/periodic-cell.png',
        'source_tile_sha256': hashlib.sha256(cell_path.read_bytes()).hexdigest(),
        'source_tile_size_px': [tile.shape[1], tile.shape[0]],
        'source_tile_field_um': [tile.shape[1]*pitch, tile.shape[0]*pitch],
        'source_tile_provenance': json.loads(source.with_suffix('.source.json').read_text(encoding='utf-8')),
        'expansion_method': 'crop one complete repeat cell, integer tile, then top-left field crop; no resizing or interpolation',
        'tile_repeats_xy': [repeats[1], repeats[0]],
        'tiled_canvas_size_px': [repeats[1]*tile.shape[1], repeats[0]*tile.shape[0]],
        'field_crop_xyxy': [0, 0, expanded.shape[1], expanded.shape[0]],
        'output_size_px': [expanded.shape[1], expanded.shape[0]],
        'assigned_field_um': [1000, 1000], 'assigned_emission_texel_um': pitch,
        'pixel_scale_changed': False,
        'resized_to_px': [texture.shape[1], texture.shape[0]],
        'resampling': 'none during field expansion; source tile retains its original BOX resampling',
        'tile_boundary_note': 'One 70x70 repeat cell is cropped before tiling. Wrap-neighbor jumps '
                              'are smaller than ordinary within-cell neighbor variation in both axes. '
                              'No seam blending, pixel repainting or procedural emitter reconstruction is applied.',
        'scale_note': 'Original 2um/texel example assumption retained, not measured panel calibration. '
                      'Increasing the field changes tile count, not emitter dimensions or spacing.',
        'emission_png_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    })
    target.with_suffix('.source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(target), 'tile_size_px': metadata['source_tile_size_px'],
                      'tile_field_um': metadata['source_tile_field_um'], 'repeats_xy': metadata['tile_repeats_xy'],
                      'cell_crop_xyxy': crop, 'boundary_statistics': boundary,
                      'output_size_px': metadata['output_size_px'], 'pitch_um': pitch}, indent=2), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    prepare(args.out)
