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
LAYOUTS = ('diamond', 'rgb-stripe', 'rgb-delta', 'windmill')


def exact_periods(image):
    """Smallest exact axis-aligned periods in x/y texels, including wrap."""
    return [next(p for p in range(1, image.shape[axis]+1) if image.shape[axis] % p == 0
                 and np.array_equal(image, np.roll(image, p, axis=axis))) for axis in (1, 0)]


def layout_cell(layout, pixel_pitch_um=70., texel_um=2.):
    """Equal nominal-address pitch; new apertures approximate public diagrams.

    These are representative geometries, not measured phone-panel masks.
    Diamond retains the existing sourced cell at P=70 um and dx=2 um.
    """
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown layout: {layout}')
    if any(not math.isfinite(v) or v <= 0 for v in (pixel_pitch_um, texel_um)):
        raise ValueError('Pixel and texel pitches must be positive and finite')
    factors = {'diamond': (2, 2), 'rgb-stripe': (1, 1),
               'rgb-delta': (3, 2), 'windmill': (2, 2)}[layout]
    dimensions = [v*pixel_pitch_um/texel_um for v in factors]
    if any(not math.isclose(v, round(v)) for v in dimensions):
        raise ValueError('Each complete cell must contain integer emission texels')
    width, height = map(round, dimensions)
    if layout == 'diamond':
        original = np.asarray(Image.open(ROOT/'assets/oled-diamond/emission.png').convert('RGB'))
        cell, crop, _, _ = crop_periodic_cell(original)
        definition = {'method': 'Existing sourced repeat cell', 'source_cell_crop_xyxy': crop}
        if cell.shape[:2] != (height, width):
            cell = np.asarray(Image.fromarray(cell).resize((width, height), Image.Resampling.BOX))
            definition['resampling'] = 'BOX at the requested declared pixel pitch'
    else:
        rectangle = [(-.145, -.44), (.145, -.44), (.145, .44), (-.145, .44)]
        hexagon = [(-.18, -.06), (-.12, -.12), (.12, -.12), (.18, -.06),
                   (.18, .06), (.12, .12), (-.12, .12), (-.18, .06)]
        if layout == 'rgb-stripe':
            sites = [(c, ((c+.5)/3, .5), rectangle) for c in range(3)]
        elif layout == 'rgb-delta':
            sites = []
            # RB/GR/BG and GB/RG/BR; staggered centers form triangles.
            for row, colors in enumerate(((0, 2, 1, 0, 2, 1), (1, 2, 0, 1, 2, 0))):
                for col, color in enumerate(colors):
                    sites.append((color, (.25+.5*col, row+.25+.5*(col % 2)), hexagon))
        else:
            diamond = lambda radius: [(0, -radius), (radius, 0), (0, radius), (-radius, 0)]
            sites = [(0, (.75, .75), diamond(.23)), (0, (1.75, 1.75), diamond(.23)),
                     (2, (1.75, .75), diamond(.34)), (2, (.75, 1.75), diamond(.34))]
            green = [(-.16, -.08), (-.08, -.16), (-.02, -.16), (.16, .02),
                     (.16, .08), (.08, .16), (.02, .16), (-.16, -.02)]
            for row in range(2):
                for col in range(2):
                    polygon = [(x, y if (row+col) % 2 == 0 else -y) for x, y in green]
                    sites.append((1, (.25+col, .25+row), polygon))
        # Integrate analytic convex polygons, excluding labels and black
        # construction lines in the references. Wrap geometry before sampling.
        samples = 8
        yy, xx = np.meshgrid((np.arange(height*samples)+.5)*texel_um/samples/pixel_pitch_um,
                             (np.arange(width*samples)+.5)*texel_um/samples/pixel_pitch_um, indexing='ij')
        masks = np.zeros((*xx.shape, 3), dtype=bool)
        for color, (cx, cy), polygon in sites:
            dx = (xx-cx+factors[0]/2) % factors[0]-factors[0]/2
            dy = (yy-cy+factors[1]/2) % factors[1]-factors[1]/2
            positive = np.ones(xx.shape, dtype=bool)
            negative = positive.copy()
            for (x0, y0), (x1, y1) in zip(polygon, polygon[1:]+polygon[:1]):
                cross = (x1-x0)*(dy-y0)-(y1-y0)*(dx-x0)
                positive &= cross >= -1e-12
                negative &= cross <= 1e-12
            masks[..., color] |= positive | negative
        coverage = masks.reshape(height, samples, width, samples, 3).mean(axis=(1, 3))
        # Encode linear area coverage as sRGB for the existing solver decoder.
        srgb = np.where(coverage <= .0031308, 12.92*coverage, 1.055*coverage**(1/2.4)-.055)
        cell = np.rint(255*srgb).astype(np.uint8)
        definition = {'method': 'Idealized polygons inspired by public diagrams; proportions are model assumptions',
                      'area_samples_per_axis': samples,
                      'sites_in_nominal_pitch_units': [{'channel': 'RGB'[c], 'center_xy': center,
                                                       'vertices_relative_xy': poly} for c, center, poly in sites]}
    return cell, {'layout': layout, 'nominal_pixel_pitch_um': pixel_pitch_um,
                  'nominal_pitch_definition': 'square logical-address pitch P; cell area/P^2 addresses; not nearest-neighbor subpixel spacing',
                  'nominal_addresses_per_cell': factors[0]*factors[1],
                  'full_cell_um_xy': [v*pixel_pitch_um for v in factors],
                  'full_rgb_period_um_xy': [p*texel_um for p in exact_periods(cell)],
                  'green_period_um_xy': [p*texel_um for p in exact_periods(cell[..., 1])], 'geometry': definition}


def prepare_layout(layout, output=None, pixel_pitch_um=70., texel_um=2., field_um=1000.):
    """Write reusable RGB/green inputs, exact periods and linear normalization."""
    from sparkle_multibackend import read_emission
    output = Path(output) if output else ROOT/'assets/oled-layouts'/layout
    output.mkdir(parents=True, exist_ok=True)
    cell, meta = layout_cell(layout, pixel_pitch_um, texel_um)
    field, repeats = tile_to_field(cell, field_um, texel_um)
    green = np.zeros_like(field)
    green[..., 1] = field[..., 1]
    Image.fromarray(cell).save(output/'periodic-cell.png')
    Image.fromarray(field).save(output/'emission.png')
    Image.fromarray(green).save(output/'emission-green.png')
    linear = read_emission(output/'emission-green.png', 'green', True)
    source = ('assets/oled-diamond/emission.png' if layout == 'diamond' else
              'assets/oled-layout-reference/'+('tianma-windmill.png' if layout == 'windmill' else 'tcl-layouts.jpg'))
    urls = {'diamond': 'https://global.samsungdisplay.com/29043/', 'windmill': 'https://www.aibangled.com/a/2255',
            'rgb-stripe': 'https://wx4.sinaimg.cn/large/006fnRbbgy1gjq5m7x6afj30zq0k3tc4.jpg',
            'rgb-delta': 'https://patents.google.com/patent/CN110379370A/zh'}
    mean = float(linear.mean(dtype=np.float64))
    meta.update({'title': f'{layout}: equal nominal pitch / full-green comparison input',
                 'source_page': urls[layout], 'reference_image': source,
                 'reference_image_sha256': hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                 'assigned_field_um': [field_um, field_um], 'assigned_emission_texel_um': texel_um,
                 'tile_repeats_yx': list(repeats), 'phase': 'repeat-cell origin at top-left of the common field',
                 'intensity_model': 'Full green, sRGB-decoded at 530 nm; linear gain applied by solver, without PNG clipping',
                 'unscaled_field_linear_green_mean': mean, 'target_field_linear_green_mean': 1.,
                 'linear_emission_scale': 1./mean,
                 'normalization_domain': 'entire finite emission field, including partial edge cells',
                 'scale_note': 'Nominal P and aperture proportions are explicit model assumptions, not measured phone calibration.'})
    for name in ('emission', 'emission-green'):
        info = {**meta, 'sha256': hashlib.sha256((output/(name+'.png')).read_bytes()).hexdigest()}
        (output/(name+'.source.json')).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


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
    parser.add_argument('--layout', choices=LAYOUTS, help='Equal-pitch comparison layout; omitted preserves existing sourced preparation')
    parser.add_argument('--pixel-pitch-um', type=float, default=70.)
    parser.add_argument('--texel-um', type=float, default=2.)
    parser.add_argument('--field-um', type=float, default=1000.)
    args = parser.parse_args()
    if args.layout:
        print(prepare_layout(args.layout, args.out, args.pixel_pitch_um, args.texel_um, args.field_um))
    else:
        prepare(args.out)
