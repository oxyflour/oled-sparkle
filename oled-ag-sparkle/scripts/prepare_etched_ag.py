#!/usr/bin/env python3
"""Generate a synthetic etched-AG proxy: rounded, irregular Voronoi bowls.

The literature motivates the morphology and order of magnitude, not the exact
height function. This is a reproducible statistical proxy, not a measured sample.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def geometry(field_um, spacing_um, seed):
    if not (math.isfinite(field_um) and math.isfinite(spacing_um)
            and 0 < spacing_um <= field_um / 4):
        raise ValueError('Require finite 0 < cell spacing <= field / 4')
    count = round((field_um / spacing_um)**2)
    rng = np.random.default_rng(seed)
    sites = np.empty((count, 2), dtype=np.float64)
    minimum_distance = .3 * spacing_um
    for index in range(count):
        for _ in range(100_000):
            candidate = rng.uniform(0, field_um, 2)
            delta = np.abs(sites[:index] - candidate)
            delta = np.minimum(delta, field_um - delta)
            if np.all(np.sum(delta**2, axis=1) >= minimum_distance**2):
                sites[index] = candidate
                break
        else:
            raise RuntimeError('Could not place sites at the specified spacing')
    return sites, rng.uniform(.7, 1.3, count)


def raw_surface(sites, depths, field_um, pitch_um, rounding_um):
    if not math.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError('Pitch must be finite and positive')
    size = round(field_um / pitch_um)
    if size < 8 or not math.isclose(size * pitch_um, field_um):
        raise ValueError('Pitch must divide the field into at least 8 samples')
    if not math.isfinite(rounding_um) or rounding_um <= 0:
        raise ValueError('Rounding must be finite and positive')
    tree = cKDTree(sites, boxsize=field_um)
    raw = np.empty((size, size), dtype=np.float64)
    counts = np.zeros(len(sites), dtype=np.int64)
    axis = (np.arange(size) + .5) * pitch_um
    for start in range(0, size, 64):
        stop = min(size, start + 64)
        xx, yy = np.meshgrid(axis, axis[start:stop])
        distances, indices = tree.query(np.column_stack((xx.ravel(), yy.ravel())), k=2)
        # d1=d2 on a Voronoi boundary. Height and first derivative approach
        # zero there; d1=0 gives a smooth bowl bottom at the site.
        ratio_squared = (distances[:, 0] / distances[:, 1])**2
        values = -depths[indices[:, 0]] * (1 - ratio_squared)**2
        raw[start:stop] = values.reshape(stop - start, size)
        counts += np.bincount(indices[:, 0], minlength=len(sites))
    return gaussian_filter(raw, rounding_um / pitch_um, mode='wrap'), counts


def roughness(height_um, pitch_um):
    centered = height_um - height_um.mean()
    gy, gx = np.gradient(height_um, pitch_um)
    return {
        'Sa_um': float(np.abs(centered).mean()),
        'Sq_um': float(centered.std()),
        'Sz_um': float(np.ptp(height_um)),
        'height_min_um': float(height_um.min()),
        'height_max_um': float(height_um.max()),
        'rms_slope': float(np.sqrt(np.mean(gx**2 + gy**2))),
        'definition': 'Areal Sa/Sq relative to the mean plane; Sz is sampled peak-to-valley, not PNG encoding range.',
    }


def prepare(output=None, pitch_um=1., field_um=1000., spacing_um=25.,
            target_sa_um=.10, rounding_um=1.2, seed=20261107,
            reference_pitch_um=.5):
    if not math.isfinite(target_sa_um) or target_sa_um <= 0:
        raise ValueError('Target Sa must be finite and positive')
    started = time.perf_counter()
    output = Path(output) if output else ROOT / 'assets/oled-1mm-etched-ag'
    sites, depths = geometry(field_um, spacing_um, seed)
    reference, counts = raw_surface(sites, depths, field_um, reference_pitch_um, rounding_um)
    reference_sa = np.abs(reference - reference.mean()).mean()
    scale = target_sa_um / reference_sa
    raw_span = np.ptp(reference)
    # Fixed calibration at the reference grid, shared by every output pitch.
    # Two percent headroom avoids per-grid min/max renormalization.
    black_raw = float(reference.min() - .02 * raw_span)
    encoding_range_um = float(1.04 * raw_span * scale)
    raw = reference if pitch_um == reference_pitch_um else raw_surface(
        sites, depths, field_um, pitch_um, rounding_um)[0]
    physical = (raw - black_raw) * scale
    encoded = physical / encoding_range_um
    if encoded.min() < 0 or encoded.max() > 1:
        raise ValueError('Reference calibration does not contain the requested grid')
    values = np.rint(encoded * 65535).astype(np.uint16)
    decoded = values.astype(np.float64) * (encoding_range_um / 65535)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'heightfield.png'
    Image.fromarray(values).save(path)
    np.save(output / 'height_um.npy', decoded)
    with (output / 'sites.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x_um', 'y_um', 'relative_bowl_depth'])
        writer.writerows(np.column_stack((sites, depths)))
    diameters = 2 * np.sqrt(counts * reference_pitch_um**2 / np.pi)
    meta = {
        'source': 'Synthetic rounded Voronoi-bowl proxy; scripts/prepare_etched_ag.py',
        'model_status': 'Illustrative morphology, not measured glass or a calibrated commercial product.',
        'morphology_reference_url': 'https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf',
        'roughness_order_of_magnitude_reference_url': 'https://www.agc-yourglass.com/sites/default/files/agc_docs/Factsheet_Anti-glare_1020_EN.pdf',
        'reference_interpretation': 'Irregular polygonal etched features motivate the proxy. AGC line Ra/Rsm specifications are not equal to generated areal Sa or equivalent cell diameter.',
        'assigned_field_um': [field_um, field_um], 'assigned_texel_um': pitch_um,
        'height_range_um': encoding_range_um, 'png_bits_per_channel': 16,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'seed': seed, 'cell_count': len(sites), 'nominal_site_spacing_um': spacing_um,
        'minimum_site_separation_um': .3 * spacing_um,
        'relative_depth_distribution': 'Uniform [0.7, 1.3], assigned independently to each cell',
        'raw_profile': '-depth_i * (1 - (d_nearest / d_second_nearest)^2)^2',
        'rounding': {'gaussian_sigma_um': rounding_um, 'boundary': 'periodic wrap'},
        'periodic_domain_um': field_um,
        'calibration': {'reference_pitch_um': reference_pitch_um, 'target_Sa_um': target_sa_um,
                        'raw_to_um_scale': float(scale), 'raw_black_level': black_raw,
                        'encoding': 'uint16 / 65535 * height_range_um; black is low, white is high',
                        'no_per_output_grid_rescaling': True},
        'equivalent_cell_diameter_um': {'definition': '2*sqrt(Voronoi cell area/pi), not Rsm or nearest-neighbor spacing',
                                       'mean': float(diameters.mean()),
                                       'p10': float(np.percentile(diameters, 10)),
                                       'p50': float(np.median(diameters)),
                                       'p90': float(np.percentile(diameters, 90))},
        'measured_surface': roughness(decoded, pitch_um),
        'maximum_quantization_error_um': float(np.max(np.abs(decoded - physical))),
        'generation_elapsed_seconds': time.perf_counter() - started,
    }
    (output / 'heightfield.source.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pitch-um', type=float, default=1.)
    parser.add_argument('--field-um', type=float, default=1000.)
    parser.add_argument('--spacing-um', type=float, default=25.)
    parser.add_argument('--sa-um', type=float, default=.10)
    parser.add_argument('--rounding-um', type=float, default=1.2)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    output = prepare(args.out, args.pitch_um, args.field_um, args.spacing_um,
                     args.sa_um, args.rounding_um, args.seed)
    print((output / 'heightfield.source.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
