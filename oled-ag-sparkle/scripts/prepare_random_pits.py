#!/usr/bin/env python3
"""Generate a reproducible nonoverlapping random circular-pit height surface."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def random_pits(field_um=1000., coverage=.5, seed=20261107,
                diameter_min_um=3., diameter_max_um=5.):
    """Draw diameters once, then place largest first with random nonoverlap positions."""
    if not (0 < diameter_min_um <= diameter_max_um < field_um):
        raise ValueError('Require 0 < minimum diameter <= maximum diameter < field')
    if not (0 < coverage <= .5):
        raise ValueError('Random sequential placement supports coverage in (0, 0.5]')
    rng = np.random.default_rng(seed)
    area = 0.
    diameters = []
    while area < coverage * field_um**2:
        diameter = float(rng.uniform(diameter_min_um, diameter_max_um))
        diameters.append(diameter)
        area += math.pi*(diameter/2)**2
    cells = {}
    circles = []
    trials = 0
    # Largest-first placement prevents a late large pit being trapped by small pits.
    # Diameters are never redrawn, preserving the requested uniform population.
    for diameter in sorted(diameters, reverse=True):
        radius = diameter / 2
        for _ in range(1_000_000):
            x, y = rng.uniform(radius, field_um-radius, size=2)
            cell = (int(x // diameter_max_um), int(y // diameter_max_um))
            trials += 1
            overlaps = any(
                (x-px)**2 + (y-py)**2 < (radius+pd/2)**2
                for cy in range(cell[1]-1, cell[1]+2)
                for cx in range(cell[0]-1, cell[0]+2)
                for px, py, pd in cells.get((cx, cy), ())
            )
            if not overlaps:
                pit = (float(x), float(y), diameter)
                circles.append(pit)
                cells.setdefault(cell, []).append(pit)
                break
        else:
            raise RuntimeError('Could not place next pit; reduce coverage or change seed')
    return np.asarray(circles, dtype=np.float64), trials


def rasterize(pits, field_um, pitch_um):
    """Point-sample h/depth = 1 - (1 - (r/R)^2)^2 inside each circular pit."""
    if not math.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError('Pitch must be positive and finite')
    size = round(field_um / pitch_um)
    if size < 8 or not math.isclose(size*pitch_um, field_um):
        raise ValueError('Pitch must evenly divide the field into at least 8 samples')
    height = np.ones((size, size), dtype=np.float64)
    for x, y, diameter in pits:
        radius = diameter/2
        x0 = max(0, math.ceil((x-radius)/pitch_um-.5))
        x1 = min(size, math.floor((x+radius)/pitch_um-.5)+1)
        y0 = max(0, math.ceil((y-radius)/pitch_um-.5))
        y1 = min(size, math.floor((y+radius)/pitch_um-.5)+1)
        xx = ((np.arange(x0, x1)+.5)*pitch_um-x)/radius
        yy = ((np.arange(y0, y1)+.5)*pitch_um-y)/radius
        q = yy[:, None]**2 + xx[None, :]**2
        profile = 1-np.maximum(0, 1-q)**2
        height[y0:y1, x0:x1] = np.minimum(height[y0:y1, x0:x1], profile)
    return np.rint(height*65535).clip(0, 65535).astype(np.uint16)


def prepare(output=None, pitch_um=1., field_um=1000., coverage=.5,
            depth_um=.5, seed=20261107):
    output = Path(output) if output else ROOT/'assets/oled-1mm-pits3to5um'
    if depth_um <= 0 or not math.isfinite(depth_um):
        raise ValueError('Depth must be positive and finite')
    started = time.perf_counter()
    pits, trials = random_pits(field_um, coverage, seed)
    height = rasterize(pits, field_um, pitch_um)
    output.mkdir(parents=True, exist_ok=True)
    geometry = output/'pits.csv'
    with geometry.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x_um_from_left', 'y_um_from_first_row', 'diameter_um'])
        writer.writerows(pits)
    path = output/'heightfield.png'
    Image.fromarray(height).save(path)
    metadata = {
        'source': 'synthetic analytic random circular pits; scripts/prepare_random_pits.py',
        'assigned_field_um': [field_um, field_um], 'assigned_texel_um': pitch_um,
        'height_range_um': depth_um, 'pit_depth_um': depth_um,
        'diameter_distribution': 'uniform on [3, 5] um; diameter drawn once per accepted pit',
        'diameter_um_min_max': [float(pits[:, 2].min()), float(pits[:, 2].max())],
        'diameter_um_mean': float(pits[:, 2].mean()),
        'placement': 'largest-first sequential placement at random positions; nonoverlapping, whole circles inside field',
        'rng': 'numpy.default_rng / PCG64', 'seed': seed,
        'target_coverage': coverage,
        'analytic_coverage': float(np.pi*np.sum((pits[:, 2]/2)**2)/field_um**2),
        'sampled_footprint_coverage': float(np.mean(height < 65535)),
        'pit_count': len(pits), 'position_trials': trials,
        'profile': 'h = depth * [1 - (1 - (r/R)^2)^2] for r < R; h = depth outside',
        'profile_note': 'smooth bowl with zero slope at center and rim; black is pit bottom',
        'sampling': 'analytic height at texel centers; 16-bit PNG; no height renormalization',
        'grid_size_px': list(height.shape),
        'sampled_height_min_max_um': [float(height.min()/65535*depth_um),
                                     float(height.max()/65535*depth_um)],
        'sampled_height_rms_about_mean_um': float(height.astype(float).std()/65535*depth_um),
        'geometry_file': geometry.name,
        'geometry_sha256': hashlib.sha256(geometry.read_bytes()).hexdigest(),
        'height_png_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'generation_elapsed_seconds': time.perf_counter()-started,
        'scale_note': 'synthetic geometry assumptions, not a measured surface',
    }
    (output/'heightfield.source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pitch-um', type=float, default=1.)
    parser.add_argument('--coverage', type=float, default=.5)
    parser.add_argument('--depth-um', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    prepare(args.out, args.pitch_um, coverage=args.coverage, depth_um=args.depth_um, seed=args.seed)
