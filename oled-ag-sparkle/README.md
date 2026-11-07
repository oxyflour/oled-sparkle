# OLED brightness / direct sparkle skill

The report contains the height and OLED emission inputs, four collected brightness
maps, complete parameters and timing, and the direct image statistic:

`100 * std(I_ROI, ddof=0) / mean(I_ROI)`

All measurement pixels in the same geometric ROI participate, including dark
pixels. Each measurement pixel area-averages collected optical intensity over its
physical footprint. CV uses these floating-point values before PNG display scaling.

## Install

```bash
uv pip install -r requirements-core.txt
uv pip install -r requirements-optional.txt
```

The pinned Chromatix source requires Python 3.12+. Choose CPU/CUDA packages for
your platform. The measured Windows run uses CUDA Torch, CPU JAX and Mitsuba
CUDA. The wrapper uses `--device auto` and up to three independent backend workers;
the HTML records actual runtime paths, versions, device, concurrency and timing.

## Reproduce the sourced OLED example

```bash
python scripts/run_oled_demo.py --backend all --out outputs/oled-1mm-height1um-50um
```

The source image is Samsung Display's Diamond Pixel OLED arrangement:
https://global.samsungdisplay.com/29043/

The height and emission inputs cover **1 mm x 1 mm**. The height map is 1000x1000
at **1um**, interpolated from the existing 500x500 height values while retaining
16-bit encoding and the 0.5um height range. The emission input remains 500x500
at **2um**. Brightness is collected into **50um x 50um pixels**: each averages 50x50
optical samples, producing a 20x20 image. All 400 pixels enter the CV calculation;
the ROI margin is zero. The 0.5um height range and optical settings are retained.

Only the Diamond Pixel panel is cropped, then resampled to 500x500 with a stated
2um texel scale. Exact source URL, crop, original hash and intensity assumptions
are in `assets/oled-1mm/emission.source.json`. The original web image retains
Samsung Display's copyright. The input height is the existing synthetic quick
height map, bilinearly resampled with its provenance retained in
`assets/oled-1mm-height1um/heightfield.source.json`.

## Outputs

- `report.html`: offline HTML with two inputs, four brightness outputs and raw downloads.
- `run.json`: full CLI/default parameters, input hashes, environment and wall time.
- `comparison.csv/json`: direct brightness CV, brightness mean/std and solver time.
- Per backend: `brightness.npy/png` (1um optical grid), `analysis_brightness.npy/png`
  (50um collected pixels used directly for CV), `roi_mask.npy`, `stats.json`.

Solver time includes any library loading/JIT/device setup occurring inside the
solver, and excludes postprocessing/report writing. The total run time is recorded
up to HTML generation. Missing/failed values are not replaced with zero.

```bash
python scripts/generate_report.py --results outputs/oled-1mm-height1um-50um
python -m unittest discover -s tests -v
```

Set `SPARKLE_TEST_OPTIONAL=1` and `SPARKLE_TEST_MITSUBA_VARIANT=cuda_ad_rgb` to
include all-backend integration tests. The historic reference-normalized outputs
are superseded; see `VALIDATION.json` for the current direct-image result.
