---
name: oled-ag-sparkle
description: Simulate collected OLED brightness with PyTorch FFT, TorchOptics, Chromatix and Mitsuba; report height/emission inputs, four brightness images, direct brightness-pixel CV, full parameters and computation times in offline HTML.
---

# OLED brightness and direct sparkle reporting

## Required result

Use the supplied height map and a sourced, representative OLED emission image.
For the bundled example, the emission image is cropped and resampled from Samsung
Display's Diamond Pixel image. Attribution, crop, scale and intensity assumptions
are recorded in `assets/oled-1mm/emission.source.json`.

The HTML report must visibly include:

1. The actual height map and emission image, their paths, SHA-256, dimensions,
   bit depth, physical scales and the emission source URL. Preserve original PNG
   bytes for download, including 16-bit height values.
2. One collected brightness image for each of `torch`, `torchoptics`, `chromatix`
   and `mitsuba`, plus raw floating-point brightness downloads.
3. The sparkle value computed **directly from brightness pixels** in the same
   geometric rectangular ROI for all backends:

   `sparkle_cv_percent = 100 * std(I_ROI, ddof=0) / mean(I_ROI)`

   Use the requested physical measurement-pixel size (`--analysis-pitch-um`).
   Area-average the optical-grid intensity over each square pixel and save it as
   `analysis_brightness.npy`. Include all full ROI pixels, including dark pixels.
   Compute CV from these float values before PNG quantization. Record pixel size,
   grid dimensions, physical ROI bounds, pixel count, mean and std.
4. All CLI parameters and defaults, actual backend devices, software versions,
   input/source sampling, measured solver/postprocessing times and run wall time.

Do not divide by a plane-reference result or use a plane-reference signal mask.
The physical measurement-pixel integration is specified by the user; do not add
other filtering or normalize out the OLED pattern. Plane-control assets are not
report inputs. If full pixels do not fit the ROI exactly, center the largest full
pixel grid inside it and record the effective boundaries.

## Units and models

Keep height texel pitch, black-to-white physical height range, emission texel
pitch, and OLED-to-AG distance distinct. Record explicit assumptions for physical
scale when a web illustration has no calibrated scale. The bundled data uses
1000x1000 height texels at 1um and 500x500 emission texels at 2um, a 0.5um height
range and a 1mm x 1mm field. Height values are interpolated from the previous
500x500 map, with that sampling refinement recorded in the source metadata. Use
50um measurement pixels (20x20, 400 total), zero ROI margin, a 500um gap,
530nm effective wavelength and camera NA 0.08. RGB input is gamma-decoded and
mapped to relative luma for this single-wavelength example.

The first three backends implement scalar ASM + phase-screen imaging. Mitsuba
uses geometric refraction and pupil quadrature. Their brightness maps are the
outputs to compare; the models remain distinct. Use source-sampling and grid
checks as needed, without changing the requested direct pixel statistic.

## Run

```bash
python scripts/run_oled_demo.py --backend all --out outputs/oled-1mm-height1um-50um
```

The wrapper selects Torch's available device with `--device auto` and uses up to
three independent backend workers. The measured Windows run uses CUDA Torch,
CPU JAX and Mitsuba `cuda_ad_rgb`; actual versions, device and runtime paths are
recorded in the report. Override `--device`, `--backend-workers` or
`--mitsuba-variant` as needed. Each 50um measurement pixel averages 50x50 optical
samples; emission remains at 2um during the height refinement.
For custom inputs use `python scripts/sparkle_multibackend.py --help`.
Use `--dry-run` before large calculations; it writes a report marked uncomputed.

Every calculation writes `report.html`, `run.json`, `comparison.json/csv` and,
per backend, `brightness.npy/png` (optical grid), `analysis_brightness.npy/png`
(measurement pixels), `roi_mask.npy` (measurement grid) and `stats.json`.
Failed, skipped and dry-run values must remain uncomputed, not zero.

Regenerate HTML without optical recomputation:

```bash
python scripts/generate_report.py --results outputs/oled-1mm-height1um-50um
```

`--recursive` combines saved runs; repeated `--figure path.png` embeds plots.
Always deliver the HTML report and code/output paths. State which backends ran.
Use `VALIDATION.json` for the latest measured result and test status.
