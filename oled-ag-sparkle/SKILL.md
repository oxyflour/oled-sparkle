---
name: oled-ag-sparkle
description: Simulate collected OLED brightness with PyTorch FFT, TorchOptics, Chromatix and Mitsuba; report height/emission inputs, four brightness images, direct brightness-pixel CV, full parameters and computation times in offline HTML.
---

# OLED brightness and direct sparkle reporting

## Required result

Use the supplied height map and a sourced, representative OLED emission image.
For the bundled example, start from the 128x128 crop of Samsung Display's Diamond
Pixel image. This complete source patch is NOT a repeatable texture. Detect its
70-texel horizontal/vertical periods and crop [26,58,96,128] to obtain a 70x70
repeat cell. Keep 2um/texel: the cell covers 140um square. Repeat the cell 8x8,
then crop the 560x560 canvas to 500x500 without resizing or interpolation.
Attribution, source/cell hashes, period detection, crop, boundary checks, repeats
and intensity assumptions are recorded in `assets/oled-1mm-periodic/emission.source.json`.

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
scale when a web illustration has no calibrated scale. Increasing the physical
calculation area must preserve emitter dimensions and spacing: repeat an existing
texture at its fixed texel pitch rather than resizing one texture to fill the new
area. First crop a complete repeat cell: check lattice phase and boundary
continuity, not merely equality under `np.tile`. The bundled cell contains two
red, four green and two blue connected bright bodies; none intersects the cell
border at a 0.4 sRGB threshold. Wrap-neighbor RGB jumps are lower than ordinary
within-cell neighbor variation in both axes. Do not blend or repaint its edges.
The physical scale remains an illustrative assumption, not panel calibration.
The bundled data uses
1000x1000 height texels at 1um and 500x500 emission texels at 2um, a 0.5um height
range and a 1mm x 1mm field. The current height surface contains randomly placed,
nonoverlapping circular pits with uniformly drawn **3-5um diameters**, **0.5um
depth** and **50% footprint coverage** (39,048 pits, seed 20261107). Pits have a
smooth bowl profile `h/depth = 1 - (1 - (r/R)^2)^2` within the radius; the plateau
is white and the pit bottom is black. Diameters are drawn before placement;
largest-first placement with random positions avoids overlap without redrawing
the diameter population. Geometry and generation assumptions are recorded in
`assets/oled-1mm-pits3to5um/heightfield.source.json` and `pits.csv`. Use
50um measurement pixels (20x20, 400 total), zero ROI margin, a 500um gap,
530nm effective wavelength and camera NA 0.08. RGB input is gamma-decoded and
mapped to relative luma for this single-wavelength example.

The first three backends implement scalar ASM + phase-screen imaging. Mitsuba
uses geometric refraction and pupil quadrature. Their brightness maps are the
outputs to compare; the models remain distinct. Use source-sampling and grid
checks as needed, without changing the requested direct pixel statistic.

## Run

```bash
python scripts/run_oled_demo.py --backend all --out outputs/oled-1mm-pits3to5um-periodic-50um
```

The wrapper selects Torch's available device with `--device auto` and uses up to
three independent backend workers. On Windows, Chromatix automatically runs in a
WSL2 CUDA worker; Torch/TorchOptics and Mitsuba run in the Windows process. Read
[references/windows-wsl.md](references/windows-wsl.md) for the WSL environment,
installation, and distribution/Python overrides. The WSL worker forces
`JAX_PLATFORMS=cuda` and reports a failure if CUDA or WSL is unavailable.
On Linux, Chromatix uses the local JAX installation. Actual versions, devices,
and the WSL Python environment are recorded in each backend's execution context.
Override `--device`, `--backend-workers` or
`--mitsuba-variant` as needed. Each 50um measurement pixel averages 50x50 optical
samples; emission remains at 2um during the height refinement.
The 1um grid gives only 3-5 samples across a pit diameter. Check optical-grid
sensitivity before treating small CV changes as resolved: regenerate the same
geometry with the following command, keeping the physical FFT padding and
emission sampling unchanged. Compare the original surface on the same refined
grid too.

```bash
python scripts/prepare_random_pits.py --pitch-um .5 --out assets/oled-1mm-pits3to5um-pitch0p5
```

The original interpolated surface remains available with `--height-surface original`.
Superseded layouts are available only for historical reproduction with
`--emission-layout legacy-stretched` or `--emission-layout legacy-whole-tile`.
Neither the stretched nor the uncropped-patch CV comparison describes the current
cropped-cell example. The default `--emission-layout periodic` crops before tiling.
For custom inputs use `python scripts/sparkle_multibackend.py --help`.
Use `--dry-run` before large calculations; it writes a report marked uncomputed.

Every calculation writes `report.html`, `run.json`, `comparison.json/csv` and,
per backend, `brightness.npy/png` (optical grid), `analysis_brightness.npy/png`
(measurement pixels), `roi_mask.npy` (measurement grid) and `stats.json`.
Failed, skipped and dry-run values must remain uncomputed, not zero.

Regenerate HTML without optical recomputation:

```bash
python scripts/generate_report.py --results outputs/oled-1mm-pits3to5um-periodic-50um
```

`--recursive` combines saved runs; repeated `--figure path.png` embeds plots.
Always deliver the HTML report and code/output paths. State which backends ran.
Use `VALIDATION.json` for the latest measured result and test status.
