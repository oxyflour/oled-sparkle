---
name: oled-ag-sparkle
description: Simulate OLED brightness with PyTorch FFT, TorchOptics, Chromatix and Mitsuba; report raw brightness CV or explicitly requested full-green SMS-reference separated sparkle, with input images, four-solver comparisons, parameters and timings in offline HTML.
---

# OLED brightness and separated sparkle reporting

The direct CV below is the default statistic. When the user explicitly requests
SMS-style separation of display periodicity, use the additional reference workflow
described below and keep separated sparkle contrast distinct from direct CV.

The bundled default heightfield is the etched-AG Voronoi surface in
`assets/oled-1mm-etched-ag-spacing6p9-sz1um/heightfield.png`. `run_oled_demo.py` is the current
default-example entrypoint; `run_legacy_demo.py` requires an explicit historical fixture.

For transmission haze of the heightfield, use the separate `ag-haze` skill and
`scripts/calculate_haze.py`. Haze uses uniform illumination and angular flux,
not OLED emission or camera brightness; its monochromatic surface-only estimate
must remain distinct from luminous-haze measurements and sparkle contrast.

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

Present inputs and their physical scales first, followed by optical/measurement
conditions, the four-solver results, and validation. Give quantitative image
previews physical coordinates, units and color-scale definitions where relevant.

1. The actual height map and emission image, their paths, SHA-256, dimensions,
   bit depth, physical scales and the emission source URL. Preserve original PNG
   bytes for download, including 16-bit height values.
2. One collected brightness image for each of `torch`, `torchoptics`, `chromatix`
   and `mitsuba`, plus raw floating-point brightness downloads.
   Keep the four-solver results, timings and image comparison visible in the main
   report, including sparkle-only views. Raw total-brightness CV may be folded
   into an appendix without hiding the solver comparison.
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
The default has 1000x1000 height texels at 1um and 500x500 emission texels at
2um over a 1mm x 1mm field. Its synthetic etched-AG surface has **21,004 irregular
Voronoi bowls**, **6.9um nominal site spacing**, variable depths and **1.2um rim
rounding**, with seed 20261107. A shared 0.5um reference grid sets post-rounding
**Sz=1um**; fine-grid Sa is about 0.128404um and Sq about 0.159244um.
The 16-bit PNG black-to-white encoding range is **1.04um**.
Read this range from `assets/oled-1mm-etched-ag-spacing6p9-sz1um/heightfield.source.json`,
as the wrapper does. Do not substitute Sz=1um for the encoding range.
The former 25um/Sa0.1um maps remain in `assets/oled-1mm-etched-ag/` and its
`-pitch0p5` counterpart for historical reproduction. The new default is an
exploratory assignment of Table 1's 6.9um statistic, not measured glass.
`scripts/prepare_etched_ag.py` and `sites.csv` record the generation and geometry.
This is an illustrative proxy, not measured commercial glass; areal Sa and
equivalent cell diameter do not equal manufacturer line Ra and Rsm. Use
50um measurement pixels (20x20, 400 total), zero ROI margin, a 500um gap,
530nm effective wavelength and camera NA 0.08. RGB input is gamma-decoded and
mapped to relative luma for this single-wavelength example.

The first three backends implement scalar ASM + phase-screen imaging. Mitsuba
uses geometric refraction and pupil quadrature. Their brightness maps are the
outputs to compare; the models remain distinct. Use source-sampling and grid
checks as needed, without changing the requested direct pixel statistic.
CUDA Mitsuba executes the complete ray/refraction/emission integration on the
GPU with a symbolic pupil loop, respecting `--ray-chunk` and preserving the
mesh/pupil sampling. It downloads only the final brightness image. Non-CUDA
variants retain the NumPy integrator; reports record the integrator used.

## Run

```bash
python scripts/run_oled_demo.py --backend all
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
For the same default geometry at a finer height pitch, the same runner retains
the shared height calibration, physical FFT padding and fixed 2um emission sampling:

```bash
python scripts/run_oled_demo.py --pitch-um .5 --backend torch
```

The original interpolated surface remains available with `--height-surface original`.
The former circular-pit example is available with `--height-surface random-pits`.
The small smoke-test fixture requires `run_legacy_demo.py --example quick`; its height
is separate from the default display-cover example. `--example 1mm` retains the
original analytical 1mm fixture.
Superseded layouts are available only for historical reproduction with
`--emission-layout legacy-stretched` or `--emission-layout legacy-whole-tile`.
Neither the stretched nor the uncropped-patch CV comparison describes the current
cropped-cell example. The default `--emission-layout periodic` crops before tiling.
For custom inputs use `python scripts/sparkle_multibackend.py --help`.
The default etched surface is selected by `run_oled_demo.py` or explicitly with
`--height-surface etched-ag`; no surface-specific execution wrapper is needed.
Its generator `scripts/prepare_etched_ag.py` defaults to 21,004 Voronoi
bowls at 6.9um nominal site spacing, variable depths and 1.2um rim rounding.
The reference-grid calibration fixes post-rounding Sz at 1um. Explicit
`--sa-um` selects Sa normalization instead. Read the actual PNG
height range from its sidecar, as the wrapper does. Areal Sa and equivalent cell
diameter are not manufacturer line Ra and Rsm. This is a synthetic morphology
proxy, not a measured product. Use `--pitch-um .5 --backend torch` with a separate
output directory for the same geometry/calibration on the finer optical grid.
Use `--dry-run` before large calculations; it writes a report marked uncomputed.

Every calculation writes `report.html`, `run.json`, `comparison.json/csv` and,
per backend, `brightness.npy/png` (optical grid), `analysis_brightness.npy/png`
(measurement pixels), `roi_mask.npy` (measurement grid) and `stats.json`.
Failed, skipped and dry-run values must remain uncomputed, not zero.

Regenerate HTML without optical recomputation:

```bash
python scripts/generate_report.py --results outputs/oled-1mm-etched-ag-spacing6p9-sz1um-periodic-50um
```

`--recursive` combines saved runs; repeated `--figure path.png` embeds plots.
Always deliver the HTML report and code/output paths. State which backends ran.
Use `VALIDATION.json` for the latest measured result and test status.

## Explicitly requested SMS-reference measurements

`scripts/run_sms_reference.py` prepares the unchanged G channel with R/B zero,
then computes the existing etched AG without LDF. A `prepare` stage writes the
inputs and fixed protocol before optical calculation. `main` runs all four backends;
`fine` repeats Torch at 0.5um and `old-na` checks the previous NA 0.08.
Use `--stage all` for the complete workflow; its new default output is
`outputs/oled-1mm-etched-ag-spacing6p9-sz1um-sms-reference`.
The SMS workflow defaults to one backend worker so all four solvers run
sequentially on the same GPU. Its prepared protocol records a batch ID,
declaration timestamp and solver SHA-256. All main and sensitivity runs in a
report must start after that declaration, use the recorded current solver and
match their backend run IDs. Show a single timing column from those same runs.
The workspace wrapper `test-results/run-sms-reference.ps1` targets fixed
historical result paths; use the maintained CLI for the new default.
The SMS runner reuses the default
runner's solver command; keep the measurement settings explicit and distinct.

Rebuild the separated sparkle report without optics through the existing report CLI:

```bash
python scripts/generate_report.py --kind sms-reference --results outputs/oled-1mm-etched-ag-spacing6p9-sz1um-sms-reference
```

`reporting.py` owns both report renderers; `sms_reference.py` owns separation and
saved-float analysis. Report generation updates result artifacts only; it does
not prepare inputs, rewrite the protocol, edit skill metadata or package archives.

`scripts/sms_reference.py` implements the public Fourier/MWA approach with explicit
choices, not proprietary SMS software. It removes known 140um supercell frequencies,
retains DC and limits bandwidth below 1/70um^-1. Evaluate a central 560um square,
then integrate to 26.25um virtual camera pixels. The reference camera is 50mm,
F/8, 400mm object distance, 3.75um sensor pitch, giving object NA about 0.007812
and magnification 1/7. Filtering precedes virtual detector integration, so detector
aliasing before filtering is omitted. Compare predeclared notch/cutoff changes and
three 140um MWA passes; they have different bandwidths, so equality is not expected.

The user-requested separation supersedes the default no-filter requirement for this
workflow. Preserve raw optical brightness and direct CV too. A fast analytical flat
wave control diagnoses residual periodicity/boundaries; do not divide by it or
subtract its CV. The report must distinguish an order-of-magnitude comparison
against Cheng 2025's 4.8-6.0 readings from calibration: the paper's camera/filter
settings are unpublished and its surface Ra differs from this proxy's areal Sa.
