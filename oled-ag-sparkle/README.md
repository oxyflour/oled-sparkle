# OLED brightness and separated sparkle skill

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

The pinned Chromatix source requires Python 3.12+. On Windows, Chromatix is
automatically delegated to a WSL2 CUDA Python environment; install that environment
using [the WSL setup instructions](references/windows-wsl.md). Windows does not need
its own Chromatix/JAX installation. Torch/TorchOptics and Mitsuba continue running
in the Windows process. On Linux, install Chromatix and a suitable local JAX build.
The wrapper uses `--device auto` and up to three independent backend workers;
the HTML records the actual WSL device, Python environment, versions and timing.

## Reproduce the sourced OLED example

The supported script entrypoints are:

| Script in `scripts/` | Responsibility |
|---|---|
| `run_oled_demo.py` | Current default OLED example, all four backends. Select `--height-surface` and `--pitch-um`; computes raw brightness CV. |
| `run_sms_reference.py` | Full-green, no-LDF separated sparkle. Default `--stage all` runs six reference cases; `--layouts` compares layouts; `--surface-study width/spacing-distribution` runs the two AG surface comparisons. |
| `generate_report.py` | Rebuild saved reports without optics. Default `--kind brightness`; choose `--kind sms-reference` explicitly for separated sparkle. |
| `calculate_haze.py` | Monochromatic heightfield haze: NumPy estimate or `--backend all` for actual four native backends, with uniform illumination and saved angular-power data. |
| `sparkle_multibackend.py` | General solver CLI for custom inputs and physical/numerical parameters. |
| `run_legacy_demo.py` | Historical test fixtures only; requires `--example quick` or `--example 1mm`, defaults to Torch. |
| `prepare_*.py` | Input generators used by the runners, or directly for input-only work. |

`reporting.py`, `sms_reference.py` and `chromatix_wsl.py` are imported
implementation modules, not additional user entrypoints. The run, solver and
report CLIs support `--help` and resolve bundled assets independently of the
working directory.

The legacy runner's `quick` fixture is a 128x128 synthetic grid at 2um, used by
CLI, report and backend regression tests. Its `1mm` fixture is the original
2000x2000 synthetic rough surface at 0.5um, with basic pixel or synthetic green
subpixel emission, retained for historical parameter checks and reproduction.

```bash
python scripts/run_oled_demo.py --backend all
```

The source image is Samsung Display's Diamond Pixel OLED arrangement:
https://global.samsungdisplay.com/29043/

The height and emission inputs cover **1 mm x 1 mm**. The default height map is
`assets/oled-1mm-etched-ag-spacing6p9-sz1um/heightfield.png`: **1000x1000 at 1um**,
16-bit grayscale, with **21,004 irregular Voronoi bowls**, **6.9um nominal site
spacing**, variable depths, **1.2um rounding**, and seed **20261107**. The shared
0.5um reference grid targets **post-rounding Sz=1um**; fine-grid Sa is about
0.128404um and Sq about 0.159244um. The PNG black-to-white encoding
range is **1.04um**, read from its sidecar without per-grid
renormalization. This surface is a synthetic etched-glass proxy, not a measured
sample. `run_oled_demo.py` is the sole current default-example entrypoint.
For a spacing-distribution study at fixed site density, the existing generator
accepts `--min-separation-ratio` (default `0.3`). It sets the minimum periodic
site distance divided by nominal spacing: `0` gives unrestricted random sites;
larger values exclude close neighbors and produce a more uniform distribution.
For example, from this directory:

```bash
python scripts/prepare_etched_ag.py --out outputs/ag-exclusion-0p7 --spacing-um 6.9 --sz-um 1 --min-separation-ratio 0.7 --pitch-um 0.5
```

The sidecar records actual nearest-neighbor mean, standard deviation and CV,
plus equivalent-cell-diameter statistics. Nominal spacing is a density scale,
not mean nearest-neighbor distance. Changing exclusion changes spatial
correlations and actual neighbor means as well as dispersion; it is not an
isolated variance adjustment. Sz is fixed after rounding, while Sa/Sq and the
individual random depths can vary. Use separate outputs for each case.

The former 25um / Sa0.1um maps remain at `assets/oled-1mm-etched-ag/` and
`assets/oled-1mm-etched-ag-pitch0p5/`; historical results still reference them.
The emission input remains 500x500
at **2um**. Brightness is collected into **50um x 50um pixels**: each averages 50x50
optical samples, producing a 20x20 image. All 400 pixels enter the CV calculation;
the ROI margin is zero. The wrapper uses the height range from the input sidecar.

Only the Diamond Pixel panel is cropped and initially resampled to a 128x128
image at an assumed 2um texel pitch, covering 256x256um. This patch is not itself
a repeat cell. Detect the 70-texel periods, then crop [26,58,96,128] to a 70x70
cell covering 140x140um. Repeat that cell 8x8, then crop the 560x560 canvas to
500x500 from the top left. No scaling or interpolation occurs during field expansion.
`scripts/prepare_tiled_emission.py` records source/cell hashes, period detection,
boundary statistics, repeat counts and crop in `assets/oled-1mm-periodic/emission.source.json`.
The cell contains 2 red, 4 green and 2 blue connected bright bodies, none touching
its boundary above 0.4 sRGB. Wrap-neighbor jumps are below ordinary interior
neighbor variation in both axes. No blending or pixel repainting is used.
The image-derived physical scale is still an assumption. The original web image retains
Samsung Display's copyright. The height is synthetic, generated by
`scripts/prepare_etched_ag.py`; site coordinates and relative bowl depths are in
`assets/oled-1mm-etched-ag-spacing6p9-sz1um/sites.csv` and provenance is in the adjacent
`heightfield.source.json`. Reproduce the former circular pits with
`--height-surface random-pits`, or the interpolated height with
`--height-surface original`. Small smoke tests use `run_legacy_demo.py --example quick`.
Earlier runs stretched the texture to fill 1mm, enlarging emitters and spacing
by 500/128 = 3.90625. Those comparisons are superseded. Use
`--emission-layout legacy-stretched` only to reproduce that historical layout;
The next historical attempt repeated the entire 128x128 patch, producing phase
breaks at its edges; use `--emission-layout legacy-whole-tile` only to reproduce
that superseded layout. Current comparisons recompute both height surfaces using
the cropped repeat cell (`--emission-layout periodic`, the default).

For a 0.5um grid of the same default surface, use
`python scripts/run_oled_demo.py --pitch-um .5 --backend torch`.
The geometry and height calibration remain fixed; emission remains at 2um and
measurement pixels remain at 50um. The old circular-pit sensitivity workflow
`test-results/check-pit-sampling.py` remains a historical comparison.

## Equal-pitch layout comparison with sampled camera

The sampled-camera result is an internal **periodic-notch contrast**, not an
SMS1000-equivalent visual sparkle reading. The public manufacturer description
includes numerical low-pass filtering after capture; this retained benchmark
has `additional_lowpass=None`. The 2026-11-08 metrology audit below documents
this known mismatch without altering historical filters or numerical results.

From this directory, reuse the verified ideal-pupil optical fields and write a
new, separately frozen camera analysis:

```bash
python scripts/run_sms_reference.py --layouts diamond rgb-stripe rgb-delta windmill --measurement sampled-camera --reuse-optics ../test-results/oled-layout-comparison --stage all --out ../test-results/oled-camera-new
python scripts/generate_report.py --kind sms-reference --results ../test-results/oled-camera-comparison
```

The second command rebuilds the retained report without optics. Omit
`--reuse-optics` to compute a new optical batch through the same runner.
Default `--backend all` runs four backends at 1um plus Torch at 0.5um per layout
(20 runs); camera comparison uses only the three scalar-wave backends because
Mitsuba geometric rays omit diffraction. `--backend torch` narrows the coarse
runs. New batches support prepare/main/fine stages; reused prepare freezes the
capture plan and hashes, and is followed by the report CLI. Existing protocols
refuse another prepare/all. Reused optical stages cannot be rerun accidentally.

Fresh optical batches use the current 6.9um/Sz1um default AG; the retained
`oled-camera-comparison` batch uses its original 25um/Sa0.1um AG. Both use a
500um gap, 530nm green and no LDF. Nominal
logical address pitch P=70um means area/address=P squared, not equal nearest
neighbor subpixel spacing. Full 1mm input mean linear green is 1 after decoding,
with unclipped solver gain and threshold=0. Emission texels/source spacing remain
2um. Exact G periods X/Y are Diamond 140/140, stripe 70/70, Delta 210/140,
Windmill 140/140um. The original Diamond cell is retained; the other three are
representative analytic polygons, not measured commercial apertures.

Camera parameters are 50mm, F/8, 400mm object distance, 3.75um square sensor
pixels, magnification 1/7 and object sampling pitch 26.25um. The ideal circular
pupil is already in the wave solver; do not convolve with another Airy PSF.
Order: incoherent source propagation -> AG phase screen -> focused pupil imaging
-> square pixel area integration -> discrete samples -> common Fourier mask
-> inverse transform -> 100*std(ddof=0)/mean. Pixels are integrated only once.
This is an explicit aberration-free, linear monochrome, 100% fill, noiseless,
unquantized camera model, not a measured lens/sensor calibration.

Frequency masking follows the method family in IEC 62977-3-9:2023 informative
Annex B and Table B.1, with camera sampling/aliasing discussed in Becker (2021)
sections 3.1-3.4. These sources do not prescribe our exact mask:
- [IEC publication](https://webstore.iec.ch/en/publication/64780)
- [Author's 2021 paper](https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf)

The 840um ROI is 32x32 camera pixels and an integer multiple of every full G
period. Enumerate reciprocal vectors within the optical intensity bandwidth
(2*NA/lambda = 29.480 cycles/mm), fold modulo sensor sampling frequency
(Nyquist 19.048 cycles/mm), and remove their union for all layouts. This fixed
common mask removes 89 non-DC bins and retains 934, with DC preserved. There are
no hand-built MWA windows, guard bands, amplitude thresholds, apodization or
extra low-pass. Overlapping periodic/random energy in a removed bin is
unidentifiable and both are removed. This metric measures camera-domain
stochastic contrast; it does not implement the additional visual weighting or
high-frequency suppression discussed by Becker, and is not a calibrated SMS or
full IEC conformity test.

Retained Torch results (1um / 0.5um) are Diamond 24.016988 / 23.977983%,
stripe 8.880589 / 8.715334%, Delta 26.721904 / 26.486939%, Windmill
24.147108 / 24.082596%. Maximum grid shift is 0.235 percentage points. Actual
FFT pupil MTF differs from the analytic circular-pupil MTF by at most 0.00333.
Finite-field flat control is at most 0.642%, while infinite periodic flat control
is below 3e-13%; neither is subtracted. Four sensor phases, a center subregion
of the full filtered image and an independently processed 420um ROI are reported.
The last check also doubles DFT bin width. One AG realization, source phase and
small field do not establish commercial-panel rankings; Diamond/Windmill are
closer than the reported phase/ROI sensitivities.

`test-results/oled-camera-comparison/` contains only protocol/source hashes,
summary/per-run camera statistics, captured pixels (`camera_pixels.npy`),
separated pixels (`camera_separated.npy`), the common mask, flat controls and
plots/HTML generated by the existing report CLI. Source optical arrays are
read-only and linked with hashes. No new scripts or execution entrypoints were
added. The old `oled-layout-comparison` custom MWA report is preserved, selected
explicitly for new historical reproductions with `--measurement optical-mwa`.
Its three-window result and historical 5.291803% Diamond baseline use another
frequency response and cannot substitute for the sampled-camera metric.

Input assets remain under `assets/oled-layouts/<layout>/`: RGB cell, 1mm RGB and
full-green solver input, plus source/normalization/period/hash metadata. Reference
pictures in `assets/oled-layout-reference/` are review material. The existing
`prepare_tiled_emission.py --layout rgb-delta` generates individual inputs.
`--pixel-pitch-um` and `--emission-texel-um` change inputs; the fixed 840um camera
ROI requires compatible full periods.

## Surface transmission haze

The independent `../ag-haze/SKILL.md` invokes `scripts/calculate_haze.py`;
`../ag-haze/agents/openai.yaml` provides discovery metadata. This new angular
observable cannot use the existing camera-pupil-clipped brightness arrays.
The calculator reuses the existing PNG height reader and roughness function;
it does not change the sparkle solver, input heights or their frozen hashes.
Run the current default with `python scripts/calculate_haze.py`. The following
explicit inputs reproduce the retained 25um study into a new directory:

```bash
python scripts/calculate_haze.py --height assets/oled-1mm-etched-ag/heightfield.png assets/oled-1mm-etched-ag-pitch0p5/heightfield.png --wavelength-nm 450 530 650 --out ../test-results/ag-haze-25um-reproduction-new
```

Defaults: current 6.9um/Sz1um AG, 530 nm, n_in=1.5, n_out=1, normal incidence,
2.5-degree acceptance half-angle. The current default gives 41.206719% on the
1um grid and 41.199873% at 0.5um. The command above explicitly selects the
historical 25um/Sa0.1um pair: its 530 nm estimate is 2.730950% on the 1um
grid and 2.730997% at 0.5um. These are surface-only scalar results, not a match
to the paper's 20-40% haze samples or a standard luminous-haze measurement.

Read pitch and black-to-white encoding range from the sidecar; never use Sz
as the range. Custom maps require calibration and justified periodicity;
`--assume-periodic` acknowledges that assumption when metadata does not declare it.
No aperture, padding or apodization is added. The output `haze.json` records
source/code hashes, parameters, haze, flat null, Parseval error, spectral edge
power and angular coverage. `*-angular.csv` records the cumulative angular flux.
Finite frequency coverage and unmeasured fine roughness remain limitations;
refining a genuinely sampled surface is different from interpolating an image.

The implemented 2D angular integral includes all represented propagating forward
flux in its denominator and uses Cartesian Fourier-bin weight |A|^2 cos(theta).
No camera NA or OLED source enters this calculation. The multi-wavelength
option gives independent monochromatic values; luminous haze requires the
illuminant, V(lambda), dispersion, transmitted-power weights and instrument
correction. See [physics-and-backends.md](references/physics-and-backends.md)
for the definition, derivation and sources.

For actual native-backend execution, use a separate protocol directory:

```bash
python scripts/calculate_haze.py --backend all --height assets/oled-1mm-etched-ag/heightfield.png assets/oled-1mm-etched-ag-pitch0p5/heightfield.png --out ../test-results/ag-haze-four-backend-new
python scripts/generate_report.py --kind haze --results ../test-results/ag-haze-four-backend
```

The retained comparison ran eight main GPU jobs and native flat controls at
530nm, n=1.5->1 and 2.5 degrees. Torch, TorchOptics and Chromatix give
2.730950% (1um) / 2.730997% (0.5um), matching the NumPy phase-screen reference.
Mitsuba gives 0.553996% / 0.613183%. Its +0.059187 percentage-point grid shift
(about 10.7% relative) remains visible; do not label it fully grid-converged.
The three wave libraries actually propagate their plane-illuminated phase fields
100um through the outgoing homogeneous medium, whose propagation phases leave
angular powers unchanged. Agreement is numerical consistency of one approximate
boundary model. Mitsuba reuses the existing mesh constructor with periodic
endpoints and traces one normal-incidence ray per projected triangle centroid,
using native Fresnel/refraction and geometric normals. It includes Fresnel
weights but omits diffraction. Neither model is established as measured-glass
truth by this comparison.

The calculator freezes inputs and code hashes before native execution and
refuses an existing native protocol. `--backend` can select one backend;
`--device`, `--propagation-um`, `--mitsuba-variant` and `--ray-chunk` control its
declared execution. Windows Chromatix reuses the existing CUDA-only WSL adapter.
The original OLED solver and frozen solver hash are unchanged.
Each case stores `stats.json`, `angular.csv` and (for waves) the actual unshifted
Fourier-bin `angular_power.npy`. The report CLI audits saved powers and curves,
ray coverage and energy balance, then writes `validation.json`, plots and
`haze.html` without launching solvers. Inputs, runtime contexts and timings are
in `haze.json` and `protocol.json`; no new wrapper scripts are introduced.

### Investigating the paper discrepancy (2026-11-08)

Rebuild an **independent** audit report from the saved native data:

```bash
python scripts/generate_report.py --kind haze --results ../test-results/ag-haze-four-backend --out ../test-results/ag-haze-audit-20261108/haze.html
python -m unittest discover -s tests -p test_haze.py -v
```

In haze mode, `--out` now writes the HTML, figures, `diagnostics.json` and
`validation.json` beside the requested HTML; it does not refresh the source
directory. Without `--out`, derived files are refreshed in `--results` as before.
The offline report computes height PSD/periodic radial ACF, height moments,
facet slopes, an analytic Snell/Fresnel check and the DC/in-cone/outside flux
partition. These are diagnostics, not new native solver runs. The independent
TE/TM shallow-interface reference is maintained in the existing `test_haze.py`
and invoked by its `InterfaceBenchmarkTests`, not a production AG backend.

The current 530nm wave flux is 60.613% DC + 36.656% inside-cone non-DC +
2.731% outside-cone. Counting all non-DC orders would give 39.387% but changes
the measurement definition. The radial height ACF first 1/e length is about
7.32um; the 2.5-degree spatial-frequency cutoff corresponds to 12.15um lateral
period, rather than the nominal 25um site spacing. Analytic geometric sampling
at 0.25/0.125um gives 0.62713/0.63123%, without additional Mitsuba execution.
An illustrative D65/V integration gives 2.868% at fixed n=1.5; using N-BK7
dispersion gives 3.129%. Neither identifies the paper glass or implements a
standard instrument. Detailed interpretation, equations, evidence and missing
measurements are in [physics-and-backends.md](references/physics-and-backends.md).
That discussion also compares real AGC product specifications (Ra, Rsm and
haze) and multiscale etched-glass measurements; these do not identify the
paper's sample or supply its missing PSD.

New retained data live only in `../test-results/ag-haze-audit-20261108/`:

| Artifact | Purpose / producer |
| --- | --- |
| `haze.html`, `haze-comparison-530nm.png`, `haze-surface-diagnostics.png`, `haze-spectral-sensitivity.png` | HTML and figures from the existing report CLI; HTML is ignored by Git and rebuilt locally. |
| `diagnostics.json`, `validation.json` | Offline surface/flux curves and audits of the original eight saved runs, produced by the report CLI. |
| `geometric-grid.json` | Temporary research command reused `prepare_etched_ag.geometry/raw_surface` and `calculate_haze.facet_slope_diagnostics`; contains the fixed calibration and independent 1/0.5/0.25/0.125um evaluations. No new height input was saved. |
| `interface-benchmark.json` | Temporary research command called `test_haze.rayleigh_sinusoid`; retains diffraction orders, TE/TM powers, flat controls, boundary residuals and convergence. Run `InterfaceBenchmarkTests` to reproduce the numerical checks. |
| `spectral-sensitivity.json`, `spectral-sensitivity.csv` | Temporary research command called the existing `load_surface/transmission_haze` functions on the 0.5um input: 380–780nm in 5nm steps, fixed n=1.5, N-BK7 n(530nm), and N-BK7 Sellmeier dispersion. JSON records coefficients, quadrature, approximate transmission weights, source hashes and every result. The existing calculator CLI can reproduce each fixed-index monochromatic sweep with `--wavelength-nm` and `--n-in`; no second spectral solver is maintained. |
| `spectral-inputs/CIE_std_illum_D65.csv`, `spectral-inputs/CIE_sle_photopic.csv` | Official CIE tables, checked against CIE's published MD5 values; URLs and SHA256 are in the spectral JSON. |
| `tests.log`, `source-protection.json` | Test output and before/after hashes confirming that the original inputs and historical haze results were preserved. |

The report reads the three optional research JSON files above from its output
directory when present; rebuilding it never reruns those experiments. No new
AG material has been designed or calibrated to the paper's haze range.

### Texture scale, depth and morphology controls

The separate study in `../test-results/ag-haze-texture-20261108/` holds the
530nm, n=1.5->1, 2.5-degree measurement fixed. `texture-sensitivity.json/csv`
records a 6-by-4 matrix of lateral factors 0.25/0.5/0.75/1/1.5/2 and height
factors 0.5/1/2/3 applied to the original calibrated 0.5um samples. Lateral
dilation changes pitch, domain, nominal site spacing and smoothing length
together; it is not a change in grid sampling at fixed physical geometry.
The 25um nominal site spacing is not a strict grating period. These are
controlled counterfactuals, not optimized or calibrated new AG products.

`spacing-sensitivity.json` additionally holds physical Gaussian sigma=1.2um
and Sq fixed while independently generating scaled site layouts. This separates
nominal spacing from the smoothing change implicit in whole-map dilation:
25/12.5/6.25um nominal spacing gives 2.731/13.984/27.458% respectively.

At fixed height, halving the lateral dimensions gives 17.256% instead of
2.731%. At fixed lateral scale, doubling/tripling height gives 15.902%/38.227%.
The 1mm periodic-box repeat control and height-sign inversion retain 2.731%.
Repeating a box does not test larger independent random samples. A matched-Sq
morphology experiment separately changes bowl power and Gaussian rounding;
its nine calibrated cases are stored under `cases/<case>/heightfield.png` and
`.source.json`. Large derived PNGs are kept locally and ignored by Git; their
sidecars retain parameters, normalization, source identity and hashes.

The existing `prepare_etched_ag.py` now accepts `--profile-power` (finite, >=1).
Its default 2 reproduces both original PNGs byte-for-byte. The same raw function
is used by the morphology study, avoiding a second bowl generator. For a
separate Sa-matched bowl example, and for an already saved study case:

```bash
python scripts/prepare_etched_ag.py --profile-power 4 --sa-um .1 --out ../test-results/ag-power4-example
python scripts/calculate_haze.py --height ../test-results/ag-haze-texture-20261108/cases/power4/heightfield.png --out ../test-results/ag-power4-haze-new
python scripts/generate_report.py --kind haze --results ../test-results/ag-haze-four-backend --out ../test-results/ag-haze-texture-20261108/haze.html
```

The first example matches Sa through the established 0.5um reference grid;
it is not a byte-identical reproduction of the study's Sq-matched 0.25um case.
To reproduce a study case exactly, its sidecar specifies `raw_surface` arguments,
raw-to-um normalization and encoding step. All new experiments used temporary
commands calling maintained functions; no new scripts were left in results.

| New artifact in the texture study | Purpose / producer |
| --- | --- |
| `texture-protocol.json`, `texture-sensitivity.json`, `texture-sensitivity.csv` | Declared scale/depth factors, source hashes, definitions and model results from `transmission_haze` and `facet_slope_diagnostics`. |
| `morphology-protocol.json`, `morphology-sensitivity.json` | Fixed sites/depths; bowl power 1/2/4; rounding 0.3/0.6/1.2/2.4/4.8um; explicit Sa/Sq normalization, input identities and diagnostics. |
| `spacing-sensitivity.json` | Six independently sampled nominal-spacing controls with fixed physical rounding and Sq, including their declared protocol and calibrated input identities. |
| `cases/*/heightfield.png`, `cases/*/heightfield.source.json` | Fifteen distinct calibrated counterfactual inputs (nine morphology and six spacing cases); created from `raw_surface`, then read through the existing calibrated loader. No original asset was changed. |
| `phase-surrogates.json` | Three conjugate-symmetric random-phase controls preserving the exact 2D height PSD and Sq, while changing height PDF and higher-order correlations. |
| `sampling-check.json`, `default-generation-check.json` | Independent 0.25/0.125um check of the sharpest rounding case, and SHA256 proof that default generation retains both original PNGs. |
| `haze-texture-sensitivity.png`, `haze-morphology-sensitivity.png` | Factor heat maps and morphology/cross-section comparison from the existing report CLI. |
| `haze.html`, `haze-comparison-530nm.png`, `haze-surface-diagnostics.png`, `diagnostics.json`, `validation.json` | Derived report and audit of the original eight native runs; generated in the new directory without updating the old results. |
| `tests.log`, `source-protection.json` | Appropriate checks and hashes verifying preservation of inputs and all earlier haze results. |

The report reads retained study data and never reruns its perturbation experiments.
See [physics-and-backends.md](references/physics-and-backends.md) for what can
and cannot be attributed to each factor without measured maps of the paper glass.

## Outputs

For an explicitly requested **SMS-inspired green-only measurement**, use:

```bash
python scripts/run_sms_reference.py --stage all
```

It keeps the etched AG geometry and 500um gap, writes the fixed protocol before
calculation, runs six green-only optical cases without LDF and generates
`sparkle.html` and `sparkle-report.html` in that output directory. Individual stages
are `prepare`, `main`, `fine`, and `old-na`; an optical stage requires the prepared
protocol. The workspace PowerShell wrapper `test-results/run-sms-reference.ps1`
selects the local CUDA environment and saved result paths.
The raw solver report retains direct CV; `sms_reference.py` additionally separates
known periodic frequencies and evaluates random contrast with an explicit bandwidth.
It also reports a three-pass full-supercell MWA alternative, flat-wave control,
finer-grid check and original-NA comparison. This is a public-method reference,
not the proprietary SMS-1000 algorithm or an IEC compliance claim. Parameters
are recorded in `protocol.json` and are not fitted to the literature target.

The default **illustrative etched-AG morphology** is generated through
`scripts/prepare_etched_ag.py`, called by `scripts/run_oled_demo.py`. It uses 21,004
random Voronoi bowls in a periodic 1 mm square, nominal site spacing 6.9 um,
independently varied bowl depths and 1.2 um Gaussian rounding. A common 0.5 um
reference grid fixes post-rounding Sz = 1 um; the actual PNG
black-to-white range is recorded in `heightfield.source.json`. Cell diameter
and areal Sa are not the manufacturer's line Rsm and Ra specifications.
The model is synthetic and is not fitted to measured commercial glass.

### Three-times-depth Diamond camera sparkle

`../test-results/oled-depth3-camera-20261108/` reruns the full-green Diamond
case with unchanged lateral structure and three times the original physical
height modulation. The two new `inputs/1um/heightfield.png` and
`inputs/0p5um/heightfield.png` preserve the original 16-bit grayscale bytes;
their separately saved `.source.json` files assign black-to-white range
1.608553758411769um rather than 0.5361845861372564um. Sa is about 0.30um and
Sq about 0.35554um. The mean AG distance remains 500um. A shared PNG hash alone
does not identify the physical surface when the calibration range changes.

The predeclared `protocol.json` stores the complete `runs[].argv` for the
existing `sparkle_multibackend.py` CLI and records the original camera protocol
as the comparison reference. The existing SMS runner executes the declared
main/fine stages; no new wrapper or solver is introduced. The 1um run executes
all four native backends, and the 0.5um run executes CUDA Torch. Sampled-camera
sparkle uses the wave models, keeping Mitsuba's geometric direct-CV output
separate. Rebuild the report without optics:

```bash
python scripts/generate_report.py --kind sms-reference --results ../test-results/oled-depth3-camera-20261108
```

Only Diamond is recomputed. The protocol retains all four reference layout
definitions as the basis of the **unchanged** common frequency mask (89 removed
non-DC bins); the renderer displays the layouts actually present in `runs`.
Camera geometry, 26.25um square object-side pixels, 840um/32x32 ROI, emission
normalization, source quadrature and phase checks match `oled-camera-comparison`.
The depth comparison verifies exact mask identity against that saved baseline.
It uses sampled-camera separation, not the historical three-140um MWA metric.

Measured Diamond sparkle is 32.414363% on the 1um Torch grid and 32.565193%
at 0.5um, a +0.150829 percentage-point grid shift. The corresponding original
depth baseline is 24.016988%/23.977983%. Thus the fine-grid depth3 increase is
+8.587209 percentage points (35.81% relative). Main-grid TorchOptics/Chromatix
give 32.414359%/32.414532%, consistent with Torch's shared scalar model.
All five optical jobs used the actual CUDA/WSL GPU backends; the original
solver hash, emission, optics and masks are unchanged. This remains a model
comparison rather than a calibrated SMS instrument value.

| New artifact | Purpose / producer |
| --- | --- |
| `inputs/{1um,0p5um}/heightfield.png` and `.source.json` | Separate depth3 calibration cases; original input bytes and sidecars remain unchanged. |
| `protocol.json` | Frozen optical commands, input calibration, camera/mask and baseline reference hashes. |
| `{1um,0p5um}/diamond/` | Original solver outputs: run metadata, native backend stats, floating optical/measurement images and raw-CV reports; camera pixels, separated pixels, mask and camera stats are added by the existing SMS analyzer. |
| `controls/`, `summary.json` | Existing SMS flat/MTF/phase/ROI checks and complete sampled-camera statistics. |
| `depth-comparison.json`, `height-depth-comparison.png` | Same-mask original/depth3 sparkle values and physical height maps on a shared color scale, generated by `reporting.py`. |
| `layout-inputs.png`, `layout-metrics.png`, `layout-sparkle.png`, `camera-capture.png`, `camera-transfer.png`, `sparkle.html` | Existing report CLI's input, sparkle and camera views; HTML is rebuilt locally. |
| `main.log`, `fine.log`, `report.log`, `tests-reporting.log`, `tests-sms.log`, `source-protection.json` | Actual execution/tests and preservation checks for original assets and earlier results. |

### Paper comparison and metric audit

The separate offline audit `../test-results/oled-depth3-metric-audit-20261108/`
checks why the depth3 result does not reproduce Cheng's approximately 5% SMS1000
reading. On the same saved optical results, the historical three 140um MWA
passes (560um ROI, filtering before 26.25um pixel integration) give:

| Optical grid / surface | Current sampled-camera metric [%] | Historical 140um MWA x3 [%] |
| --- | ---: | ---: |
| 1um / original | 24.016988 | 5.287834 |
| 1um / depth3 | 32.414363 | 8.422367 |
| 0.5um / original | 23.977983 | 5.132102 |
| 0.5um / depth3 | 32.565193 | 7.989120 |

The older unnormalized green optical run reproduces its historical 5.291803%
MWA value. These pipelines have different bandwidths, ROI and operation order;
the legacy MWA diagnostic is not an implementation of the paper's proprietary
instrument. The sampled-camera metric has not been calibrated to SMS1000 either.
Height scaling increases both metrics in this example. Matching scalar haze
alone therefore does not reproduce the paper sparkle, and choosing the filter
that yields approximately 5% would not establish calibration. The MWA 1-to-0.5um
grid shifts remain visible, particularly about -0.433 percentage points for depth3.

`comparison.json` records all nine offline checks, source hashes, fixed MWA
parameters and both available metric values. The nine `*-MWA-pixels.npy` files
retain the actual 21x21 floating measurement arrays from `sms_reference.measure_mwa`;
`source-protection.json` verifies their statistics and unchanged optical sources.
No optical solver or new persistent script was run, and the existing camera
report and its headline metric are preserved. Detailed interpretation is in
[physics-and-backends.md](references/physics-and-backends.md).

### SMS1000 definition and paper-parameter audit

`../test-results/sms1000-metrology-20261108/` audits the public measurement
definition and source provenance. It reuses the depth3 optical files through
the existing `--reuse-optics` branch, retaining exactly the previous camera
mask and 32.565193% fine-grid headline. Rebuild its separate report with:

```bash
python scripts/generate_report.py --kind sms-reference --results ../test-results/sms1000-metrology-20261108
```

Confirmed findings:

- SMS1000 reports the filtered-image standard deviation/mean and explicitly
  includes numerical low-pass filtering. Removing only periodic lines leaves
  extra stochastic frequencies; our current benchmark is not that complete
  measurement. The actual paper kernel, bandwidth and software settings remain
  unknown; none was fitted to 5%.
- The 2023 brochure lists 50mm standard optics, 3.75um CCD pixels, typical
  sparkle aperture >=F/8, and roughly 200–900mm adjustable working distance,
  usually above 500mm. These are device-family capabilities, not Cheng's
  recorded camera settings. Current 400mm and its derived 26.25um object pixel
  are model assumptions; paper camera settings were not reported.
- Paper display pitch is about 97um; the model assigns nominal 70um to a
  sourced Diamond proxy. Paper 145um is LDF-to-pixel distance, not total
  AG-to-emitter gap. It must not replace the current 500um AG gap.
- On the fine-grid depth3 image, frequencies above 1/70um account for 15.369%
  of retained camera variance. The lower-frequency component is still 29.958%
  when normalized by the original mean. This is an independently specified
  frequency audit, not a proposed new SMS score.
- A numerical rectangular-Nyquist decomposition leaves a 32.524% low-band
  component (same mean convention), and the Diamond-only periodic-mask
  diagnostic gives 33.071% rather than 32.565%. Neither explains the full gap to 5%.
  Finite-DFT boundary and covariance qualifications are recorded in the data.

| New artifact | Purpose / producer |
| --- | --- |
| `audit-protocol.json`, `camera-spectrum.json` | Predeclared geometry-based frequency boundaries, Parseval/variance checks, cumulative variance and sampling/mask diagnostics from saved floating images; temporary commands reuse the existing sampler/mask functions. |
| `metrology-audit.json` | Parameter-by-parameter vendor/paper/model audit with confirmed mismatches, assumptions, unknowns and source URLs. |
| `sources.json`, `sources/*.pdf` | Source URLs, SHA256, page counts and exact evidence locations; valid read-only local PDF copies, ignored by Git. The IEC file is only the public six-page preview, not its Annex B text. |
| `metrology-frequency-variance.png` | Existing report CLI's cumulative-variance plot, with fixed geometric reference frequencies and no fitted cutoff. |
| `protocol.json`, `summary.json`, `controls/`, `{1um,0p5um}/diamond/` | Existing reuse-optics camera analysis, raw-source identity checks, controls and pixel arrays; original optical files stay read-only. |
| `depth-comparison.json`, `height-depth-comparison.png`, `layout-*.png`, `camera-*.png`, `sparkle.html` | Existing report CLI's unchanged headline/baseline views plus the new metrology section. |
| `report.log`, `tests-reporting.log`, `tests-sms.log`, `source-protection.json` | Report/test outputs and source-preservation checks. |

This audit identifies several concrete mismatches and limits numerical
explanations, but public evidence does not uniquely apportion the remaining
paper/model difference. A genuine SMS comparison needs its actual instrument
export/settings and measured display/AG geometry. No new filter or optical
solve was introduced.

### Exploratory 6.9um spacing and 1um Sz

`../test-results/ag-spacing6p9-sz1um-20261108/` is a separate user-requested
counterfactual. It assigns **6.9um nominal Voronoi site spacing** and **1um Sz
after Gaussian rounding on the 0.5um reference grid**. Table 1 calls 6.9um a
width variance; assigning it to spacing is an explicit exploration, not a
reconstruction of that statistic or the paper glass. The 1mm domain, seed
20261107, bowl power 2, independent Uniform[0.7,1.3] relative depths and physical
rounding sigma 1.2um are retained. This creates 21,004 sites.

The existing surface generator accepts mutually exclusive `--sz-um` or
`--sa-um` normalization. The exploratory study preserved the original 25um/Sa
inputs; the later default promotion below selects the new Sz-based case.
Example preparation in a **new** directory, from this skill root:

```bash
python scripts/prepare_etched_ag.py --out ../test-results/ag-spacing6p9-sz1um-new/inputs/1um --pitch-um 1 --spacing-um 6.9 --sz-um 1
python scripts/prepare_etched_ag.py --out ../test-results/ag-spacing6p9-sz1um-new/inputs/0p5um --pitch-um 0.5 --spacing-um 6.9 --sz-um 1
```

Each output is independently sampled with the same sites and reference-grid
amplitude calibration; the coarse map is not interpolated. Quantized fine-grid
Sz is 1.000009um and Sa is 0.128404um. **PNG black-to-white height range is
1.04um**, read from the sidecar; it must not be replaced by Sz=1um.

`case.json` records the declaration and generator commands. The frozen
`sparkle/protocol.json` uses commands from the existing SMS command builder and
is executed by `run_sms_reference.py --stage main` / `--stage fine`. Compared
with the retained depth3 run, only optical height path, height range and output
change. Main-grid optics execute all four native backends; fine-grid optics
execute CUDA Torch. The 70um nominal emitter pitch, 140um exact G repeat,
source quadrature, 530nm wavelength, 500um AG gap, camera, ROI and common
four-layout periodic mask remain unchanged. There is no added low-pass or MWA.
The report's optional `surface_comparison` protocol field reuses the existing
baseline/hash/mask checks for surfaces with different lateral geometry;
`height_scale_comparison` continues to check the original height-only scaling.

| Optical grid | Scalar-wave haze [%] | Native Mitsuba haze [%] | Sampled-camera separated sparkle, Torch [%] |
| --- | ---: | ---: | ---: |
| 1um | 41.206719 | 54.691702 | 13.964710 |
| 0.5um | 41.199873 | 56.048276 | 13.789377 |

Coarse TorchOptics/Chromatix sparkle is 13.964793%/13.964572%; their optical
relative L2 differences from Torch are below 7.3e-6. Fine-grid sparkle drops
18.775815 percentage points (57.66%) from the retained depth3 baseline, while
the optical-grid shift is -0.175333 percentage points. Wave haze shifts only
-0.006846 percentage points; geometric haze shifts +1.356575 points (2.48%
relative), so its residual grid dependence is still reported. The wave models
share the thin phase screen, and the wave/ray values are not physical bounds.
Both lateral structure and height statistics change in this case; the sparkle
reduction cannot be assigned to spacing alone. No cutoff was fitted to 5%.

The haze CLI runs all four native backends on the same two maps, with uniform
normal incidence, n=1.5->1 and a 2.5-degree acceptance half-angle. It remains
separate from the camera-pupil-limited sparkle calculation. Rebuild both
reports offline without solving or regenerating inputs:

```bash
python scripts/generate_report.py --kind haze --results ../test-results/ag-spacing6p9-sz1um-20261108/haze
python scripts/generate_report.py --kind sms-reference --results ../test-results/ag-spacing6p9-sz1um-20261108/sparkle
```

| New artifact | Purpose / producer |
| --- | --- |
| `case.json` | Predeclared physical assumptions, held-fixed settings and complete generator commands. |
| `inputs/{1um,0p5um}/heightfield.png`, `.source.json`, `sites.csv` | New calibrated inputs, physical statistics, shared geometry and source hashes from `prepare_etched_ag.py`; derived `height_um.npy` stays local. |
| `haze/protocol.json`, `haze/haze.json`, `haze/{1um,0p5um}/` | Existing haze CLI's actual native results, controls, angular powers/curves, timing and execution environments. |
| `haze/haze.html`, `haze/diagnostics.json`, `haze/validation.json`, `haze/*.png` | Existing report CLI's offline haze/shape/sampling audit. |
| `sparkle/protocol.json`, `sparkle/{1um,0p5um}/`, `sparkle/controls/`, `sparkle/summary.json` | Frozen optical plan, native solver outputs, saved camera floats and the existing SMS numerical controls. |
| `sparkle/surface-comparison.json`, `sparkle/height-surface-comparison.png`, `sparkle/*.png`, `sparkle/sparkle.html` | Same-camera/same-mask comparison to the retained depth3 case, physical 100um height crops, camera and sparkle views from the existing report CLI. |
| `*.log`, `default-generation-check.json`, `regression-check.json`, `source-protection.json` | Actual generation/solve/test/report logs, default-generator identity, retained depth-only report regression and before/after preservation checks. |

### Four OLED layouts on the exploratory 6.9um / Sz1um AG

`../test-results/ag-spacing6p9-sz1um-layouts-20261108/` refreshes Diamond,
RGB stripe, RGB-Delta and Windmill on the exact two calibrated height maps
from `ag-spacing6p9-sz1um-20261108/inputs/`. Those inputs and all prior results
are read-only. Haze is independent of the OLED layout and remains the retained
41.199873% scalar-wave / 56.048276% geometric fine-grid estimate; its report is
linked rather than recomputed.

All layouts retain nominal logical-address pitch 70um, full-green field mean
1 after linear decoding, 2um emission texels, source threshold zero and no LDF.
The 500um AG gap, 530nm wavelength, camera and 840um ROI are unchanged. The
four-layout common mask still removes exactly 89 non-DC bins; no low-pass,
MWA, guard-band adjustment or fitted cutoff is used. The aperture patterns are
the existing illustrative proxies, not measured commercial panels. Equal mean
brightness does not equalize aperture area or within-aperture radiance; the
comparison includes those input differences as well as the arrangement.

The frozen `protocol.json` contains eight commands derived from the existing
SMS command builder. The existing runner executes four native backends for
each 1um layout and CUDA Torch for each 0.5um layout, twenty native jobs in
total. Only emission input, its mean-normalizing gain and output differ among
layouts at a given optical grid. `surface_source` records the parent height
case and haze result hashes; the optional surface comparison verifies the
same-layout original 25um / Sa0.1um AG baseline without modifying it.

| Layout | 1um Torch sparkle [%] | 0.5um Torch sparkle [%] | Fine minus coarse [percentage points] |
| --- | ---: | ---: | ---: |
| Diamond | 13.964710 | 13.789377 | -0.175333 |
| RGB stripe | 5.449757 | 5.419533 | -0.030225 |
| RGB-Delta | 14.410891 | 14.184711 | -0.226180 |
| Windmill | 12.871896 | 12.933972 | +0.062075 |

Maximum coarse wave-backend optical relative L2 difference is 7.79e-6. The
repeated Diamond optical arrays match the previous 6.9um case exactly on both
grids. Stripe is close to the paper's approximate numerical scale under this
retained internal metric; this does not identify the paper's emitter geometry
or implement its unknown SMS1000 response. No layout or filter was adjusted
to obtain that value. Rebuild the refreshed report offline:

```bash
python scripts/generate_report.py --kind sms-reference --results ../test-results/ag-spacing6p9-sz1um-layouts-20261108
```

| New artifact | Purpose / producer |
| --- | --- |
| `protocol.json` | Frozen commands, input hashes, camera, shared mask and read-only surface/haze provenance. |
| `{1um,0p5um}/{diamond,rgb-stripe,rgb-delta,windmill}/` | Existing native solver run metadata, brightness arrays and backend stats, followed by sampled camera pixels, separated pixels, masks and stats from the existing SMS analyzer. |
| `controls/`, `summary.json` | Existing flat, MTF, sensor-phase, ROI, backend and optical-grid checks plus complete layout statistics. |
| `surface-comparison.json`, `height-surface-comparison.png`, `layout-*.png`, `camera-*.png`, `sparkle.html` | Existing report CLI's same-mask surface baseline, AG crops, emission layouts and refreshed camera/sparkle comparison. |
| `main.log`, `fine.log`, `report.log`, `validation.json` | Actual native execution and report output; preservation, mask/normalization checks, independently recomputed float metrics and repeated-Diamond identity check. |

### Current default heightfield and preserved inputs

The 6.9um/Sz1um case is now the default of `prepare_etched_ag.py`,
`run_oled_demo.py`, both SMS modes and `calculate_haze.py`. `DEFAULT_INPUTS`
in `prepare_etched_ag.py` is the shared coarse/fine asset selection. Explicit
`--sa-um` still requests Sa normalization; with no amplitude option the
generator uses post-rounding reference-grid Sz=1um. A historical surface can
be regenerated into a separate folder using `--spacing-um 25 --sa-um 0.1`.

| New asset directory | Purpose |
| --- | --- |
| `assets/oled-1mm-etched-ag-spacing6p9-sz1um/` | Bundled 1um default: height PNG, calibrated sidecar and site/depth CSV copied byte-for-byte from the tested exploratory case. |
| `assets/oled-1mm-etched-ag-spacing6p9-sz1um-pitch0p5/` | Independently sampled 0.5um counterpart, sharing geometry and reference calibration; same three asset types. |

The old 25um asset directories, exploratory inputs and all historical results
remain unchanged. The selected case is still an exploratory assignment of
Table 1's 6.9um statistic, not a measured AG reconstruction. Changing the
default height does not change emission, camera settings or the separation
filter. No OLED/native-backend solve was repeated for this switch; the default
haze CLI was smoke-tested with its NumPy estimate.

Default output paths also use the new case name, keeping them separate from
the former defaults:

- OLED: `outputs/oled-1mm-etched-ag-spacing6p9-sz1um-periodic-50um`, with
  `-pitch0p5` for the finer grid.
- SMS: `outputs/oled-1mm-etched-ag-spacing6p9-sz1um-sms-reference` or
  `outputs/oled-1mm-etched-ag-spacing6p9-sz1um-sms-layouts`.
- Haze: `outputs/oled-1mm-etched-ag-spacing6p9-sz1um-haze`.

The switch audit is in `../test-results/default-heightfield-spacing6p9-sz1um-20261108/`:
`asset-provenance.json` records each promoted file and source hash;
`regeneration.json` checks fresh default generation against the tested PNGs;
`regenerate-*.log` contains the generator metadata from those temporary runs;
`tests-default.log`, `tests.log` and `skills-validation.log` contain actual
entrypoint/regression/skill checks; `validation.json` records input/history
preservation and the selected default commands. No new runner was added.

### Spacing-distribution sensitivity at fixed density

`../test-results/ag-spacing-distribution-sz1um-20261108/` compares minimum
site-distance ratios `0`, `0.3` (current default), `0.5` and `0.7`. All cases
retain 21,004 sites in 1mm square, nominal density spacing 6.9um, reference
post-rounding Sz=1um, Gaussian sigma=1.2um, profile power 2 and uniform
relative-depth law [0.7,1.3]. They use the same full-green Diamond input,
camera, 840um ROI and 89-bin common periodic mask as the retained baseline.
There is no extra low-pass, MWA or LDF.

Seeds 20261107/20261108/20261109 are evaluated at 1um; the first seed also
has a 0.5um check. Its default case reuses verified saved optics, giving
14 new native CUDA Torch runs and 16 analyzed float arrays in total.

| Minimum distance / nominal spacing | Nearest-neighbor mean [um], 3-seed average | Nearest-neighbor CV, 3-seed average | 1um sparkle mean [percent] | 1um sample min–max [percent] | First-seed 0.5um [percent] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.457990 | 0.522007 | 15.186055 | 14.523279–15.917458 | 15.950665 |
| 0.3 (default) | 3.974195 | 0.351438 | 13.147987 | 12.430875–13.964710 | 13.789377 |
| 0.5 | 4.680790 | 0.209192 | 11.132372 | 10.247746–12.209046 | 10.945824 |
| 0.7 | 5.527268 | 0.109525 | 10.037931 | 9.850287–10.143245 | 10.079975 |

Every seed gives the same descending order. The 0.5/0.7 coarse-grid means
are 15.33%/23.65% below the current-distribution mean; unrestricted random
sites are 15.50% higher. The maximum first-seed optical-grid shift is
0.175333 percentage points. These are three sample realizations, not
confidence intervals or a universal optimum. Exclusion also changes actual
neighbor means and spatial correlations; Sa/Sq and individual random depth
draws are not fixed, so this is not an isolated spacing-variance experiment.
The metric retains the same uncalibrated camera-domain scope as the baseline.

`distribution-sparkle.png` shows actual neighbor CDFs and sparkle with seed
ranges; `height-distributions.png` compares the four physical height crops.
`distribution-sparkle.csv` and `summary.json` contain per-seed metrics,
distribution/roughness statistics and limitations. The root and per-case
`protocol.json` files record exact generation/solver arguments, hashes and
read-only baseline provenance. Primary cases retain the existing SMS report;
secondary coarse checks use `sms_reference.analyze_layout_runs`, saving its
camera floats, masks and stats without adding another report generator.
`validation.json` records the float/mask/site checks, 14 passing tests and
574 unchanged historical/input files. `default-regeneration.json` verifies
that both default height PNGs and site CSVs still regenerate byte-for-byte.
The two experiments now have reusable commands in the existing SMS runner;
see the [two comparison commands](../README.md#ag-表面对比命令). From this
directory, the equivalent commands are:

```bash
python scripts/run_sms_reference.py --surface-study width --backend torch --stage all --out ../test-results/ag-width-sweep-new
python scripts/run_sms_reference.py --surface-study spacing-distribution --backend torch --stage all --out ../test-results/ag-spacing-distribution-new
```

These compute all cases afresh (12/16 native Torch solves respectively).
`--widths-um`, `--min-separation-ratios` and `--study-seeds` override the study
matrix. `--stage prepare/main/fine` can split a new batch. Prepared protocols
are immutable; resume with the same study and output root. Rebuild the new
root's plots and `sparkle.html` with `generate_report.py --kind sms-reference
--results ROOT`, without optics. Root outputs also include `surface-study.png`,
`surface-inputs.png`, `comparison.csv`, `summary.json` and `validation.json`;
each case retains its generated inputs, frozen commands, solver and camera
float arrays/stats. The existing runner and report implementation own these
operations; no separate experiment or plotting entrypoint is needed.

### Default etched-AG brightness outputs

Run `python scripts/run_oled_demo.py --backend all`.
Use `--pitch-um .5 --backend torch` for the same geometry at finer sampling.
The old `test-results/run-etched-ag.ps1` targets fixed historical result
directories; use the maintained CLI above for the new default.
`test-results/oled-etched-ag/report.html` retains its original 25um results.

- `report.html`: offline HTML with two inputs, four brightness outputs and raw downloads.
- `run.json`: full CLI/default parameters, input hashes, environment and wall time.
- `comparison.csv/json`: direct brightness CV, brightness mean/std and solver time.
- Per backend: `brightness.npy/png` (1um optical grid), `analysis_brightness.npy/png`
  (50um collected pixels used directly for CV), `roi_mask.npy`, `stats.json`.

Solver time includes any library loading/JIT/device setup occurring inside the
solver, and excludes postprocessing/report writing. The total run time is recorded
up to HTML generation. Missing/failed values are not replaced with zero.

CUDA Mitsuba keeps intersections, Snell refraction, Fresnel transmission,
piecewise-constant emission lookup and brightness accumulation on the GPU.
A Dr.Jit symbolic loop integrates every pupil node within each spatial chunk;
`--ray-chunk` still bounds the number of live ray queries. Only the completed
brightness image is downloaded. Ray-origin calculations and accumulation use FP64
precision, with FP32 Mitsuba rays/optical arithmetic. Mesh stride, pupil nodes,
physical parameters and detector integration are unchanged. Non-CUDA Mitsuba
variants retain the original NumPy integrator. The execution context records
`integrator: drjit_resident` or `numpy`.

```bash
python scripts/generate_report.py --results outputs/oled-1mm-etched-ag-spacing6p9-sz1um-periodic-50um
python scripts/generate_report.py --kind sms-reference --results outputs/oled-1mm-etched-ag-spacing6p9-sz1um-sms-reference
python -m unittest discover -s tests -v
```

SMS report generation reads existing optical floats and protocol; it neither
prepares inputs nor changes the fixed protocol, skill metadata or archive.
Optional `--out` copies its lightweight overview to another HTML entry and fixes
relative links. The complete report stays at `<results>/sparkle-report.html`.
Both reports show inputs/scales first, measurement conditions second, then all
four solver results, validation and the method appendix. SMS processing/plots
require NumPy, SciPy, Pillow and Matplotlib; CUDA is unnecessary for report-only use.

The input previews outline the measurement ROI. The 1mm field is ±500um; the
predeclared central 560um crop holds 21 complete 26.25um pixels, so Figure 2 spans
±275.625um (551.25um wide). All 441 pixels enter the sparkle statistic.
Each prepared SMS protocol records a new execution batch, declaration time and
solver SHA-256. Report generation requires every run to use that solver and to
start after this declaration, and every backend result must match its run ID.
The four backends run sequentially by default (`--backend-workers 1`). Their
sparkle values, images and solver wall times all come from that batch, with one
timing column. WSL Chromatix timing includes worker launch and result transfer.

Set `SPARKLE_TEST_MITSUBA=1` to enable five CUDA checks against the retained
NumPy integrator, including flat Fresnel transmission, rough surfaces, total
internal reflection, missed/zero emission and chunk/mesh-stride behavior.
From the repository root, replay all four saved full-pupil reference cases with:

```bash
python test-results/benchmark-mitsuba-resident.py --with-legacy-baseline
```

The script predeclares accuracy limits, preserves original input hashes and all
physical/sampling parameters, writes new reports under
`test-results/mitsuba-resident-regression/`, and times one complete legacy solve
for an isolated performance comparison. Omitting `--with-legacy-baseline` skips
that several-minute timing step. Historical result directories stay untouched.

Set `SPARKLE_TEST_OPTIONAL=1` and `SPARKLE_TEST_MITSUBA_VARIANT=cuda_ad_rgb` to
include all-backend integration tests. The historic reference-normalized outputs
are superseded; see `VALIDATION.json` for the current direct-image result.
