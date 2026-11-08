---
name: ag-haze
description: Estimate transmission haze from calibrated periodic AG heightfields, inspect wavelength and sampling effects, and distinguish surface-only scalar estimates from standard luminous-haze measurements. Use for AG haze or angular-scatter calculations; use the OLED workflow for camera-domain sparkle.
---

# AG transmission haze

This skill uses the maintained calculator in the `oled-sparkle` repository.
Locate that checkout from the current workspace or this skill's resolved parent;
on the configured machine it is `C:/Projects/oled-sparkle`. The implementation is
`oled-ag-sparkle/scripts/calculate_haze.py`; use the repository's Python environment.
Do not create a second calculator inside the installed skill.

## Calculate and interpret

Read the PNG's `.source.json` calibration. Height pitch and the PNG black-to-white
range are separate inputs; sampled peak-to-valley Sz is not the encoding range.
The calculator reuses the existing height reader and checks the source hash.
For a custom map without a sidecar, specify both dimensions explicitly. This
estimator requires periodic continuation; `--assume-periodic` is an explicit
physical assumption, not a remedy for arbitrary measured-patch edge jumps.

From the repository root:

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\calculate_haze.py --out test-results\ag-haze
```

The defaults are the current etched AG, normal-incidence illumination,
530 nm, n_in=1.5, n_out=1 and a **2.5 degree half-angle** about the direct beam.
For the existing independently sampled convergence pair and spectral sensitivity:

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\calculate_haze.py --height oled-ag-sparkle\assets\oled-1mm-etched-ag\heightfield.png oled-ag-sparkle\assets\oled-1mm-etched-ag-pitch0p5\heightfield.png --wavelength-nm 450 530 650 --out test-results\ag-haze
```

For actual native-backend comparison, add `--backend all` and use a separate
output directory. Torch/TorchOptics require the selected `--device` (CUDA by
default); Windows Chromatix uses the existing CUDA-only WSL adapter. Mitsuba
defaults to `cuda_ad_rgb`. Keep the NumPy estimate as the common reference:

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\calculate_haze.py --backend all --height oled-ag-sparkle\assets\oled-1mm-etched-ag\heightfield.png oled-ag-sparkle\assets\oled-1mm-etched-ag-pitch0p5\heightfield.png --out test-results\ag-haze-four-backend-new
.venv\Scripts\python.exe oled-ag-sparkle\scripts\generate_report.py --kind haze --results test-results\ag-haze-four-backend
```

The first command declares `protocol.json` before execution and refuses an
existing protocol. The second audits/plots retained data without launching a
solver. Report execution devices, native operations and solver timings. Wave
angular powers are saved as float NPY arrays; each model has a cumulative angular
CSV and per-run stats. Preserve the original estimate and previous comparisons.

Report `haze.json` and the angular cumulative-power CSVs. State wavelength,
indices, acceptance angle, surface hash, grid, angular coverage and model scope.
Check flat-plane zero haze, Parseval error, spectral-edge power and independently
sampled grid convergence. Interpolating a coarse height map does not recover
missing fine roughness. Never change the height scale to approach a target haze
unless the user explicitly requests a new, separately identified surface design.

## Meaning of the result

Use uniform incident light on the AG surface, not an OLED image or the saved
camera brightness. A camera pupil clips angles and cannot give total transmitted
power for a haze denominator. The scalar phase screen is
`exp(i*2*pi*(n_in-n_out)*(h-mean(h))/lambda)`; integrate its Fourier-bin normal
flux outside the acceptance cone, divided by all represented propagating normal
flux. Include the DC order and the narrow-angle diffuse power in the denominator.
The Cartesian bin weight contains `cos(theta)`; do not multiply it by another
solid-angle Jacobian. Use the two-dimensional cone, not a one-dimensional slit.

The three wave backends share the original thin phase screen; actual homogeneous
ASM output propagation does not change propagating angular powers. Their
agreement with NumPy is a numerical consistency check. Mitsuba traces uniform
forward illumination through a periodic closure of the original height mesh,
using geometric normals and native Fresnel/refraction. Two triangle-centroid
rays per original cell exactly weight projected facet area for the single-hit
geometric model. Each ray already carries power; do not apply the wave-bin
cos(theta) factor a second time to ray weights. Mitsuba omits diffraction;
wave/ray disagreement is model dependence, not proof of either model's accuracy.
Show geometric grid sensitivity as well as wave-grid sensitivity.

Label the wave result **monochromatic, surface-only scalar haze estimate**. The model
omits bulk scatter, unresolved roughness, polarization, angle-dependent Fresnel
coefficients and multiple interfaces/reflections. Its finite angular coverage is
a numerical limitation even when the spectral boundary contains little power.
Mitsuba includes single-interface unpolarized Fresnel transmission and records
ray coverage and energy balance. Neither model is an ASTM/ISO test certificate.
The phase-screen normalized ratio is not an absolute transmittance.

For standard luminous haze, obtain the illuminant spectrum, V(lambda), refractive
dispersion, transmitted-power weighting and instrument geometry/correction.
Several monochromatic haze values must not simply be averaged into a white-light
value. Haze and sparkle describe different measurements; equal Sa or Sq does not
imply equal haze because lateral correlations determine the angular scatter.

Definition and approximation boundaries:
[ASTM D1003-21](https://store.astm.org/d1003-21.html) and
[Simonsen et al. (2009), sections III and IV.1](https://arxiv.org/html/0904.3753).
The latter supports angular-power normalization and the acceptance convention;
its one-dimensional analytic approximation is not the calculator's 2D solver.
