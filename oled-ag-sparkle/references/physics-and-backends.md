# Brightness models and requested statistic

Torch, TorchOptics and Chromatix use scalar angular-spectrum propagation,
a thin height-dependent phase screen and ideal NA imaging. Incoherent source
nodes sample the supplied emission map. Mitsuba intersects a height mesh,
refracts fixed pupil directions and samples the emission texture with Fresnel
transmission. It omits phase and diffraction.

Each backend returns a floating-point collected brightness image. Integrate it
over the requested square measurement-pixel footprint using overlap-area weights,
then divide by the pixel area to obtain average brightness. For the current
1mm square field, the height/optical grid is 1000x1000 at 1um; each 50um pixel
averages 50x50 samples, giving 20x20 (400) measurement pixels with no ROI crop.
The emission grid is 500x500 at 2um. Crop [26,58,96,128] from the original 128x128
patch to obtain a complete 70x70, 140um repeat cell, repeat 8x8, then crop to
500x500. Increasing the field preserves texel pitch, emitter dimensions and
spacing. The previous texture-to-field resize enlarged emitters by 3.90625;
the following whole-patch tiling introduced phase breaks. Both are superseded.
Verify repeat period, uncut bright bodies and low boundary-neighbor jumps before
repeating an input patch. No edge blending is used. Physical scale remains an
illustrative assumption. The default height is a synthetic etched-AG proxy with
21,004 irregular Voronoi bowls, 6.9um nominal site spacing, variable depths and
1.2um rim rounding, calibrated at post-rounding reference-grid Sz=1um.
Its 16-bit PNG encoding range is 1.04um, distinct from sampled Sz. The input sidecar
records calibration and geometry. Compare the same geometry and scale on a 0.5um
grid to assess sampling sensitivity. Circular pits remain an explicit legacy case.
The former 25um/Sa0.1um surface and all numerical studies below retain their
original inputs; their reported values are not silently relabelled as the new
default. The 6.9um choice remains an exploratory interpretation, not a measured
mean period established by Table 1.

The statistic is directly `100 * std(I_ROI, ddof=0) / mean(I_ROI)` on the saved
`analysis_brightness.npy` values. All full measurement pixels, including dark
pixels, enter the statistic. The largest full pixel grid is centered inside the
requested geometric ROI; record its exact boundaries. No reference division,
signal-based masking or additional frequency filtering is used. Mean must be positive.

Source quadrature, FFT padding, height-grid density, mesh resolution and pupil
quadrature affect the simulated images. Record them with source provenance,
height mapping, wavelength, indices, gap, camera NA and actual computation times.
PNG previews are only display representations; metrics use the saved float array.

The bundled example uses the synthetic etched-AG height map and a crop of Samsung
Display's Diamond Pixel image. Physical scale and effective monochromatic luma
interpretation are explicit example assumptions in the source metadata.

## Separate surface transmission haze

`scripts/calculate_haze.py` is called by the sibling `ag-haze` skill. It reads the
same calibrated PNG using `sparkle_multibackend.read_height`, and reuses
`prepare_etched_ag.roughness`. No OLED source, camera pupil or detector sampling
is involved. Its target is angular transmission under a uniform, normally
incident plane wave at one dielectric rough interface.

For the existing thin phase-screen approximation, use

`t(x,y) = exp(i*2*pi*(n_in-n_out)*(h(x,y)-mean(h))/lambda)`.

Let A be the normalized Cartesian 2D Fourier coefficients of t. The outgoing
direction is `sin(theta) = lambda*sqrt(fx^2+fy^2)/n_out`; only sin(theta)<1
propagates. Scalar Helmholtz normal flux is proportional to
`Im(conj(U)*dU/dz)`. Fourier orthogonality therefore gives each bin the weight
`|A|^2*kz`, or equivalently `|A|^2*cos(theta)` after cancelling common constants.
The same df_x*df_y area belongs to each bin, so no extra solid-angle Jacobian
should be multiplied into this discrete sum.

For a half-angle alpha=2.5 degrees, the reported estimate is

`H_lambda = 100 * sum_{theta>alpha} |A|^2*cos(theta) / sum_{propagating} |A|^2*cos(theta)`.

Include DC and small-angle diffuse light in the denominator. Removing only DC
would count narrow-angle scatter as haze. The surface is periodically continued,
consistent with the bundled generator's periodic geometry; there is no finite
aperture diffraction, zero padding, window function or spatial smoothing.
Applying this assumption to an arbitrary nonperiodic measured patch can create
spurious seam scatter. Mean height removal changes only global phase.

Only represented directions enter the calculation. At 530 nm and n_out=1, the
complete-azimuth frequency coverage reaches about 15.37 degrees on a 1um grid,
and 32.01 degrees at 0.5um. Spectral edge power and independently sampled grid
convergence diagnose numerical resolution, but cannot reveal fine structure
absent from the height measurement. The periodic 1mm region also sets a finite
angular-frequency resolution. No angle-dependent Fresnel factor, polarization,
volume scatter, multiple interfaces/reflections or absorption is included.
The normalized flux ratio is not an absolute transmittance.

Independent wavelengths are not a luminous-haze calculation. With physical
transmitted powers P_outside and P_total, luminous haze would require
`100*integral(S(lambda)*V(lambda)*P_outside(lambda) d lambda) /
integral(S(lambda)*V(lambda)*P_total(lambda) d lambda)`.
Illuminant spectrum, dispersion, transmission weights and instrument correction
must be specified before making a standard white-light comparison.

Definition and limits:

- [ASTM D1003-21](https://store.astm.org/d1003-21.html) describes standard haze
  and luminous-transmittance measurement; the present monochromatic estimator
  does not implement its complete integrating-sphere/instrument procedure.
- [Simonsen et al., Phys. Rev. A 79, 063813 (2009), III and IV.1](https://arxiv.org/html/0904.3753)
  defines haze by angular scattered power normalized by transmitted power and
  discusses the 2.5-degree acceptance convention, roughness correlations and
  approximation limits. Its 1D phase-perturbation formula is not used as our 2D
  solver or as evidence that the present synthetic AG matches a measured glass.

### Native-backend haze case

`calculate_haze.py --backend all` executes all four libraries under uniform
normal incidence. The original `sparkle_multibackend.py` imaging solver is not
modified; its calibrated reader and mesh constructor are reused. New observable
functions live in the existing haze calculator, called by its established CLI.
The Windows adapter `chromatix_wsl.py` shares one request/array transport for the
old brightness operation and the new haze operation, and still requires GPU JAX.

Torch applies phase modulation, homogeneous ASM propagation and an angular FFT
in its native tensor implementation. TorchOptics uses `Field.modulate` and
`Field.propagate_to_z(..., propagation_method='ASM', asm_pad=(0,0))`, followed
by a native FFT. Chromatix uses `plane_wave`, `phase_change` and `asm_propagate`
with zero padding width, followed by JAX FFT. Double precision is declared.
The phase-screen boundary approximation is unchanged across the three.
For propagating modes, the output propagation transfer has unit modulus, so
changing output propagation distance cannot change its angular power. Native
agreement with the original unpropagated NumPy reference is therefore expected;
it verifies implementations and normalization, not Maxwell boundary accuracy.

Mitsuba uses the existing measured-height mesh construction with one extra
periodic endpoint row/column, giving N*dx projected coverage. It traces vertical
rays from the incident dielectric side through the centroid of each triangle,
two triangles per original height cell. The geometric normal is constant on
each face, so centroid rays weighted by equal projected triangle areas exactly
integrate this particular single-hit faceted model. The native `fresnel` and
`refract` operators supply transmission weights and outgoing directions. Ray
weights already carry incident projected-area power; do not multiply by another
cos(theta). Record hit coverage, reflected/transmitted fractions and balance.
This is a different, non-diffractive approximation, and its facet-normal grid
sensitivity can exceed the phase-screen angular-power grid sensitivity.

The native data directory declares a protocol before solving, preserves actual
wave Fourier powers, per-model angular curves/stats, devices, versions and
timings. `generate_report.py --kind haze` reads and audits those results only.
Analytical tests cover wave Bessel diffraction orders, propagation invariance,
and the geometric projected-facet Snell/Fresnel integral. Native API references:

- [TorchOptics fields and propagation](https://torchoptics.readthedocs.io/en/latest/user-guide/fields.html)
- [Chromatix functional API](https://chromatix.readthedocs.io/en/latest/api/functional/)
- [Mitsuba Fresnel/refraction API](https://mitsuba.readthedocs.io/en/v3.7.0/src/api_reference.html)

### Discrepancy investigation, 2026-11-08

Evidence is retained in `test-results/ag-haze-audit-20261108/` (from the
repository root). The report CLI reads the original saved native powers and
curves; the original inputs, native protocol/stats and earlier reports are
preserved. No OLED imaging solver, height encoding or material geometry was
changed. New numerical diagnostics live in the existing `calculate_haze.py`;
their maintained caller is the existing haze report. One-off experiments used
temporary commands calling these functions, with no persistent wrapper scripts.

**Measurement and normalization.** At 530nm on the 1um grid, the transmitted
scalar forward-flux partition is:

| Direction set | Percent of represented transmitted forward flux |
| --- | ---: |
| DC order | 60.613031 |
| Nonzero orders inside the 2.5-degree cone | 36.656019 |
| Orders outside the cone | 2.730950 |

All nonzero orders sum to 39.386969%, but calling that haze would include
narrow-angle scattering excluded by the stated measurement. A periodic-cell
DC order should also not be identified with the exact ensemble-coherent
component of a random material. Using non-DC power alone as the denominator
would produce yet another statistic (about 6.93%). None replaces total
transmitted power in a haze denominator. Removing the cosine weight changes
2.730950% to 2.735265%, only +0.004315 percentage points. A constant flat
Fresnel factor cancels between numerator and denominator; it cannot supply
the missing tens of percentage points. An angle-dependent Fresnel factor,
additional interfaces and the full instrument remain different questions.

For the defined smooth synthetic surface, the 1-to-0.5um wave-grid change is
only +0.0000471 percentage points. This diagnoses the present function's
sampling, not unmeasured real-glass high-frequency roughness. The 1mm domain
gives about 0.03037 degrees near-axis angular spacing; reducing pitch keeps
this spacing and expands coverage. Low edge power cannot establish that an
actual glass lacks structure beyond the input bandwidth.

**Height, correlation and slopes.** Diagnostics use the calibrated decoded
height without detrending other than its mean. They find approximately:

| Descriptor | 1um grid | 0.5um grid |
| --- | ---: | ---: |
| Radial circular ACF first 1/e crossing [um] | 7.3307 | 7.3222 |
| Height skewness | -0.74151 | -0.74151 |
| Height kurtosis (not excess) | 2.57485 | 2.57484 |
| Spectral RMS slope | 0.0332239 | 0.0332242 |
| Height variance above the haze spatial-frequency cutoff [%] | 4.02564 | 4.02572 |
| Phase RMS at 530nm [rad] | 0.70248 | 0.70249 |

ACF is circular covariance/Sq^2, azimuthally averaged using nearest-radius
shells at their mean actual radii. Its first 1/e crossing is interpolated;
it is not a fitted Gaussian correlation parameter. It differs from the nominal
25um site spacing, the equivalent Voronoi diameter, and line-profile Rsm.
The height PDF is not Gaussian, so Gaussian roughness formulas cannot silently
replace the measured statistics. Likewise, line Ra with unspecified sampling
and filtering cannot be converted directly into areal Sa.

At 530nm in air the 2.5-degree cutoff is 0.0823007 cycles/um, corresponding
to a 12.1506um spatial period. Much of the current height variance is at longer
lateral scales. The optical field is exp(i*phase), so its angular power is not
the height PSD: phase RMS around 0.70rad is not an asymptotically small
perturbation. The 4.03% height-variance tail is therefore a descriptor, not a
first-order haze prediction.

For a vertical incident ray and local slope magnitude s, let
gamma=atan(s), eta=n_in/n_out. Before total internal reflection, the outgoing
deviation magnitude is `abs(asin(eta*sin(gamma))-gamma)`. For eta>1 and
acceptance alpha, its threshold is

`s_alpha = sin(alpha)/(eta-cos(alpha)) = 0.0870730` for eta=1.5, alpha=2.5deg.

Only 0.5540%/0.6132625% of the original triangle projected area exceeds this
slope on the two grids. The 0.5um p90/p99 slopes are 0.05241/0.08194, both
below threshold. Fresnel-weighting this tail gives 0.553996%/0.613258%,
closely reproducing the native geometric results. The approximately 0.000075
percentage-point fine-grid difference is small compared with its grid shift;
the analytic audit uses float64 whereas the native mesh/directions use float32.
This agreement audits the same geometry, not independent interface physics.

Independent evaluations of `prepare_etched_ag.raw_surface` at finer pitch,
using the original sites/depths, 1.2um Gaussian smoothing, fixed reference
calibration and uint16 quantization, give:

| Pitch [um] | Analytic single-hit geometric haze [%] |
| --- | ---: |
| 1 | 0.553996 |
| 0.5 | 0.613258 |
| 0.25 | 0.627130 |
| 0.125 | 0.631225 |

These were in-memory independent samples, not coarse-map interpolation, and
were not new Mitsuba runs. The last change is +0.004095 percentage points
(about 0.65% relative). This supports a value near 0.63% for this specific
single-hit geometry, while retaining the fact that the original two native
grids alone were not fully converged. It does not make geometry and diffraction
the lower/upper bounds of real haze.

**Spectral weighting and dispersion.** The existing phase-screen function was
evaluated on the 0.5um surface from 380 to 780nm in 5nm steps. Official
[CIE D65](https://cie.co.at/datatable/cie-standard-illuminant-d65) and
[CIE photopic V(lambda)](https://cie.co.at/datatable/cie-spectral-luminous-efficiency-photopic-vision)
tables were checked against their published MD5 values and retained with SHA256.
An illustrative transmission weight is explicitly defined as

`Q(lambda) = [4*n_in*n_out/(n_in+n_out)^2] * sum_propagating |A|^2*cos(theta)`.

The integral `integral(D65*V*Q*H)/integral(D65*V*Q)` then gives:

| Index assumption | 5nm quadrature [%] | 10nm quadrature [%] |
| --- | ---: | ---: |
| n=1.5, constant | 2.867985 | 2.867757 |
| N-BK7 n(530nm)=1.519584, frozen | 3.146550 | 3.146301 |
| N-BK7 Sellmeier dispersion | 3.128794 | 3.128550 |

The [SCHOTT N-BK7 datasheet](https://media.schott.com/api/public/content/41e799d0bf874807a0bb8e702fbb75b5?v=54856406)
supplies an illustrative real dispersion, not the identity of the AG glass.
Its formula and coefficients are retained in `spectral-sensitivity.json`.
Changing from n=1.5 to N-BK7 changes both the base index and dispersion;
the frozen-N-BK7 comparison isolates dispersion, about -0.01776 percentage
points here. The fixed-n monochromatic range is 1.9904–3.9021%; the dispersive
example is 2.4401–4.0842%. The increase toward red for this surface is compatible
with the wavelength-dependent angular cutoff sampling more of the narrow-angle
lobe; one cannot infer haze simply from a lambda^-2 height-variance rule.

This is a truncated-band, approximate surface-model sensitivity calculation.
Q is not a measured absolute transmittance, and flat-interface Fresnel does not
provide vector angle-dependent amplitudes. Missing source/instrument settings,
actual substrate dispersion, other interfaces and volume scattering preclude
a standard luminous-haze claim. For any specified nonnegative spectral and
transmitted-power weights, the integral is a weighted mean of H(lambda);
these few-percent model values do not become 20–40% by averaging them.

**Independent small-scale electromagnetic check.** The maintained reference
in `tests/test_haze.py::rayleigh_sinusoid`, called by `InterfaceBenchmarkTests`,
solves shallow sinusoidal interfaces by oversampled Rayleigh boundary matching.
For z=h(x), the incident field is exp(i*k1*z), reflected orders are
`r_m*exp(i*q_m*x-i*kz1_m*z)` and transmitted orders are
`t_m*exp(i*q_m*x+i*kz2_m*z)`. TE matches E_y and its normal derivative; TM
matches H_y and epsilon^-1 times that derivative. The common normal-length
factor cancels. Both reflected/transmitted evanescent orders participate.
The time convention and radiation branches are consistent with the boundary
formulation in [Depine & Lakhtakia, sections II–III](https://arxiv.org/html/physics/0408050).
This is direct boundary collocation, not their Fourier-projection implementation.

At 530nm, n=1.5->1, normal incidence, 2.5deg acceptance and amplitude 0.12um
(Sq=0.08485um), the results are:

| Sinusoid period [um] | Thin phase screen [%] | Equal-incident-power TE/TM interface [%] | Geometric [%] |
| --- | ---: | ---: | ---: |
| 8 | 22.964315 | 23.007134 | 24.999909 |
| 16 | 0.744015 | 0.751910 | 0 |

Flat controls recover T=0.96, R=0.04 and zero haze for both polarizations.
Orders +/-8,16,32,48 and boundary oversampling 4x/6x were inspected. At the
finest solve the maximum energy error is 6.7e-16 and the independent shifted,
double-density boundary relative residual is 1.1e-15. The +/-32-to-48 haze
change is below 1e-8 percentage points. TE/TM haze is combined by transmitted
power for equal incident powers, rather than averaging their ratios blindly.

For these shallow 1D examples, thin-screen errors are -0.04282 and -0.00789
percentage points. The 16um case has no local refracted rays outside the cone
but has nonzero electromagnetic haze. Thus a small ray slope tail does not
exclude outside-cone diffraction. This supports the mechanism behind the
wave/ray difference, and provides a check beyond homogeneous ASM propagation.
It does **not** quantify the thin-screen error on the non-Gaussian 2D AG.
Rayleigh convergence is not assumed for arbitrary steep or general 2D relief;
such a claim would need a suitable converged vector solver on representative
patches with controlled boundaries and size.

**What remains unidentified.**
[Cheng et al., Table 3](https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf)
reports different haze values even for samples sharing Ra=0.35um or 0.25um.
It does not supply their 2D height maps, PSD/slope distributions or enough
haze-instrument settings to reconstruct a calibrated sample. The 97um OLED
pixel pitch and 530nm LDF design wavelength do not specify the haze test source.
The current 0.10um Sa, synthetic bowls and smoothing were never matched to
these sample statistics. The available evidence rules out the audited cosine
factor, present smooth-surface grid errors and these illustrative spectral
weights as sufficient explanations of a tens-of-percentage-point discrepancy;
it does not uniquely apportion the unknown real-sample discrepancy.

Real commercial AG also spans a wide range not ordered by Ra alone. The
[AGC November 2022 acid-etched-glass factsheet, performance table](https://www.agc-yourglass.com/sites/default/files/2022-12/Factsheet_Anti-glare_1122_EN.pdf)
lists LST 100 at haze 2.5%, Ra=0.11um, Rsm=50um; VLST 40 at haze 15%,
Ra=0.15um, Rsm=25um; and NST 40 at haze 30%, Ra=0.05um, Rsm=15um.
These are cross-family specifications, not a controlled one-parameter experiment
or data for Cheng's samples. They demonstrate why selecting a single Ra target
does not identify an AG material. The vendor's Rsm is not our Voronoi spacing
or ACF length; the table does not determine a 2D PSD, slope PDF, measurement
bandwidth or a one-interface model. No unique causal contribution should be
inferred from these three product rows.

[Jafari et al., Characterization of Etched Glass Surfaces by Wave Scattering,
sections 2 and 5](https://arxiv.org/html/physics/0503032) combine AFM, larger-scale
Talysurf and optical scattering on etched glass, estimating both roughness and
correlation length. Their changing etch regimes and reflection measurements
are not a transmission-haze calibration for display AG. They support measuring
multiple spatial scales rather than relying on a single small AFM patch or a
single height-amplitude statistic.

The next calibration evidence should include multiscale measured surface maps
with lateral/vertical calibration, bandwidth and filtering, line-profile Ra
measurement conditions, 2D PSD/ACF and slope tails, plus spectral angular
transmission and total transmission with documented sample orientation. Multiple
patches are needed to quantify representativeness. Surface, back-face/coating
and bulk contributions should be separated experimentally (for example,
controlled index-matching or polished controls with their altered Fresnel and
transmission accounted for). Inclusions, volume index variations or particles
could scatter, but this paper supplies no basis to assign a bulk-scatter
coefficient to its AG glass. No ad hoc bulk term or height rescaling was fitted
to the target haze. A future parameterized AG must be a separately saved case
with measured bandwidth/correlation evidence, rather than a fitted Sa alone.

### Controlled attribution to scale, depth and morphology

The separate `test-results/ag-haze-texture-20261108/` study reuses the calibrated
loader, phase-screen calculation, surface diagnostics and analytic facet
refraction at 530nm, n=1.5->1, normal incidence and 2.5deg acceptance. These
are controlled counterfactuals, not a reconstruction of Cheng's glass. There
are no new native GPU runs, bulk-scatter fits or general Maxwell solves.

**Three different length scales.** Input pitch (1/0.5um) controls sampling;
nominal 25um spacing and the roughly 7.32um ACF length describe different
physical lateral statistics; 1mm is the periodic calculation domain. The
Voronoi texture is not a 25um grating. Applying the grating equation directly
to its nominal spacing is a scale analogy, not its diffraction-order solution.

For `h_new(x,y)=d*h_original(x/s,y/s)`, d changes height and phase strength,
and s changes all lateral dimensions. The pit count is unchanged; pitch,
domain, spacing and rounding length all scale by s without interpolation.
This differs from sampling a fixed surface more finely. The 6-by-4 matrix
and source hashes are in `texture-sensitivity.json/csv` and the predeclared
`texture-protocol.json`.

**Depth alone.** Keeping all lateral structure fixed gives:

| Height factor d | Sa [um] | Sq [um] | Wave haze [%] | Analytic geometry [%] |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 0.05 | 0.05926 | 0.547552 | 0 |
| 1 | 0.10 | 0.11851 | 2.730997 | 0.613258 |
| 2 | 0.20 | 0.23703 | 15.902151 | 17.921481 |
| 3 | 0.30 | 0.35554 | 38.226925 | 40.830189 |

Phase RMS becomes d*0.7025rad. This is not uniformly weak phase, so haze
does not simply scale as d^2. Slopes also increase by d, moving much more
area beyond the 0.08707 geometric threshold. These Sa controls cannot be
replaced by the paper's line Ra. Similar numerical haze does not identify
or calibrate a real sample.

**Lateral structure, with and without changing rounding.** At fixed height,
whole-map dilation to nominal spacings 6.25/12.5/18.75/25/37.5/50um gives
wave haze 34.3030/17.2560/6.6602/2.7310/0.5207/0.1118%. Rounding changes
simultaneously to 0.3/0.6/0.9/1.2/1.8/2.4um. This cannot be attributed only
to pit spacing.

To separate these factors, `spacing-sensitivity.json` holds physical Gaussian
sigma=1.2um and Sq=0.11851254um fixed, scales the original site layout and
domain, then independently samples the existing generator:

| Nominal spacing [um] | ACF 1/e [um] | Wave haze [%] | Analytic geometry [%] |
| --- | ---: | ---: | ---: |
| 6.25 | 2.6446 | 27.458297 | 38.253189 |
| 12.5 | 4.1012 | 13.983891 | 10.088634 |
| 18.75 | 5.6862 | 5.790009 | 2.265908 |
| 25 | 7.3222 | 2.730997 | 0.613258 |
| 37.5 | 10.6658 | 0.783926 | 0.057825 |
| 50 | 14.0557 | 0.277582 | 0.003400 |

Even at fixed phase RMS and physical rounding, shorter correlations can
produce tens-of-percent scalar haze. Sa and the entire height PDF are not
fixed by the Sq constraint. The 12.15um spatial-period scale of the acceptance
cutoff helps explain the trend, but the random texture has a broad spectrum
and nonlinear phase modulation creates higher orders.

The factors are coupled. Geometric slopes scale as d/s, making this single-hit
model depend on the slope distribution through that ratio. For example,
(s,d)=(0.5,0.5) and (1,1) both give geometric haze 0.613258%, but wave haze
4.605211% and 2.730997%. Waves also depend on independent absolute length
scales relative to wavelength. At (s,d)=(0.5,2), wave haze is 52.1498%, not a
product of separate haze-ratio gains. The sharpest/deepest matrix corners
include some total internal reflection and change the adequacy of single-hit
geometry and the thin-screen approximation.

**Bowl shape and smoothing at matched height statistics.** The existing
`prepare_etched_ag.raw_surface` now exposes power p in
`-depth*(1-(d_nearest/d_second_nearest)^2)^p`, followed by Gaussian rounding.
Power 1 retains a rim slope before smoothing; p>1 has zero rim derivative;
larger p concentrates the bowl toward its center. Default p=2 regenerates both
original PNGs byte-for-byte (`default-generation-check.json`). The existing
prepare CLI accepts `--profile-power`; there is no second surface generator.

With the same 1600 sites/depths, 25um nominal spacing, matched Sq and independent
0.25um sampling, the study finds:

| Change | Wave haze [%] | Analytic geometry [%] |
| --- | ---: | ---: |
| p=1, sigma=1.2um | 3.590580 | 0.466891 |
| p=2, sigma=1.2um | 2.731293 | 0.626914 |
| p=4, sigma=1.2um | 2.732318 | 1.392991 |
| p=2, sigma=0.3um | 3.859554 | 2.302687 |
| p=2, sigma=0.6um | 3.577979 | 1.864578 |
| p=2, sigma=2.4um | 1.164430 | 0 |
| p=2, sigma=4.8um | 0.135940 | 0 |

Equal Sq fixes phase RMS, not Sa, Sz, skewness, PSD or slope tails. In
particular, p=4 gives skewness about -1.346 instead of the original -0.742.
Additional Sa=0.10um controls give wave haze 3.576450% for p=1 and 3.110016%
for p=4, with explicitly different Sq values. This bowl-power family changes
wave haze by less than one percentage point at matched Sq, while the relative
geometric tail changes can be large. It does not establish a small shape
effect for arbitrary real surfaces or missing finer structure.

Increasing sigma removes fine structure, lengthens correlations and reduces
high slopes even after restoring Sq; model haze changes over a factor of 28
between sigma=0.3 and 4.8um. Independent 0.25/0.125um evaluations of sigma=0.3um
with fixed reference amplitude and encoding step give wave haze
3.85955344/3.85957197% and geometry 2.30268747/2.31173094%. The small wave
shift and 0.39% geometric relative shift are in `sampling-check.json`; this
does not establish the roughness bandwidth of real glass. Fifteen separately
calibrated inputs (nine morphology and six spacing controls) are saved locally
under `cases/`, with sidecars retaining normalization, source identity and hashes.

**Periodicity, height sign and higher-order controls.** Exact 2-by-2 tiling
expands the calculation domain from 1mm to 2mm without changing physical
texture; haze remains 2.730997%. This is expected Fourier consistency, not
independent larger-sample convergence or validation of a nonperiodic boundary.

Height inversion h->-h retains the same wave and geometric haze. Conjugating
exp(i*phase) reverses Fourier power in frequency, preserving a centered circular
cone integral; slope magnitudes also stay unchanged. Thus the label pits versus
bumps alone cannot explain this discrepancy in these models. This symmetry is
not established for general vector, oblique or multiple-interface scattering.

Three conjugate-symmetric random-phase surrogates keep the exact 2D height
Fourier magnitudes and Sq (PSD relative L2 error below 1e-12), but change height
PDF and higher spatial correlations. Wave haze becomes 2.8988–2.9191%, versus
2.7310%; kurtosis approaches 3 and skewness approaches 0. The approximately
0.17–0.19 percentage-point shift shows that equal height PSD need not mean equal
nonlinear optical phase spectrum. This is not an isolated height-PDF test and
does not bound all shapes with that PSD. See `phase-surrogates.json`.

These controls show that depth and lateral correlation mismatch are sufficient
to produce an order-of-magnitude model difference, while the tested bowl-power
changes alone are not. Physical smoothing is also a strong factor. They do not
identify which factor dominates the actual paper samples: calibrated maps,
PSD/slope statistics and metrology are missing. Material comparison must match
height and lateral statistics together; Ra or target haze alone is insufficient.

### Why the depth3 camera value still does not reproduce the paper's 5%

The depth3 calculation gives 32.565193% on the 0.5um sampled-camera pipeline,
compared with 23.977983% for the source surface. The measurement camera/mask
and optical commands match; this establishes an internal model increase, not
an absolute correspondence to the paper SMS1000 reading. The sampled-camera
pipeline was recommended for a defined processing order, but has not been
calibrated to that instrument. Its numerical value cannot directly replace the
paper value merely because both describe sparkle.

An offline audit reuses the unchanged floating optical images and the existing
`sms_reference.measure_mwa` function, with the historical fixed settings:
three 140um square MWA passes on the optical grid, a 560um central ROI, followed
by 26.25um measurement-pixel integration and population 100*std/mean. The
effective 21x21 pixel extent is 551.25um. No parameter was chosen to target 5%.

| Surface | 1um MWA [%] | 0.5um MWA [%] |
| --- | ---: | ---: |
| Original depth, current normalized-source optics | 5.287834 | 5.132102 |
| Depth x3, same normalized-source optics | 8.422367 | 7.989120 |

The earlier optical source run separately recovers 5.291803%. The source
normalization and positive-emitter selection were subsequently made explicit,
so the current and older original-depth values are close rather than identical.
Both original and depth3 results are also checked with TorchOptics/Chromatix
at 1um. All arrays, settings and hashes are retained in
`test-results/oled-depth3-metric-audit-20261108/`.

The two measurement pipelines differ in more than frequency-mask shape: the
current 32x32 camera captures an 840um ROI before masking; the legacy MWA
smooths before integrating pixels and uses a smaller ROI. Smoothing suppresses
much of the short-scale stochastic variance. The lower MWA value is therefore
not evidence that the larger camera value was an arithmetic error, nor that MWA
has reproduced the instrument. MWA's own coarse-to-fine change is about -0.156
percentage points for the original and -0.433 for depth3, so it should not be
described as fully converged on these two grids alone.

[Cheng et al., Table 3](https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf)
reports without-LDF sparkle values 4.8–6.0 and with-LDF values 1.5–2.2.
The paper identifies SMS1000, but does not provide enough camera, spectral
weighting and separation settings to reconstruct that measurement chain.
Our nominal logical pixel pitch is 70um, whereas the described module is about
97um. Its real etched texture and line Ra are also not our synthetic Voronoi
surface and areal Sa. Matching the monochromatic scalar haze around 38% by
height scaling does not constrain all the spatial intensity statistics that
determine measured sparkle.

These results leave both measurement-chain and material/geometry mismatch to
resolve. The depth3 experiment cannot be presented as a successful reproduction
of the paper. Further comparison should fix an independently justified camera
transfer and processing definition, match the emitter geometry and surface
statistics, and report both retained metrics without tuning a filter to 5%.

### SMS1000 source audit: confirmed differences and remaining unknowns

The separate `test-results/sms1000-metrology-20261108/` audit reads public
manufacturer material, the complete Cheng paper and retained floating results.
It uses the existing reuse-optics/report entrypoints. No optical solve, new
headline filter, fitted cutoff or replacement of the prior results is involved.

**Public definition.** The [2023 SMS1000 brochure](https://www.display-messtechnik.de/fileadmin/template/main/docs/SMS1000-05.2023-7.pdf),
PDF pages 5 and 11 (printed pages 3 and 9), describes numerical low-pass
processing of captured images, followed by grey-level standard deviation/mean.
A movable AG can instead use two exposures with a translated layer before
filtering. The brochure gives adjustable imaging rather than one universal
camera setup. Its specifications are device-family information, not settings
documented for Cheng's test. The 2021 brochure has a different distance range;
the locally verified 2023 document gives roughly 200–900mm, typically above 500mm.

[Becker 2021, section 3.4 and Figs. 8–9](https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf)
distinguishes removal of periodic peaks from restriction of the intervening
stochastic frequencies. Removing periodic peaks alone does not impose the
visual frequency response. Section 3.6 also shows aperture dependence and
does not establish F/8 as a universal visual optimum. Therefore the earlier
description of high-frequency suppression as merely optional was insufficient
when discussing an SMS1000-equivalent result. The unchanged common-notch
pipeline remains useful as a defined internal camera-domain statistic, but
must not be represented as a complete implementation of that meter.

The brochure does not disclose the exact software/filter settings of the
paper, and the paper does not supply them. A low-pass requirement does not
identify a unique kernel, cutoff, pass count, difference-image mode or software
version. The historical optical-grid MWA is likewise not proven to be the
paper's camera-image algorithm. No arbitrary correction factor is justified.

**Traceability of geometry.** Visual review of the complete
[Cheng PDF](https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf), especially
Figs. 5–6 and Table 3, establishes:

- The described module pitch is about 97um. The simulation assigns 70um nominal
  pitch to a Samsung-image-inspired Diamond proxy, with a 140um exact green
  intensity repeat. The paper provides no calibrated 2D emitter aperture map
  confirming the same arrangement, fill factor or within-aperture radiance.
- The stated 145um is the physical separation from the emitting pixels to
  the LDF, with other LDF spacings in Table 2. It is not the total distance to
  the outer AG surface. Neither the diagrams nor text specifies an AG gap
  that would authorize replacing the assumed 500um by 145um.
- Green pictures are used for testing. The 530nm value is introduced for LDF
  design; the measured green spectral distribution is not supplied.
- Focal length, F-number, working distance/ISR, focus implementation, sensor
  configuration, ROI, exposure and numerical filter parameters are not stated.
  The figures are module-stack diagrams, not measurement-camera diagrams.
- Table 1 labels its quantities as width/depth variance and does not give
  enough distribution/definition information to infer a mean pit period or
  a 2D PSD. Its 6.9um entry must not be substituted for an average etched
  feature spacing. Table 3's line Ra does not determine areal Sa or Sq.

The present 50mm focal length and 3.75um sensor pitch have manufacturer-spec
support, but are not established as the actual paper configuration. F/8 is
one allowed operating choice, 400mm is an explicit model assumption, and
26.25um is derived from those choices by thin-lens geometry:

`m=f/(u-f)=1/7; q_object=p_sensor/m=26.25um`.

Nominal ISR is then 70/26.25=2.667; the exact G repeat spans 5.333 detector
pixels. Substituting P=97um while leaving this assumed camera unchanged would
give a nominal ISR of 3.695, not a recovered value for the paper. The current
optical intensity cutoff is 29.480 cycles/mm versus an axial sensor Nyquist
of 19.048 cycles/mm. Real lens MTF and exact lens pupil geometry are unknown.
The vendor's full 1280x960 array would span 33.6x25.2mm under these assumptions,
whereas the model headline samples only an 840um square (32x32 pixels). This
is a finite-field/statistical limitation, not evidence of the paper ROI size.

The paper cites Paz et al.'s histogram-analysis paper, whose public abstract
describes a Gaussian fit, but that citation does not identify Cheng's camera,
firmware or filter settings. Borrowing another paper's lens or fitting method
would not establish a matched experiment. Table 3 labels its entries as
Sparkle Value without a percent sign or an export definition; treating the
values as approximately 5% follows the current comparison convention and
should be checked against the actual measurement report.

**Numerical diagnostics without a new headline filter.** Frequency boundaries
were declared from the existing 70um nominal pitch, 140um G repeat and 26.25um
detector pitch before evaluating them. No boundary was found by searching for
5%. Parseval variance sums use the spatial Fourier coefficients of the camera
signal, not angular optical power or the haze cosine-flux weights. For depth3,
0.5um optical sampling gives:

| Diagnostic | Observed result |
| --- | ---: |
| Unchanged common-notch contrast | 32.565193% |
| Variance above 1/70um (14.286 cycles/mm) | 15.368589% of retained variance |
| Component below that radial boundary, normalized by original mean | 29.958457% |
| Component below 1/140um (7.143 cycles/mm), same normalization | 22.892363% |
| Low component of rectangular pre-sampling Nyquist decomposition | 32.523517% |
| Contrast with Diamond-only periodic mask (56 bins vs common 89 bins) | 33.070773% |

Thus missing bandwidth restriction is a confirmed definition mismatch, but
simply excluding frequencies above the nominal pixel frequency cannot alone
explain the entire gap to 5%. These components are not claimed SMS1000 scores.
The rectangular Nyquist split is performed on a periodic DFT of the full 1mm
optical image and then passed through the unchanged linear pixel sampler and
mask. It has a finite-DFT boundary assumption and keeps the covariance term:
`var(total)=var(low)+var(high)+2*cov(low,high)`. The variance balance error is
below 1e-12. It is a diagnostic decomposition, not a simulation of the actual
meter's anti-alias optics. The corresponding high-component variance is about
0.926% of total and the covariance contribution about -0.671%; treating those
as independent powers would be incorrect.

The additional common-mask notches slightly reduce the result, so they do not
explain why the current value is high. Four sensor phases give about
32.504–32.565%; an independently sampled/reprocessed 420um ROI gives 29.647%,
and the center of the full-mask result gives 34.586%. These finite checks do
not produce the paper value. The main/fine-grid change is 0.151 percentage
points, native wave L2 differences are below 1e-5 and the finite flat control
is about 0.539%. These numerical checks do not validate the material model,
but provide no evidence for a factor-six arithmetic/backend error.

The verified source set includes only the public six-page IEC preview, whose
contents identify Annex B but do not include its details. Full IEC-conformity
or a particular standard mask is not asserted. The independent 2025 matched-MWA
paper concerns fractional sampled-image kernel matching, not the unknown
SMS1000 settings used by Cheng. Both sources are scoped explicitly in the
manifest rather than used to fill absent settings.

The evidence identifies a known measurement-response mismatch and a known
display-geometry mismatch, while the real AG statistics and camera settings
remain unidentified. Depth3 still increases both retained comparison metrics.
A unique percentage-point allocation of the remaining paper/model discrepancy
cannot be inferred from the public record. Independent instrument settings,
emitter/stack geometry and calibrated AG data are required before a paper
reproduction or an instrument-equivalent corrected number can be claimed.

### Exploratory assignment: 6.9um nominal AG spacing, post-rounding Sz=1um

The user-requested case `test-results/ag-spacing6p9-sz1um-20261108/` explicitly
assigns Table 1's 6.9um entry to **nominal site spacing**, without interpreting
the paper's width variance as a measured mean period. The 97um number remains
the paper's emitting-display pixel pitch. This controlled surface comparison
keeps the existing 70um nominal display proxy and camera/mask unchanged.

The maintained `prepare_etched_ag.py --sz-um 1 --spacing-um 6.9` normalizes the
post-Gaussian reference-grid peak-to-valley to 1um. It retains the 1mm periodic
field, seed 20261107, bowl power 2, relative depth distribution and physical
Gaussian sigma 1.2um. The 21,004 sites are independently sampled on 1/0.5um
grids with a shared 0.5um reference calibration. Fine-grid decoded Sa, Sq, Sz
are 0.128404, 0.159244, 1.000009um; the PNG encoding range is **1.04um**.
This Sz choice does not assert a 1um nominal single-pit depth, Sa, Sq or Ra.
Its radial height-ACF first 1/e crossing is about 2.815um, compared with about
7.32um for the retained original/depth3 lateral pattern. The 1.2um smoothing
length is held physically fixed, so its ratio to site spacing also changes.

Under the unchanged 530nm, n=1.5->1, uniform normal-incidence, 2.5-degree cone
measurement, all three native wave backends give haze 41.206719% / 41.199873%
on the coarse/fine grids. Native Mitsuba gives 54.691702% / 56.048276% with its
single-interface Fresnel model. Wave grid shift is -0.006846 percentage points;
ray shift is +1.356575 points (2.48% relative), which is not treated as zero
or as proof of full geometric convergence. The angular integration, flux
denominator, flat controls and finite angular-coverage diagnostics are retained.
Shared-wave agreement checks implementation consistency, not interface Maxwell
accuracy. Wave/ray estimates do not define physical lower and upper bounds.

The same 77,568 full-green sources, 500um AG gap, ideal f=50mm F/8 camera at
400mm, 26.25um square object pixels and 840um ROI are reused. The exact original
four-layout union mask removes 89 non-DC bins; no low-pass, MWA, apodization or
new notch width is introduced. Native coarse Torch/TorchOptics/Chromatix give
separated contrast 13.964710% / 13.964793% / 13.964572%; fine Torch gives
13.789377%. Grid shift is -0.175333 percentage points (about -1.26% relative).
The finite flat residual is 0.539403% at the fine grid and is not subtracted.
Native geometric raw CV is excluded from this diffraction-camera metric.

Compared with the saved depth3 fine result 32.565193%, the new sparkle falls
18.775815 percentage points, or 57.66%. Height Sq also decreases from about
0.35554um to 0.15924um and lateral correlation changes, so this experiment
does not isolate spacing from amplitude or morphology. Its higher haze and
lower camera contrast show that the two observables are not interchangeable
within the retained model. It remains above the paper's approximately 5 value
under an unmatched measurement response and physical display/AG configuration;
it is not an SMS1000 correction or a calibrated paper reproduction.

### Four-layout refresh on the same exploratory surface

`test-results/ag-spacing6p9-sz1um-layouts-20261108/` executes the existing
Diamond, RGB-stripe, RGB-Delta and Windmill proxies on the same calibrated
6.9um/Sz1um AG. It retains 70um nominal address pitch, mean linear G=1,
530nm, the camera, source quadrature, 840um ROI and the original 89-bin union
mask. Haze does not depend on the OLED arrangement and is linked to the prior
surface calculation without another haze solve.

| Layout | 1um separated camera contrast [%] | 0.5um [%] |
| --- | ---: | ---: |
| Diamond | 13.964710 | 13.789377 |
| RGB stripe | 5.449757 | 5.419533 |
| RGB-Delta | 14.410891 | 14.184711 |
| Windmill | 12.871896 | 12.933972 |

All four coarse cases execute all four native backends, and each fine case
executes CUDA Torch. The camera metric uses the wave models; geometric raw
CV remains separate. Maximum coarse wave optical relative L2 error is
7.79e-6. Repeated Diamond optical arrays match the earlier same-surface
arrays exactly. No low-pass, MWA or fitted cutoff is introduced.

These inputs differ in aperture area and within-aperture radiance as well
as arrangement. For example, after mean normalization, the stripe input
peak is about 3.976 versus Diamond's 15.061. The values therefore compare
these complete illustrative inputs rather than isolating lattice topology
or ranking commercial panels. Stripe's approximately 5.42 result does not
establish a reproduction of Cheng's undocumented camera/filter chain.
