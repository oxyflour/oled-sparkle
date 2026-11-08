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
1,600 irregular Voronoi bowls, 25um nominal site spacing, variable depths and
1.2um rim rounding, calibrated at areal Sa=0.10um. Its 16-bit PNG encoding range
is 0.5361845861372564um, distinct from sampled Sz about 0.515um. The input sidecar
records calibration and geometry. Compare the same geometry and scale on a 0.5um
grid to assess sampling sensitivity. Circular pits remain an explicit legacy case.

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
