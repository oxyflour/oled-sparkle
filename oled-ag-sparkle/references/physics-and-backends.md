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
The emission grid remains 500x500 at 2um. The refined height map interpolates the
existing 16-bit height data; it does not add new measured surface detail.

The statistic is directly `100 * std(I_ROI, ddof=0) / mean(I_ROI)` on the saved
`analysis_brightness.npy` values. All full measurement pixels, including dark
pixels, enter the statistic. The largest full pixel grid is centered inside the
requested geometric ROI; record its exact boundaries. No reference division,
signal-based masking or additional frequency filtering is used. Mean must be positive.

Source quadrature, FFT padding, height-grid density, mesh resolution and pupil
quadrature affect the simulated images. Record them with source provenance,
height mapping, wavelength, indices, gap, camera NA and actual computation times.
PNG previews are only display representations; metrics use the saved float array.

The bundled example uses the existing synthetic height map and a crop of Samsung
Display's Diamond Pixel image. Physical scale and effective monochromatic luma
interpretation are explicit example assumptions in the source metadata.
