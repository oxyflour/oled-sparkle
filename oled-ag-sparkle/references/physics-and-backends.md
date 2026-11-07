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
illustrative assumption. The current height is a synthetic random
population of nonoverlapping circular pits, 3-5um in diameter, 0.5um deep, covering
50% of the field. Their smooth bowl profile and random-placement assumptions are
recorded in the input sidecar. Only 3-5 samples span a pit at 1um pitch; compare
both the pit surface and original surface at a finer pitch to assess sensitivity.

The statistic is directly `100 * std(I_ROI, ddof=0) / mean(I_ROI)` on the saved
`analysis_brightness.npy` values. All full measurement pixels, including dark
pixels, enter the statistic. The largest full pixel grid is centered inside the
requested geometric ROI; record its exact boundaries. No reference division,
signal-based masking or additional frequency filtering is used. Mean must be positive.

Source quadrature, FFT padding, height-grid density, mesh resolution and pupil
quadrature affect the simulated images. Record them with source provenance,
height mapping, wavelength, indices, gap, camera NA and actual computation times.
PNG previews are only display representations; metrics use the saved float array.

The bundled example uses the synthetic random-pit height map and a crop of Samsung
Display's Diamond Pixel image. Physical scale and effective monochromatic luma
interpretation are explicit example assumptions in the source metadata.
