"""Transparent SMS-inspired sparkle analysis, not an SMS-1000 implementation.

Public basis: Becker 2021, sections 3.3-3.5, allows Fourier masking or moving
window averaging to separate display periodicity. Exact proprietary settings
used by Cheng 2025 were not published. All choices here are explicit.
"""
import math
import numpy as np
from scipy.ndimage import convolve1d

from sparkle_multibackend import collect_analysis_pixels


def centered_crop(image, pitch_um, width_um):
    size = round(width_um / pitch_um)
    if not math.isclose(size * pitch_um, width_um) or size > min(image.shape):
        raise ValueError('ROI must fit and contain an integer number of optical samples')
    y = (image.shape[0] - size) // 2
    x = (image.shape[1] - size) // 2
    return np.asarray(image, dtype=np.float64)[y:y+size, x:x+size]


def fourier_separate(image, pitch_um, supercell_um=140., cutoff_um_inv=1/70., notch_radius_bins=0):
    """Keep DC and low-frequency random modulation; notch known lattice lines.

    Input ROI spans an integer number of the *complete* emission supercell.
    Notches depend on the input geometry, never on a target contrast value.
    """
    values = np.asarray(image, dtype=np.float64)
    ny, nx = values.shape
    counts = (ny * pitch_um / supercell_um, nx * pitch_um / supercell_um)
    if any(not math.isclose(v, round(v)) for v in counts):
        raise ValueError('Fourier ROI must span full emission supercells')
    if cutoff_um_inv <= 0 or not 0 <= notch_radius_bins <= 2:
        raise ValueError('Invalid spectral filter settings')
    fy = np.fft.fftfreq(ny, d=pitch_um)
    fx = np.fft.fftfreq(nx, d=pitch_um)
    radius = np.hypot(fy[:, None], fx[None, :])
    band = radius < cutoff_um_inv - 1e-12
    transfer = band.copy()
    ky, kx = (round(v) for v in counts)
    max_order = math.ceil(cutoff_um_inv * supercell_um)
    notched_bins = []
    for iy in range(-max_order, max_order+1):
        for ix in range(-max_order, max_order+1):
            if iy == ix == 0:
                continue
            if math.hypot(iy / supercell_um, ix / supercell_um) >= cutoff_um_inv - 1e-12:
                continue
            row, col = (iy * ky) % ny, (ix * kx) % nx
            notched_bins.append([iy * ky, ix * kx])
            for dy in range(-notch_radius_bins, notch_radius_bins+1):
                for dx in range(-notch_radius_bins, notch_radius_bins+1):
                    transfer[(row+dy) % ny, (col+dx) % nx] = False
    transfer[0, 0] = True
    spectrum = np.fft.fft2(values)
    separated = np.fft.ifft2(spectrum * transfer).real
    energy = np.abs(spectrum)**2
    periodic_energy = float(energy[band & ~transfer].sum())
    random_energy = float(energy[transfer].sum() - energy[0, 0])
    return separated, transfer, {
        'method': 'Fourier lattice notches plus circular band limit; DC retained',
        'supercell_um': supercell_um, 'cutoff_cycles_per_um': cutoff_um_inv,
        'cutoff_wavelength_um': 1 / cutoff_um_inv,
        'notch_radius_bins': notch_radius_bins, 'notched_lattice_bins_yx': notched_bins,
        'retained_non_DC_bins': int(transfer.sum()-1),
        'removed_periodic_power_within_band': periodic_energy,
        'retained_random_power': random_energy,
        'mean_preservation_error': float(separated.mean()-values.mean()),
    }


def box_kernel(width_samples):
    """Exact area-overlap box on the optical grid, including fractional edges."""
    if width_samples < 1 or not math.isfinite(width_samples):
        raise ValueError('Box must be at least one sample wide')
    half = math.ceil(width_samples/2+.5)
    centers = np.arange(-half, half+1, dtype=float)
    weights = np.maximum(0, np.minimum(centers+.5, width_samples/2)
                         - np.maximum(centers-.5, -width_samples/2))
    weights = weights[weights > 0]
    return weights / weights.sum()


def moving_average(image, pitch_um, width_um=140., passes=3):
    kernel = box_kernel(width_um / pitch_um)
    out = np.asarray(image, dtype=float)
    for _ in range(passes):
        for axis in (0, 1):
            out = convolve1d(out, kernel, axis=axis, mode='reflect')
    return out


def measurement_statistics(image, optical_pitch_um, camera_pitch_um):
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Filtered signal must be a finite 2D array')
    # Fourier masks are linear, not positivity-preserving. Integrate signed
    # filtered signals without clipping; reuse the brightness integrator through
    # a constant shift and restore that constant after its linear area average.
    offset = min(0., float(image.min()))
    pixels, geometry = collect_analysis_pixels(image-offset, optical_pitch_um, camera_pitch_um, 0.)
    pixels += offset
    if not np.isfinite(pixels).all() or pixels.mean() <= 0:
        raise ValueError('Measurement pixels must be finite with positive mean')
    mean = float(pixels.mean())
    std = float(pixels.std(ddof=0))
    return pixels, {'sparkle_contrast_percent': 100 * std / mean,
                    'mean': mean, 'std': std, 'ddof': 0,
                    'minimum': float(pixels.min()), 'maximum': float(pixels.max()),
                    'negative_filtered_measurement_pixels': int(np.count_nonzero(pixels < 0)),
                    'num_pixels': pixels.size, **geometry}


def analyze(image, optical_pitch_um=1., roi_width_um=560., camera_pitch_um=26.25):
    roi = centered_crop(image, optical_pitch_um, roi_width_um)
    separated, transfer, params = fourier_separate(roi, optical_pitch_um)
    pixels, stats = measurement_statistics(separated, optical_pitch_um, camera_pitch_um)
    sensitivities = []
    for label, cutoff, notch in (('notch_guard_1_bin', 1/70., 1),
                                  ('cutoff_0p8_over_green_pitch', .8/70., 0)):
        candidate, _, setting = fourier_separate(roi, optical_pitch_um,
                                                  cutoff_um_inv=cutoff, notch_radius_bins=notch)
        _, metric = measurement_statistics(candidate, optical_pitch_um, camera_pitch_um)
        sensitivities.append({'case': label, 'settings': setting, **metric})
    mwa = centered_crop(moving_average(image, optical_pitch_um), optical_pitch_um, roi_width_um)
    mwa_pixels, mwa_stats = measurement_statistics(mwa, optical_pitch_um, camera_pitch_um)
    sensitivities.append({'case': 'three_MWA_full_supercells',
                          'settings': {'width_um': 140., 'passes': 3,
                                       'reason': 'Complete exact 140 um source supercell; four nonidentical green bodies per cell.'},
                          **mwa_stats})
    _, direct = measurement_statistics(roi, optical_pitch_um, camera_pitch_um)
    return {'optical_roi': roi, 'separated': separated, 'transfer': transfer,
            'measurement_pixels': pixels,
            'mwa_filtered_optical_roi': mwa, 'mwa_measurement_pixels': mwa_pixels,
            'summary': {'primary': stats, 'primary_filter': params,
                        'direct_unseparated_camera_CV_percent': direct['sparkle_contrast_percent'],
                        'sensitivities': sensitivities,
                        'roi_width_um': roi_width_um,
                        'filter_order': 'Optical-grid spectral separation, then virtual detector area integration. This omits pre-filter detector aliasing; not proprietary SMS software.',
                        'includes_LDF': False}}


def flat_wave_control(model):
    """Exact flat phase-screen limit of the existing incoherent wave solver.

    Uses a single convolution with |IFFT(pupil)|^2, avoiding thousands of FFTs.
    No division by this control is applied to the AG image.
    """
    from sparkle_multibackend import enumerate_sources
    H, W = model.padded_shape
    fy = np.fft.fftfreq(H, d=model.dx_m)
    fx = np.fft.fftfreq(W, d=model.dx_m)
    pupil = (fy[:, None]**2 + fx[None, :]**2) <= (model.na / model.wavelength_m)**2
    psf = np.abs(np.fft.ifft2(pupil))**2
    sources = np.zeros((H, W), dtype=float)
    for y, x, weight in enumerate_sources(model):
        sources[y+model.pad, x+model.pad] += weight
    result = np.fft.ifft2(np.fft.fft2(sources) * np.fft.fft2(psf)).real
    h, w = model.shape
    return np.maximum(result[model.pad:model.pad+h, model.pad:model.pad+w], 0).astype(np.float32)
