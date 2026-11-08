"""Transparent SMS-inspired sparkle analysis, not an SMS-1000 implementation.

Public basis: Becker 2021, sections 3.3-3.5, allows Fourier masking or moving
window averaging to separate display periodicity. Exact proprietary settings
used by Cheng 2025 were not published. All choices here are explicit.

The public SMS1000 brochure also specifies numerical low-pass filtering of
captured images. The retained sampled-camera common-notch benchmark omits that
visual bandwidth restriction and is not an SMS1000-equivalent measurement.
Do not change its historical plan silently or fit a cutoff to the paper value.
"""
import math
import numpy as np
from scipy.ndimage import convolve1d

from sparkle_multibackend import collect_analysis_pixels


def sampled_camera_plan(periods_um_xy, camera, wavelength_um=.530):
    """IEC 62977-3-9:2023 Annex B frequency masking, after camera capture.

    The standard specifies the method family, not this particular mask. Here
    exact reciprocal-lattice lines, including their sampled aliases, are removed
    with one common union mask. No additional visual low-pass or MWA is applied.
    """
    for periods in periods_um_xy:
        if len(periods) != 2 or any(not math.isfinite(p) or p <= 0 or
                                   not math.isclose(840/p, round(840/p)) or
                                   not math.isclose(420/p, round(420/p)) for p in periods):
            raise ValueError('Camera ROIs 840/420 um must span integer full G periods')
    return {
        'primary_metric': 'sampled_camera_common_notch',
        'method': 'Camera sampling first; common aliased reciprocal-lattice notches; DC retained',
        'order': ['incoherent sources -> AG thin phase screen -> focused circular-pupil wave imaging',
                  'square detector area integration -> discrete sensor samples',
                  'DFT -> common periodic-bin mask -> inverse DFT -> 100*std/mean'],
        'camera_pitch_um': camera['object_pitch_um'], 'roi_width_um': 840.,
        'sensitivity_roi_width_um': 420.,
        'sensor_phase_offsets_um_xy': [[0., 0.], [7., 0.], [0., 7.], [7., 7.]],
        'notch_periods_um_xy': [list(p) for p in periods_um_xy],
        'optical_intensity_cutoff_cycles_per_um': 2*camera['object_NA']/wavelength_um,
        'wavelength_um': wavelength_um, 'notch_radius_bins': 0,
        'additional_lowpass': None, 'apodization': None,
        'roi_rule': '840 um = 32 detector pixels = integer multiples of every exact G period; 420 um nested ROI is a sensitivity check',
        'mask_rule': 'Enumerate reciprocal vectors within the optical intensity bandwidth, fold modulo sensor sampling frequency, remove the union of exact DFT bins for all layouts; retain DC',
        'detector': 'Monochrome, linear, 100% square active area; noiseless, unquantized; no Bayer/ISP',
        'focus': 'OLED emission plane through the flat reference stack; ideal aberration-free pupil already present in the optical solver',
        'metric_scope': 'Camera-domain stochastic contrast under the declared mask, not a visually weighted or SMS-1000 calibrated reading',
        'references': [
            {'title': 'IEC 62977-3-9:2023, informative Annex B and Table B.1',
             'url': 'https://webstore.iec.ch/en/publication/64780',
             'use': 'Frequency-domain removal of periodic display modulation; compare only identical filter designs. The exact common mask here is explicitly specified by this protocol, not mandated by IEC.'},
            {'title': 'Becker (2021), 10 years of sparkle measurement, sections 3.1-3.4',
             'url': 'https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf',
             'use': 'Camera MTF, detector sampling, aliased periodic peaks, then Fourier masking of electronic images. The optional visual-frequency cutoff is not used for this camera-domain metric.'},
        ],
    }


def sample_camera(image, optical_pitch_um, roi_width_um, camera_pitch_um, offset_um_xy=(0., 0.)):
    """Capture full square pixels once, using the solver's area integrator."""
    values = np.asarray(image)
    size = round(roi_width_um/optical_pitch_um)
    shifts = [round(v/optical_pitch_um) for v in offset_um_xy]
    if not math.isclose(size*optical_pitch_um, roi_width_um) or any(
            not math.isclose(s*optical_pitch_um, v) for s, v in zip(shifts, offset_um_xy)):
        raise ValueError('Capture ROI and offsets must lie on the optical grid')
    x = (values.shape[1]-size)//2+shifts[0]
    y = (values.shape[0]-size)//2+shifts[1]
    if min(x, y) < 0 or x+size > values.shape[1] or y+size > values.shape[0]:
        raise ValueError('Camera ROI exceeds the saved optical field')
    pixels, geometry = collect_analysis_pixels(values[y:y+size, x:x+size], optical_pitch_um,
                                               camera_pitch_um, 0.)
    if not math.isclose(pixels.shape[0]*camera_pitch_um, roi_width_um):
        raise ValueError('Camera ROI must contain an integer number of sensor pixels')
    cx = (x+size/2-values.shape[1]/2)*optical_pitch_um
    cy = (y+size/2-values.shape[0]/2)*optical_pitch_um
    geometry['analysis_extent_um_xyxy'] = [cx-roi_width_um/2, cy-roi_width_um/2,
                                         cx+roi_width_um/2, cy+roi_width_um/2]
    geometry['sensor_phase_offset_um_xy'] = [cx, cy]
    geometry['metric_source_file'] = 'camera_pixels.npy'
    return pixels, geometry


def sampled_periodic_mask(shape, camera_pitch_um, periods_um_xy, optical_cutoff_um_inv):
    """Exact line notches with physical pre-sampling bandwidth and alias folding."""
    ny, nx = shape
    if min(ny, nx) < 2 or camera_pitch_um <= 0 or optical_cutoff_um_inv <= 0:
        raise ValueError('Invalid camera grid or bandwidth')
    transfer = np.ones(shape, dtype=bool)
    harmonics, bins = [], set()
    fs = 1/camera_pitch_um
    for px, py in periods_um_xy:
        if min(px, py) <= 0 or any(not math.isclose(v, round(v)) for v in
                                  (nx*camera_pitch_um/px, ny*camera_pitch_um/py)):
            raise ValueError('Sampled ROI must span integer full emission periods')
        for iy in range(-math.ceil(py*optical_cutoff_um_inv), math.ceil(py*optical_cutoff_um_inv)+1):
            for ix in range(-math.ceil(px*optical_cutoff_um_inv), math.ceil(px*optical_cutoff_um_inv)+1):
                fx, fy = ix/px, iy/py
                if not (ix or iy) or math.hypot(fx, fy) > optical_cutoff_um_inv+1e-12:
                    continue
                row, col = round(fy*ny*camera_pitch_um) % ny, round(fx*nx*camera_pitch_um) % nx
                folded = [(fx+fs/2) % fs-fs/2, (fy+fs/2) % fs-fs/2]
                harmonics.append({'period_um_xy': [px, py], 'harmonic_xy': [ix, iy],
                                  'frequency_cycles_per_um_xy': [fx, fy],
                                  'folded_cycles_per_um_xy': folded, 'bin_yx': [row, col],
                                  'aliased': abs(fx) >= fs/2 or abs(fy) >= fs/2})
                if row or col:
                    bins.add((row, col))
    for row, col in bins:
        transfer[row, col] = False
    return transfer, {'removed_bins_yx': sorted(bins), 'removed_non_DC_bins': len(bins),
                      'retained_non_DC_bins': int(transfer.sum()-1),
                      'sensor_nyquist_cycles_per_um': fs/2,
                      'optical_intensity_cutoff_cycles_per_um': optical_cutoff_um_inv,
                      'harmonics': harmonics, 'DC_aliases_unidentifiable': [h for h in harmonics if h['bin_yx'] == [0, 0]]}


def sampled_statistics(pixels, geometry=None):
    """Statistics of already captured pixels; no second integration or clipping."""
    values = np.asarray(pixels, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or values.mean() <= 0:
        raise ValueError('Captured image must be finite with positive mean')
    mean, std = float(values.mean()), float(values.std(ddof=0))
    return {**(geometry or {}), 'mean': mean, 'std': std, 'ddof': 0,
            'sparkle_contrast_percent': 100*std/mean, 'num_pixels': values.size,
            'minimum': float(values.min()), 'maximum': float(values.max()),
            'negative_filtered_measurement_pixels': int(np.count_nonzero(values < 0))}


def analyze_sampled_camera(image, optical_pitch_um, plan, roi_width_um=None, offset_um_xy=(0., 0.)):
    pixels, geometry = sample_camera(image, optical_pitch_um, roi_width_um or plan['roi_width_um'],
                                     plan['camera_pitch_um'], offset_um_xy)
    transfer, mask_info = sampled_periodic_mask(pixels.shape, plan['camera_pitch_um'],
                                               plan['notch_periods_um_xy'],
                                               plan['optical_intensity_cutoff_cycles_per_um'])
    separated = np.fft.ifft2(np.fft.fft2(pixels)*transfer).real
    primary = sampled_statistics(separated, {**geometry, 'metric_source_file': 'camera_separated.npy'})
    primary['mean_preservation_error'] = float(separated.mean()-pixels.mean())
    return {'camera_pixels': pixels, 'primary_measurement_pixels': separated, 'transfer': transfer,
            'summary': {'primary': primary, 'primary_filter': {**plan, **mask_info},
                        'direct_unseparated_camera_CV_percent': sampled_statistics(pixels)['sparkle_contrast_percent'],
                        'includes_LDF': False}}


def circular_pupil_mtf(frequency, cutoff):
    """Incoherent MTF of an ideal unobscured circular pupil (object coordinates)."""
    rho = np.clip(np.abs(frequency)/cutoff, 0., 1.)
    return 2/np.pi*(np.arccos(rho)-rho*np.sqrt(1-rho*rho))


def camera_mtf_audit(model, plan):
    """Check the actual discretized wave pupil against the circular-pupil OTF."""
    h, w = model.padded_shape
    fy, fx = np.fft.fftfreq(h, model.dx_m), np.fft.fftfreq(w, model.dx_m)
    pupil = fy[:, None]**2+fx[None, :]**2 <= (model.na/model.wavelength_m)**2
    psf = np.abs(np.fft.ifft2(pupil))**2
    otf = np.fft.fft2(psf).real/psf.sum()
    frequencies = fx*1e-6
    keep = (frequencies >= 0) & (frequencies <= plan['optical_intensity_cutoff_cycles_per_um'])
    frequencies, numerical = frequencies[keep], otf[0, keep]
    analytical = circular_pupil_mtf(frequencies, plan['optical_intensity_cutoff_cycles_per_um'])
    return {'frequency_cycles_per_um': frequencies.tolist(), 'numerical_pupil_MTF': numerical.tolist(),
            'analytic_circular_pupil_MTF': analytical.tolist(),
            'max_absolute_MTF_error': float(np.max(np.abs(numerical-analytical))),
            'padded_field_um_yx': [h*model.dx_m*1e6, w*model.dx_m*1e6],
            'note': 'Finite FFT pupil discretization; the existing wave pupil already supplies diffraction. Detector sinc is applied by subsequent pixel integration, not by another lens blur.'}


def periodic_camera_control(model, plan):
    """Infinite periodic flat-panel control, captured by analytic Fourier integrals.

    This removes finite-field edge effects from the null check. The actual AG
    result is never divided by either the finite or this periodic flat control.
    """
    width = plan['sensitivity_roi_width_um']
    cell = centered_crop(model.emission, model.em_dx_m*1e6, width)
    n = cell.shape[0]
    frequencies = np.fft.fftfreq(n, d=model.em_dx_m*1e6)
    valid = np.abs(frequencies) <= plan['optical_intensity_cutoff_cycles_per_um']
    f = frequencies[valid]
    coefficients = (np.fft.fft2(cell)/(n*n))[np.ix_(valid, valid)]
    radius = np.hypot(f[:, None], f[None, :])
    coefficients *= circular_pupil_mtf(radius, plan['optical_intensity_cutoff_cycles_per_um'])
    # Exact square detector integral of every Fourier component.
    sensor = np.sinc(f*plan['camera_pitch_um'])
    coefficients *= sensor[:, None]*sensor[None, :]
    size = round(plan['roi_width_um']/plan['camera_pitch_um'])
    centers = (np.arange(size)+.5)*plan['camera_pitch_um']-model.em_dx_m*1e6/2
    exponent = np.exp(2j*np.pi*centers[:, None]*f[None, :])
    pixels = (exponent@coefficients@exponent.T).real
    mask, _ = sampled_periodic_mask(pixels.shape, plan['camera_pitch_um'],
                                    plan['notch_periods_um_xy'], plan['optical_intensity_cutoff_cycles_per_um'])
    separated = np.fft.ifft2(np.fft.fft2(pixels)*mask).real
    return pixels, sampled_statistics(separated)


def centered_crop(image, pitch_um, width_um):
    size = round(width_um / pitch_um)
    if not math.isclose(size * pitch_um, width_um) or size > min(image.shape):
        raise ValueError('ROI must fit and contain an integer number of optical samples')
    y = (image.shape[0] - size) // 2
    x = (image.shape[1] - size) // 2
    return np.asarray(image, dtype=np.float64)[y:y+size, x:x+size]


def fourier_separate(image, pitch_um, supercell_um=140., cutoff_um_inv=1/70., notch_radius_bins=0,
                     notch_periods_um=None):
    """Keep DC and low-frequency random modulation; notch known lattice lines.

    Input ROI spans an integer number of the *complete* emission supercell.
    Notches depend on the input geometry, never on a target contrast value.
    """
    values = np.asarray(image, dtype=np.float64)
    ny, nx = values.shape
    periods = notch_periods_um if notch_periods_um is not None else [supercell_um]
    periods = [tuple([float(p)]*2) if np.isscalar(p) else tuple(p) for p in periods]
    if not periods or any(len(p) != 2 or any(not math.isfinite(v) or v <= 0 for v in p) for p in periods):
        raise ValueError('Periods must be positive finite x/y pairs')
    for px, py in periods:
        counts = (ny*pitch_um/py, nx*pitch_um/px)
        if any(not math.isclose(v, round(v)) for v in counts):
            raise ValueError('Fourier ROI must span full emission supercells')
    if cutoff_um_inv <= 0 or not 0 <= notch_radius_bins <= 2:
        raise ValueError('Invalid spectral filter settings')
    fy = np.fft.fftfreq(ny, d=pitch_um)
    fx = np.fft.fftfreq(nx, d=pitch_um)
    radius = np.hypot(fy[:, None], fx[None, :])
    band = radius < cutoff_um_inv - 1e-12
    transfer = band.copy()
    notched_bins = set()
    for px, py in periods:
        ky, kx = round(ny*pitch_um/py), round(nx*pitch_um/px)
        for iy in range(-math.ceil(cutoff_um_inv*py), math.ceil(cutoff_um_inv*py)+1):
            for ix in range(-math.ceil(cutoff_um_inv*px), math.ceil(cutoff_um_inv*px)+1):
                if (iy or ix) and math.hypot(iy/py, ix/px) < cutoff_um_inv-1e-12:
                    notched_bins.add((iy*ky, ix*kx))
    for row, col in sorted(notched_bins):
        row, col = row % ny, col % nx
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
        'supercell_um': supercell_um, 'notch_periods_um_xy': periods, 'cutoff_cycles_per_um': cutoff_um_inv,
        'cutoff_wavelength_um': 1 / cutoff_um_inv,
        'notch_radius_bins': notch_radius_bins, 'notched_lattice_bins_yx': sorted(notched_bins),
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


def moving_average(image, pitch_um, width_um=140., passes=3, windows_um_xy=None):
    """Cascade rectangular boxes; an explicit common schedule overrides repeats."""
    width_xy = [width_um]*2 if np.isscalar(width_um) else list(width_um)
    windows = windows_um_xy if windows_um_xy is not None else [width_xy]*passes
    if not windows or any(len(widths) != 2 for widths in windows):
        raise ValueError('MWA windows must be nonempty x/y pairs')
    out = np.asarray(image, dtype=float)
    for widths in windows:
        kernels = [box_kernel(width/pitch_um) for width in widths[::-1]]
        for axis in (0, 1):
            out = convolve1d(out, kernels[axis], axis=axis, mode='reflect')
    return out


def common_mwa_plan(periods_um_xy, pixel_pitch_um=70., field_um=1000.):
    """Period-derived common three-box cascade with the legacy 6P support.

    Include one integer-period box for every distinct lattice on each axis.
    Fill the remaining passes to retain the original total width 3*(2P).
    For 70/140/210 um X and 70/140 um Y this gives X=(70,140,210),
    Y=(140,140,140), without consulting any computed contrast values.
    """
    reference_width = 2*pixel_pitch_um
    total_width = 3*reference_width
    axis_windows = []
    for axis in (0, 1):
        periods = sorted({float(p[axis]) for p in periods_um_xy})
        if not periods or any(not math.isfinite(p) or p <= 0 for p in periods):
            raise ValueError('Positive finite periods are required')
        required = [p for p in periods if not any(q > p and math.isclose(q/p, round(q/p)) for q in periods)]
        remaining = 3-len(required)
        if remaining < 0 or (remaining == 0 and not math.isclose(sum(required), total_width)):
            raise ValueError('Periods do not fit three MWA passes with legacy total support')
        widths = required+[(total_width-sum(required))/remaining]*remaining if remaining else required
        if any(w <= 0 for w in widths):
            raise ValueError('Periods exceed the legacy MWA support budget')
        axis_windows.append(sorted(widths))
    windows = [list(pair) for pair in zip(*axis_windows)]
    roi = math.floor(min(560., field_um-total_width-20.))
    if roi <= 0:
        raise ValueError('Insufficient edge-free field for the common MWA cascade')
    return {'method': 'Three common rectangular MWA passes; exact period rejection; no LDF',
            'windows_um_xy': windows, 'reference_window_um': reference_width,
            'total_support_width_um_xy': [total_width, total_width],
            'roi_width_um': roi, 'camera_pitch_um': 26.25,
            'required_periods_um_xy': [list(p) for p in periods_um_xy],
            'selection': 'Geometry only: period-covering boxes plus positive widths filling the original three-box support; no contrast-value fitting'}


def measure_mwa(image, optical_pitch_um, roi_width_um=560., camera_pitch_um=26.25,
                width_um=140., passes=3, windows_um_xy=None):
    filtered = centered_crop(moving_average(image, optical_pitch_um, width_um, passes, windows_um_xy),
                             optical_pitch_um, roi_width_um)
    pixels, stats = measurement_statistics(filtered, optical_pitch_um, camera_pitch_um)
    return filtered, pixels, stats


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


def analyze(image, optical_pitch_um=1., roi_width_um=560., camera_pitch_um=26.25,
            supercell_um=140., cutoff_um_inv=1/70., notch_periods_um=None, mwa_passes=3):
    roi = centered_crop(image, optical_pitch_um, roi_width_um)
    separated, transfer, params = fourier_separate(roi, optical_pitch_um, supercell_um,
                                                 cutoff_um_inv, notch_periods_um=notch_periods_um)
    pixels, stats = measurement_statistics(separated, optical_pitch_um, camera_pitch_um)
    sensitivities = []
    for label, cutoff, notch in (('notch_guard_1_bin', cutoff_um_inv, 1),
                                  ('cutoff_0p8_over_green_pitch', .8*cutoff_um_inv, 0)):
        candidate, _, setting = fourier_separate(roi, optical_pitch_um,
                                                  supercell_um=supercell_um, cutoff_um_inv=cutoff,
                                                  notch_radius_bins=notch, notch_periods_um=notch_periods_um)
        _, metric = measurement_statistics(candidate, optical_pitch_um, camera_pitch_um)
        sensitivities.append({'case': label, 'settings': setting, **metric})
    mwa, mwa_pixels, mwa_stats = measure_mwa(image, optical_pitch_um, roi_width_um, camera_pitch_um,
                                           supercell_um, mwa_passes)
    sensitivities.append({'case': f'{mwa_passes}_MWA_full_supercells',
                          'settings': {'width_um': supercell_um, 'passes': mwa_passes,
                                       'reason': 'Complete exact source period; different window widths imply different bandwidths.'},
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


def analyze_saved_runs(results):
    """Reprocess the six saved reference runs and validate their float statistics."""
    import hashlib
    import json
    from pathlib import Path
    from datetime import datetime
    from sparkle_multibackend import ALL_BACKENDS, model_from_args, parse_args

    results = Path(results).expanduser().resolve()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    protocol = read(results / 'protocol.json')
    execution = protocol['execution']
    declared_at = datetime.fromisoformat(execution['created_at_utc'])
    assert protocol['includes_LDF'] is False
    assert execution['solver_sha256'] == hashlib.sha256(
        Path(__file__).with_name('sparkle_multibackend.py').read_bytes()).hexdigest()
    records = []
    arrays = {}
    source_hashes = set()
    for grid, names, pitch in (('1um', ALL_BACKENDS, 1.), ('0p5um', ('torch',), .5),
                                ('old-na-0p08', ('torch',), 1.)):
        run = read(results / grid / 'run.json')
        assert run['status'] == 'complete'
        assert run['parameters']['emission_channel'] == 'green'
        assert run['parameters']['backend_workers'] == execution['backend_workers']
        assert run['solver']['sha256'] == execution['solver_sha256']
        assert datetime.fromisoformat(run['started_at_utc']) >= declared_at
        for role in ('height', 'emission'):
            info = run['input_files'][role]
            assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest() == info['sha256']
            if role == 'emission':
                source_hashes.add(info['sha256'])
        for backend in names:
            folder = results / grid / backend
            raw = np.load(folder / 'brightness.npy')
            assert np.isfinite(raw).all() and np.all(raw >= 0)
            result = analyze(raw, pitch)
            data = result['summary']
            direct = read(folder / 'stats.json')
            assert direct['run_id'] == run['run_id']
            assert run['backend_runs'][backend]['status'] == 'complete'
            assert datetime.fromisoformat(direct['started_at_utc']) >= declared_at
            context = direct['execution_context']
            if backend in ('torch', 'torchoptics'):
                assert str(context['device']).startswith('cuda')
            elif backend == 'chromatix':
                assert context['default_backend'] == 'gpu' and context['runtime'] == 'wsl'
            else:
                assert context['variant'] == 'cuda_ad_rgb'
            assert abs(data['direct_unseparated_camera_CV_percent'] - direct['sparkle_cv_percent']) < 1e-9
            pixels = result['measurement_pixels']
            cv = 100 * pixels.std(ddof=0) / pixels.mean()
            assert abs(cv - data['primary']['sparkle_contrast_percent']) < 1e-10
            mwa_pixels = result['mwa_measurement_pixels']
            mwa_cv = 100 * mwa_pixels.std(ddof=0) / mwa_pixels.mean()
            assert abs(mwa_cv-data['sensitivities'][2]['sparkle_contrast_percent']) < 1e-10
            assert pixels.shape == (21, 21)
            assert abs(data['primary_filter']['mean_preservation_error']) < 1e-12
            for filename, values in (('sms_filtered_optical_roi.npy', result['separated']),
                                     ('sms_measurement_pixels.npy', pixels),
                                     ('sms_frequency_mask.npy', result['transfer']),
                                     ('sms_mwa_optical_roi.npy', result['mwa_filtered_optical_roi']),
                                     ('sms_mwa_measurement_pixels.npy', mwa_pixels)):
                np.save(folder / filename, values)
            data.update({'grid': grid, 'backend': backend, 'object_NA': run['parameters']['na'],
                         'physics': direct['physics_group'], 'solver_seconds': direct['elapsed_seconds'],
                         'batch_id': execution['batch_id'], 'run_id': run['run_id'],
                         'solver_sha256': run['solver']['sha256'],
                         'started_at_utc': direct['started_at_utc'],
                         'finished_at_utc': direct['finished_at_utc'],
                         'execution_context': context,
                         'measured_ROI_global_um_xyxy': [-275.625, -275.625, 275.625, 275.625],
                         'filtered_ROI_global_um_xyxy': [-280., -280., 280., 280.]})
            (folder / 'sms_stats.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
            arrays[(grid, backend)] = result
            records.append(data)
    assert len(source_hashes) == 1
    for grid in ('1um', 'old-na-0p08'):
        run = read(results / grid / 'run.json')
        model = model_from_args(parse_args(run['argv']))
        control = flat_wave_control(model)
        result = analyze(control, model.dx_m * 1e6)
        folder = results / 'controls' / grid
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'brightness.npy', control)
        np.save(folder / 'sms_measurement_pixels.npy', result['measurement_pixels'])
        (folder / 'sms_stats.json').write_text(json.dumps(result['summary'], indent=2), encoding='utf-8')
        arrays[(grid, 'flat')] = result
    return records, arrays


def analyze_layout_runs(results):
    """Audit frozen optical runs and apply their explicitly selected measurement.

    New camera protocols capture pixels before the common aliased lattice mask.
    Historical optical-grid Fourier/MWA protocols remain reproducible.
    Existing optical protocols and raw brightness arrays are never rewritten.
    """
    import hashlib
    import json
    from pathlib import Path
    from datetime import datetime
    from sparkle_multibackend import model_from_args, parse_args

    results = Path(results).resolve()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    protocol = read(results/'protocol.json')
    if protocol['kind'] != 'oled-layout-comparison':
        raise ValueError('Not a layout comparison protocol')
    solver_hash = hashlib.sha256(Path(__file__).with_name('sparkle_multibackend.py').read_bytes()).hexdigest()
    assert protocol['execution']['solver_sha256'] == solver_hash
    optical_root = Path(protocol.get('optical_root', results))
    camera_mode = protocol['analysis'].get('primary_metric') == 'sampled_camera_common_notch'
    if 'optical_source_protocol_sha256' in protocol:
        assert hashlib.sha256((optical_root/'protocol.json').read_bytes()).hexdigest() == protocol['optical_source_protocol_sha256']
        for filename, expected in protocol['reused_files_sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    declared_at = datetime.fromisoformat(protocol['execution']['created_at_utc'])
    periods = sorted({tuple(meta['green_period_um_xy']) for meta in protocol['layouts'].values()})
    if camera_mode:
        assert protocol['analysis'] == sampled_camera_plan(periods, protocol['camera'])
    if not camera_mode:
        common = common_mwa_plan(periods, protocol['nominal_pixel_pitch_um'],
                                 next(iter(protocol['layouts'].values()))['assigned_field_um'][0])
        if 'common_mwa' in protocol['analysis']:
            assert common == protocol['analysis']['common_mwa']
        common_settings = {k: common[k] for k in ('windows_um_xy', 'roi_width_um', 'camera_pitch_um')}
        reference_settings = {'roi_width_um': common['roi_width_um'], 'camera_pitch_um': common['camera_pitch_um'],
                              'width_um': common['reference_window_um'], 'passes': 3}
    records, arrays, masks = [], {}, {}
    optical_signature = None
    for plan in protocol['runs']:
        run = read(optical_root/plan['output']/'run.json')
        assert run['status'] == 'complete' and run['argv'] == plan['argv']
        assert run['solver']['sha256'] == solver_hash
        assert datetime.fromisoformat(run['started_at_utc']) >= declared_at
        for role in ('height', 'emission'):
            info = run['input_files'][role]
            actual = hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()
            assert actual == info['sha256']
            if role == 'height':
                assert actual == protocol['height_sha256'][info['path']]
            else:
                assert actual == protocol['layouts'][plan['layout']]['sha256']
        args = parse_args(run['argv'])
        model = model_from_args(args)
        mean = float(model.emission.mean(dtype=np.float64))
        assert abs(mean-protocol['target_linear_green_mean']) < 2e-7
        assert args.emission_threshold == 0 and args.emission_channel == 'green'
        if camera_mode:
            signature = (model.na, model.wavelength_m, model.gap_m, model.n_stack, model.n_out,
                         model.em_dx_m, model.source_samples, args.emission_threshold,
                         model.emission.shape, tuple(model.shape), model.pad)
            # Grid shape and padding deliberately differ in the fine-grid run;
            # all optical constants, emission quadrature and field sizes agree.
            physical_signature = signature[:8]+(signature[8],
                tuple(v*model.dx_m for v in model.shape), model.pad*model.dx_m)
            if optical_signature is None:
                optical_signature = physical_signature
            assert physical_signature == optical_signature
            assert math.isclose(model.na, protocol['camera']['object_NA'])
            assert math.isclose(model.dx_m*1e6, plan['pitch_um'])
            assert math.isclose(model.em_dx_m*1e6, protocol['layouts'][plan['layout']]['assigned_emission_texel_um'])
            assert math.isclose(model.wavelength_m*1e6, protocol['analysis']['wavelength_um'])
            capture = protocol['analysis']
            flat_values = flat_wave_control(model)
            flat = analyze_sampled_camera(flat_values, plan['pitch_um'], capture)
            periodic_pixels, periodic_null = periodic_camera_control(model, capture)
            assert periodic_null['sparkle_contrast_percent'] < 1e-8
            control = results/'controls'/plan['output']
            control.mkdir(parents=True, exist_ok=True)
            mtf_audit = camera_mtf_audit(model, capture)
            assert mtf_audit['max_absolute_MTF_error'] < .03
            np.save(control/'finite_flat_camera_pixels.npy', flat['camera_pixels'])
            np.save(control/'finite_flat_separated.npy', flat['primary_measurement_pixels'])
            np.save(control/'periodic_flat_camera_pixels.npy', periodic_pixels)
            flat_checks = []
            for offset in capture['sensor_phase_offsets_um_xy']:
                f = analyze_sampled_camera(flat_values, plan['pitch_um'], capture, offset_um_xy=offset)
                flat_checks.append({'offset_um_xy': offset, **f['summary']['primary']})
            small_flat = analyze_sampled_camera(flat_values, plan['pitch_um'], capture,
                                                roi_width_um=capture['sensitivity_roi_width_um'])
            (control/'camera_stats.json').write_text(json.dumps({
                'finite_flat': flat['summary'], 'periodic_flat_null': periodic_null, 'camera_mtf_audit': mtf_audit,
                'phase_checks': flat_checks, 'nested_ROI': small_flat['summary']['primary']}, indent=2), encoding='utf-8')
            for backend in plan['backends']:
                if backend == 'mitsuba':
                    continue  # Geometric rays do not implement the diffraction-limited camera.
                source_folder = optical_root/plan['output']/backend
                direct = read(source_folder/'stats.json')
                assert direct['run_id'] == run['run_id'] and run['backend_runs'][backend]['status'] == 'complete'
                assert datetime.fromisoformat(direct['started_at_utc']) >= declared_at
                assert direct['physics_group'] == 'scalar_wave'
                raw_path = source_folder/'brightness.npy'
                raw = np.load(raw_path)
                assert np.isfinite(raw).all() and np.all(raw >= 0)
                result = analyze_sampled_camera(raw, plan['pitch_um'], capture)
                if plan['grid'] in masks:
                    assert np.array_equal(result['transfer'], masks[plan['grid']])
                masks[plan['grid']] = result['transfer']
                phases = []
                for offset in capture['sensor_phase_offsets_um_xy']:
                    candidate = analyze_sampled_camera(raw, plan['pitch_um'], capture, offset_um_xy=offset)
                    phases.append({'offset_um_xy': offset, **candidate['summary']['primary']})
                small = analyze_sampled_camera(raw, plan['pitch_um'], capture,
                                               roi_width_um=capture['sensitivity_roi_width_um'])
                folder = results/plan['output']/backend
                folder.mkdir(parents=True, exist_ok=True)
                for filename, data in (('camera_pixels', result['camera_pixels']),
                                       ('camera_separated', result['primary_measurement_pixels']),
                                       ('camera_frequency_mask', result['transfer'])):
                    np.save(folder/(filename+'.npy'), data)
                record = {**result['summary'], 'primary_metric': capture['primary_metric'],
                          'grid': plan['grid'], 'layout': plan['layout'], 'backend': backend,
                          'solver_seconds': direct['elapsed_seconds'], 'execution_context': direct['execution_context'],
                          'run_id': run['run_id'], 'batch_id': protocol['execution']['batch_id'],
                          'raw_optical_file': str(raw_path), 'raw_optical_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                          'height_sha256': run['input_files']['height']['sha256'],
                          'normalized_input_mean': mean, 'linear_emission_scale': args.emission_scale,
                          'green_period_um_xy': protocol['layouts'][plan['layout']]['green_period_um_xy'],
                          'flat_control_percent': flat['summary']['primary']['sparkle_contrast_percent'],
                          'periodic_flat_null_percent': periodic_null['sparkle_contrast_percent'],
                          'camera_mtf_audit': mtf_audit,
                          'phase_checks': phases, 'nested_ROI': small['summary']['primary'],
                          'nested_ROI_flat_percent': small_flat['summary']['primary']['sparkle_contrast_percent'],
                          'nested_ROI_mask_removed_bins': small['summary']['primary_filter']['removed_non_DC_bins'],
                          'nested_ROI_same_full_mask': sampled_statistics(
                              centered_crop(result['primary_measurement_pixels'], capture['camera_pitch_um'],
                                            capture['sensitivity_roi_width_um']))}
                (folder/'camera_stats.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                records.append(record)
                arrays[(plan['grid'], plan['layout'], backend)] = result
            continue
        settings = {k: v for k, v in protocol['analysis'].items()
                    if k in ('roi_width_um', 'camera_pitch_um', 'cutoff_um_inv', 'notch_periods_um', 'mwa_passes')}
        settings['supercell_um'] = protocol['layouts'][plan['layout']]['green_period_um_xy']
        flat_values = flat_wave_control(model)
        flat = analyze(flat_values, plan['pitch_um'], **settings)
        _, flat_common_pixels, flat_common_stats = measure_mwa(flat_values, plan['pitch_um'], **common_settings)
        _, _, flat_reference_stats = measure_mwa(flat_values, plan['pitch_um'], **reference_settings)
        _, _, input_periodic = measure_mwa(model.emission, model.em_dx_m*1e6, **common_settings)
        assert input_periodic['sparkle_contrast_percent'] < 1e-8
        control = results/'controls'/plan['output']
        control.mkdir(parents=True, exist_ok=True)
        np.save(control/'sms_measurement_pixels.npy', flat['measurement_pixels'])
        np.save(control/'sms_common_mwa_measurement_pixels.npy', flat_common_pixels)
        control_stats = {**flat['summary'], 'fourier': flat['summary']['primary'],
                         'primary': flat_common_stats, 'primary_filter': common,
                         'reference_mwa': flat_reference_stats}
        (control/'sms_stats.json').write_text(json.dumps(control_stats, indent=2), encoding='utf-8')
        for backend in plan['backends']:
            folder = results/plan['output']/backend
            direct = read(folder/'stats.json')
            assert direct['run_id'] == run['run_id'] and run['backend_runs'][backend]['status'] == 'complete'
            assert datetime.fromisoformat(direct['started_at_utc']) >= declared_at
            raw = np.load(folder/'brightness.npy')
            assert np.isfinite(raw).all() and np.all(raw >= 0)
            result = analyze(raw, plan['pitch_um'], **settings)
            if plan['grid'] in masks:
                assert np.array_equal(result['transfer'], masks[plan['grid']])
            else:
                masks[plan['grid']] = result['transfer']
            measured = result['measurement_pixels']
            assert abs(100*measured.std()/measured.mean()-result['summary']['primary']['sparkle_contrast_percent']) < 1e-10
            common_roi, common_pixels, common_stats = measure_mwa(raw, plan['pitch_um'], **common_settings)
            _, reference_pixels, reference_stats = measure_mwa(raw, plan['pitch_um'], **reference_settings)
            common_stats['metric_source_file'] = 'sms_common_mwa_measurement_pixels.npy'
            reference_stats['metric_source_file'] = 'sms_reference_mwa_measurement_pixels.npy'
            for name, data in (('sms_filtered_optical_roi', result['separated']),
                               ('sms_measurement_pixels', measured), ('sms_frequency_mask', result['transfer']),
                               ('sms_mwa_measurement_pixels', result['mwa_measurement_pixels']),
                               ('sms_common_mwa_optical_roi', common_roi),
                               ('sms_common_mwa_measurement_pixels', common_pixels),
                               ('sms_reference_mwa_measurement_pixels', reference_pixels)):
                np.save(folder/(name+'.npy'), data)
            record = {**result['summary'], 'fourier': result['summary']['primary'],
                      'fourier_filter': result['summary']['primary_filter'],
                      'primary': common_stats, 'primary_filter': common,
                      'primary_metric': 'common_three_pass_MWA', 'reference_mwa': reference_stats,
                      'grid': plan['grid'], 'layout': plan['layout'], 'backend': backend,
                      'solver_seconds': direct['elapsed_seconds'], 'execution_context': direct['execution_context'],
                      'run_id': run['run_id'], 'batch_id': protocol['execution']['batch_id'],
                      'height_sha256': run['input_files']['height']['sha256'],
                      'normalized_input_mean': mean, 'linear_emission_scale': args.emission_scale,
                      'green_period_um_xy': settings['supercell_um'],
                      'flat_control_percent': flat_common_stats['sparkle_contrast_percent'],
                      'reference_mwa_flat_percent': flat_reference_stats['sparkle_contrast_percent'],
                      'fourier_flat_percent': flat['summary']['primary']['sparkle_contrast_percent'],
                      'input_periodic_residual_percent': input_periodic['sparkle_contrast_percent']}
            (folder/'sms_stats.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            records.append(record)
            result['common_mwa_measurement_pixels'] = common_pixels
            result['primary_measurement_pixels'] = common_pixels
            arrays[(plan['grid'], plan['layout'], backend)] = result
    return protocol, records, arrays
