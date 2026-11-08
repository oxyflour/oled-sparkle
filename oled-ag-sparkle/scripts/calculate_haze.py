#!/usr/bin/env python3
"""Estimate monochromatic transmission haze from a periodic AG height map.

Uniform normally incident illumination; the existing thin phase-screen model,
without OLED emitters or a camera pupil. This is a surface-only scalar estimate,
not an ASTM/ISO luminous-haze measurement. Run --help for the reusable CLI.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

from sparkle_multibackend import read_height
from prepare_etched_ag import DEFAULT_CASE, DEFAULT_INPUTS, roughness

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = [
    {'title': 'ASTM D1003-21, haze and luminous transmittance measurement',
     'url': 'https://store.astm.org/d1003-21.html',
     'use': 'Defines the measurement context; this scalar monochromatic calculation does not implement the complete standard.'},
    {'title': 'Simonsen et al., Phys. Rev. A 79, 063813 (2009), sections III and IV.1',
     'url': 'https://arxiv.org/html/0904.3753',
     'use': 'Normalize angular scattered power by total transmitted power; 2.5 degree acceptance interval. Its 1D phase-perturbation approximation is not used as a 2D solver here.'},
]


def transmission_haze(height_m, pitch_m, wavelength_m=530e-9,
                      n_in=1.5, n_out=1., acceptance_deg=2.5, angular_power=None):
    """Return scalar normal-flux haze and an angular cumulative-power curve.

    A=FFT(exp(i*2*pi*(n_in-n_out)*(h-mean(h))/lambda))/N.
    sin(theta)=lambda*sqrt(fx^2+fy^2)/n_out; each Cartesian Fourier bin carries
    normal scalar flux proportional to |A|^2*cos(theta). The common df_x*df_y
    cancels in the ratio. No extra solid-angle Jacobian is applied to these bins.
    Evanescent bins carry no far-field flux. All represented propagating bins,
    including the DC order and narrow-angle diffuse light, enter the denominator.
    """
    h = np.asarray(height_m, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 8 or not np.isfinite(h).all():
        raise ValueError('Height must be a finite 2D array at least 8 by 8')
    if not all(math.isfinite(v) and v > 0 for v in (pitch_m, wavelength_m, n_in, n_out)):
        raise ValueError('Pitch, wavelength and refractive indices must be positive and finite')
    if not math.isfinite(acceptance_deg) or not 0 < acceptance_deg < 90:
        raise ValueError('Acceptance half-angle must lie strictly between 0 and 90 degrees')
    nyquist_sin = wavelength_m/(2*pitch_m*n_out)
    if math.sin(math.radians(acceptance_deg)) >= nyquist_sin:
        raise ValueError('Height sampling cannot cover the acceptance cone in every azimuth; refine the grid')

    if angular_power is None:
        phase = 2*np.pi*(n_in-n_out)*(h-h.mean())/wavelength_m
        coefficients = np.fft.fft2(np.exp(1j*phase), norm='forward')
        power = np.abs(coefficients)**2
    else:
        power = np.asarray(angular_power, dtype=np.float64)
        if power.shape != h.shape or not np.isfinite(power).all() or np.any(power < 0):
            raise ValueError('Backend angular power must be finite, nonnegative and match the height grid')
    fy = np.fft.fftfreq(h.shape[0], d=pitch_m)
    fx = np.fft.fftfreq(h.shape[1], d=pitch_m)
    sin_theta = wavelength_m/n_out*np.hypot(fy[:, None], fx[None, :])
    propagating = sin_theta < 1
    cos_theta = np.sqrt(np.maximum(0., 1-sin_theta**2))
    flux = power*cos_theta
    flux[~propagating] = 0.
    total = float(flux.sum())
    if total <= 0:
        raise ValueError('No represented propagating transmitted power')
    cutoff = math.sin(math.radians(acceptance_deg))
    outside = propagating & (sin_theta > cutoff)
    haze = 100*float(flux[outside].sum())/total

    # Raw angle integration, not smoothing or an angular filter. nextafter
    # makes each curve row include bins exactly on its stated cone boundary.
    angles = np.unique(np.r_[np.linspace(0., 90., 901), acceptance_deg])
    theta = np.degrees(np.arcsin(np.clip(sin_theta[propagating], 0., 1.)))
    histogram, _ = np.histogram(theta, bins=np.r_[-1., np.nextafter(angles, np.inf)],
                                weights=flux[propagating])
    cumulative = np.clip(np.cumsum(histogram)/total, 0., 1.)
    curve = [{'half_angle_deg': float(a), 'within_cone_percent': float(100*c),
              'outside_cone_percent': float(100*(1-c))} for a, c in zip(angles, cumulative)]
    edge = (np.abs(fy[:, None]) >= .9/(2*pitch_m)) | (np.abs(fx[None, :]) >= .9/(2*pitch_m))
    result = {
        'haze_percent': haze, 'wavelength_nm': wavelength_m*1e9,
        'n_in': n_in, 'n_out': n_out, 'acceptance_half_angle_deg': acceptance_deg,
        'definition': '100 * scalar forward flux outside acceptance cone / all represented propagating scalar forward flux',
        'phase_model': 'exp(i*2*pi*(n_in-n_out)*(h-mean(h))/vacuum_wavelength)',
        'flux_weight': '|normalized FFT(field)|^2 * cos(theta); Cartesian frequency bins',
        'height_grid_yx': list(h.shape), 'height_pitch_um': pitch_m*1e6,
        'field_um_xy': [h.shape[1]*pitch_m*1e6, h.shape[0]*pitch_m*1e6],
        'represented_scalar_forward_flux': total,
        'zero_order_forward_flux_percent': 100*float(flux[0, 0])/total,
        'within_cone_nonzero_order_flux_percent': 100*float(flux[propagating & ~outside].sum()-flux[0, 0])/total,
        'nonzero_order_forward_flux_percent': 100*(total-float(flux[0, 0]))/total,
        'haze_without_cosine_weight_percent': 100*float(power[outside].sum())/float(power[propagating].sum()),
        'parseval_relative_error': abs(float(power.sum())-1.),
        'evanescent_field_power_percent': 100*float(power[~propagating].sum())/float(power.sum()),
        'spectral_edge_field_power_percent': 100*float(power[edge].sum())/float(power.sum()),
        'spectral_edge_definition': 'Diagnostic only: outer 10% of either Cartesian frequency axis; no bins are filtered by this diagnostic',
        'complete_azimuth_angle_limit_deg': math.degrees(math.asin(min(1., nyquist_sin))),
        'near_axis_angle_step_deg_xy': [math.degrees(math.asin(min(1., wavelength_m/(n_out*s*pitch_m))))
                                        for s in h.shape[::-1]],
        'acceptance_frequency_cycles_per_um': n_out*cutoff/wavelength_m*1e-6,
        'boundary': 'Periodic continuation; no aperture, zero padding, apodization or detrending beyond mean removal',
        'model_status': 'Monochromatic surface-only scalar estimate; no standard-conformity or measured-material claim',
    }
    return result, curve


def facet_slope_diagnostics(height_m, pitch_m, n_in=1.5, n_out=1., acceptance_deg=2.5):
    """Audit the two periodic triangles per cell without tracing rays.

    This is an analytic check of the existing single-hit mesh approximation,
    not another native backend or an electromagnetic interface solution.
    """
    h = np.asarray(height_m, dtype=np.float64)
    gx = (np.roll(h, -1, axis=1)-h)/pitch_m
    gy = (np.roll(h, -1, axis=0)-h)/pitch_m
    # Same diagonal as create_mitsuba_mesh: (v0,v1,v2), (v1,v3,v2).
    slopes = np.concatenate((np.hypot(gx, gy).ravel(),
        np.hypot(np.roll(gx, -1, axis=0), np.roll(gy, -1, axis=1)).ravel()))
    incidence = np.arctan(slopes)
    eta = n_in/n_out
    sine = eta*np.sin(incidence)
    propagating = sine < 1.
    ci = np.cos(incidence)
    ct = np.sqrt(np.maximum(0., 1-sine*sine))
    rs = ((eta*ci-ct)/(eta*ci+ct))**2
    rp = ((ci-eta*ct)/(ci+eta*ct))**2
    weights = np.where(propagating, 1-(rs+rp)/2, 0.)
    angles = np.abs(np.arcsin(np.clip(sine, 0., 1.))-incidence)
    outside = angles > math.radians(acceptance_deg)
    alpha = math.radians(acceptance_deg)
    threshold = math.sin(alpha)/(eta-math.cos(alpha)) if eta > 1 else None
    summary = {
        'definition': 'Projected-area statistics of the original periodic triangular facets; float64 analytic Snell/Fresnel, no native ray execution',
        'rms_slope': float(np.sqrt(np.mean(slopes**2))),
        'slope_quantiles': dict(zip(('p50', 'p90', 'p99', 'p99_9'), map(float, np.percentile(slopes, [50, 90, 99, 99.9])))),
        'slope_threshold_for_acceptance': threshold,
        'area_above_slope_threshold_percent': float(100*np.mean(slopes > threshold)) if threshold is not None else None,
        'single_hit_haze_percent': float(100*weights[outside].sum()/weights.sum()) if weights.sum() else None,
        'transmission_fraction': float(weights.mean()),
        'total_internal_reflection_area_percent': float(100*np.mean(~propagating)),
    }
    edges = np.linspace(0., max(.15, float(slopes.max())), 301)
    count, _ = np.histogram(slopes, edges)
    curve = [{'slope': float(s), 'area_above_percent': float(100*(1-c))}
             for s, c in zip(edges[1:], np.cumsum(count)/slopes.size)]
    return summary, curve


def surface_diagnostics(height_m, pitch_m, wavelength_m=530e-9,
                        n_in=1.5, n_out=1., acceptance_deg=2.5):
    """Height spectrum/correlation and slope audit for the existing haze report.

    Radial spectrum rows contain shell-integrated height variance, not a PSD
    density or an optical power. Correlation is circular and azimuthally averaged;
    its 1/e crossing is a declared descriptor, not a fitted Gaussian length.
    """
    h = np.asarray(height_m, dtype=np.float64)
    z = h-h.mean()
    variance = float(np.mean(z*z))
    power = np.abs(np.fft.fft2(z, norm='forward'))**2
    fy = np.fft.fftfreq(h.shape[0], pitch_m)
    fx = np.fft.fftfreq(h.shape[1], pitch_m)
    frequency = np.hypot(fy[:, None], fx[None, :])
    cutoff = n_out*math.sin(math.radians(acceptance_deg))/wavelength_m
    step = min(1/(s*pitch_m) for s in h.shape)
    shells = np.rint(frequency/step).astype(np.int64)
    counts = np.bincount(shells.ravel())
    shell_power = np.bincount(shells.ravel(), weights=power.ravel())
    shell_frequency = np.bincount(shells.ravel(), weights=frequency.ravel())/np.maximum(counts, 1)
    spectrum = [{'frequency_cycles_per_um': float(f*1e-6), 'height_variance_um2': float(v*1e12)}
                for f, v, count in zip(shell_frequency, shell_power, counts) if count]
    # Wrap displacements to their shortest periodic distance. Use each shell's
    # mean actual radius, rather than assigning floor(r/dx)*dx to all its points.
    radius = np.hypot(fy[:, None]*h.shape[0]*pitch_m, fx[None, :]*h.shape[1]*pitch_m)*pitch_m
    rings = np.rint(radius/pitch_m).astype(np.int64)
    count = np.bincount(rings.ravel())
    radii = np.bincount(rings.ravel(), weights=radius.ravel())/np.maximum(count, 1)
    covariance = np.fft.ifft2(power, norm='forward').real
    acf = np.bincount(rings.ravel(), weights=covariance.ravel())/np.maximum(count, 1)
    acf = acf/variance if variance else np.ones_like(acf)
    valid = (count > 0) & (radii <= min(h.shape)*pitch_m/2)
    radii, acf = radii[valid], acf[valid]
    crossings = np.flatnonzero(acf < 1/math.e)
    crossing = None
    if len(crossings):
        j = crossings[0]
        crossing = float(np.interp(1/math.e, acf[j-1:j+1][::-1], radii[j-1:j+1][::-1])*1e6)
    facets, slopes = facet_slope_diagnostics(h, pitch_m, n_in, n_out, acceptance_deg)
    summary = {
        'height_skewness': float(np.mean(z**3)/variance**1.5) if variance else None,
        'height_kurtosis': float(np.mean(z**4)/variance**2) if variance else None,
        'circular_radial_acf_first_1e_length_um': crossing,
        'acf_definition': 'Circular height covariance / Sq^2; nearest-radius shells at their mean actual radii; first 1/e crossing with linear interpolation, no Gaussian fit',
        'spectral_rms_slope': float(np.sqrt(np.sum((2*np.pi*frequency)**2*power))),
        'height_variance_above_acceptance_frequency_percent': float(100*power[frequency > cutoff].sum()/variance) if variance else None,
        'acceptance_frequency_cycles_per_um': cutoff*1e-6,
        'acceptance_lateral_period_um': 1e6/cutoff,
        'phase_rms_rad': 2*np.pi*abs(n_in-n_out)*math.sqrt(variance)/wavelength_m,
        'facet_model': facets,
        'interpretation': 'Height PSD is not angular optical power. Phase RMS need not be small; no first-order height-PSD haze formula is assumed.',
    }
    curves = {'radial_height_spectrum': spectrum,
              'radial_height_acf': [{'radius_um': float(r*1e6), 'correlation': float(c)} for r, c in zip(radii, acf)],
              'facet_slope_survival': slopes}
    return summary, curves


def solve_wave_haze(height_m, pitch_m, wavelength_m, n_in, n_out,
                    backend, device='cuda', propagation_um=100.):
    """Run each native wave library with plane illumination and no camera pupil.

    Homogeneous output propagation is actually executed. Its phase transfer has
    unit modulus on propagating bins, so angular power should be independent of
    propagation distance. This check does not validate the thin-screen boundary.
    """
    if not math.isfinite(propagation_um) or propagation_um <= 0:
        raise ValueError('Output propagation distance must be positive and finite')
    if backend == 'chromatix':
        import sys
        if sys.platform == 'win32':
            from chromatix_wsl import simulate_haze_chromatix_wsl
            return simulate_haze_chromatix_wsl(height_m, dict(
                pitch_m=pitch_m, wavelength_m=wavelength_m, n_in=n_in, n_out=n_out,
                propagation_um=propagation_um))
        import jax
        import jax.numpy as jnp
        import chromatix
        import chromatix.functional as cx
        jax.config.update('jax_enable_x64', True)
        h = jnp.asarray(height_m, dtype=jnp.float64)*1e6
        wavelength = wavelength_m*1e6
        field = cx.plane_wave(shape=h.shape, dx=pitch_m*1e6, spectrum=wavelength)
        incident_power = jnp.mean(jnp.abs(field.u)**2)
        phase = 2*jnp.pi*(n_in-n_out)*(h-h.mean())/wavelength
        field = cx.phase_change(field, phase, spectrally_modulate=False)
        propagated = cx.asm_propagate(field, z=propagation_um, n=n_out,
                                      pad_width=0, remove_evanescent=True)
        power = jnp.abs(jnp.fft.fft2(propagated.u.reshape(h.shape), norm='forward'))**2/incident_power
        power.block_until_ready()
        return np.asarray(power, dtype=np.float64), {
            'library': 'Chromatix', 'library_version': version('chromatix'),
            'jax_version': jax.__version__, 'default_backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()], 'dtype': str(power.dtype),
            'native_operations': 'plane_wave -> phase_change -> asm_propagate -> JAX angular FFT',
            'output_propagation_um': propagation_um}
    if backend not in ('torch', 'torchoptics'):
        raise ValueError(f'Not a wave backend: {backend}')
    import torch
    h = torch.as_tensor(np.asarray(height_m, dtype=np.float64), device=device)
    phase = 2*torch.pi*(n_in-n_out)*(h-h.mean())/wavelength_m
    modulation = torch.exp(1j*phase)
    context = {'library': backend, 'torch_version': torch.__version__, 'device': str(h.device),
               'dtype': 'complex128', 'output_propagation_um': propagation_um}
    if device.startswith('cuda'):
        context['gpu_name'] = torch.cuda.get_device_name(h.device)
    if backend == 'torchoptics':
        import torchoptics
        torchoptics.set_default_dtype(torch.float64)
        field = torchoptics.Field(torch.ones_like(modulation), z=0.,
                                 wavelength=wavelength_m/n_out, spacing=(pitch_m, pitch_m)).to(device)
        field = field.modulate(modulation)
        propagated = field.propagate_to_z(propagation_um*1e-6, propagation_method='ASM',
                                          asm_pad=(0, 0)).data
        context.update({'library_version': torchoptics.__version__,
                        'native_operations': 'Field -> modulate -> propagate_to_z(ASM) -> Torch angular FFT'})
    else:
        fy = torch.fft.fftfreq(h.shape[0], d=pitch_m, dtype=torch.float64, device=device)
        fx = torch.fft.fftfreq(h.shape[1], d=pitch_m, dtype=torch.float64, device=device)
        kz2 = (n_out/wavelength_m)**2-fy[:, None]**2-fx[None, :]**2
        kz = torch.sqrt(torch.clamp(kz2, min=0))
        transfer = torch.exp(2j*torch.pi*propagation_um*1e-6*(kz-n_out/wavelength_m))*(kz2 >= 0)
        propagated = torch.fft.ifft2(torch.fft.fft2(modulation)*transfer)
        context['native_operations'] = 'Torch plane field -> phase modulation -> ASM -> angular FFT'
    power = torch.abs(torch.fft.fft2(propagated, norm='forward'))**2
    return power.detach().cpu().numpy(), context


def solve_mitsuba_haze(height_m, pitch_m, wavelength_m, n_in, n_out,
                       acceptance_deg, variant='cuda_ad_rgb', ray_chunk=65536):
    """Forward collimated illumination of the existing measured-height mesh.

    Periodic endpoint vertices cover exactly N*dx. One vertical ray through each
    triangle centroid integrates its constant geometric normal/Fresnel weight
    exactly over projected area. Use geometric face normals, not shading normals.
    Each ray already represents incident power: do not multiply by cos(theta)
    again. Only first-interface transmission is considered.
    """
    import mitsuba as mi
    import drjit as dr
    from types import SimpleNamespace
    from sparkle_multibackend import create_mitsuba_mesh
    if ray_chunk < 1:
        raise ValueError('Ray chunk must be positive')
    mi.set_variant(variant)
    h = np.asarray(height_m, dtype=np.float64)
    ny, nx = h.shape
    periodic = np.pad(h, ((0, 1), (0, 1)), mode='wrap')
    model = SimpleNamespace(h_m=periodic, gap_m=0., dx_m=pitch_m,
                            mesh_stride=1, shape=periodic.shape)
    mesh, vertices, faces = create_mitsuba_mesh(mi, model)
    scene = mi.load_dict({'type': 'scene', 'ag_surface': mesh})
    dx = pitch_m*1e6
    bottom = (float(periodic.min())-float(periodic.mean()))*1e6-4*dx
    count = 2*ny*nx
    cutoff_cos = math.cos(math.radians(acceptance_deg))
    angles = np.unique(np.r_[np.linspace(0., 90., 901), acceptance_deg])
    histogram = np.zeros(len(angles), dtype=np.float64)
    total = outside = reflected = 0.
    hits = transmitted = 0
    for start in range(0, count, ray_chunk):
        end = min(start+ray_chunk, count)
        index = dr.arange(mi.UInt, end-start)+start
        cell = index//2
        offset = dr.select(index % 2 == 0, 1/3, 2/3)
        x = (mi.Float(cell % nx)+offset-nx/2)*dx
        y = (mi.Float(cell//nx)+offset-ny/2)*dx
        ray = mi.Ray3f(mi.Point3f(x, y, bottom), mi.Vector3f(0., 0., 1.))
        si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.Minimal, coherent=False)
        # The normal points to the outside medium; incidence is from inside.
        wi = mi.Vector3f(0., 0., -1.)
        cos_i = dr.dot(wi, si.n)
        reflectance, cos_t, _, eta_ti = mi.fresnel(cos_i, n_in/n_out)
        outgoing = mi.refract(wi, si.n, cos_t, eta_ti)
        valid = si.is_valid() & (outgoing.z > 0) & (reflectance < 1)
        weights = dr.select(valid, 1-reflectance, 0.)
        dr.eval(weights, outgoing, reflectance)
        w = np.asarray(weights, dtype=np.float64)
        cosines = np.asarray(outgoing.z, dtype=np.float64)
        hit = np.asarray(si.is_valid(), dtype=bool)
        hits += int(hit.sum())
        transmitted += int(np.count_nonzero(w > 0))
        total += float(w.sum())
        outside += float(w[cosines < cutoff_cos].sum())
        reflected += float(np.asarray(dr.select(si.is_valid(), reflectance, 0.), dtype=np.float64).sum())
        theta = np.degrees(np.arccos(np.clip(cosines, -1., 1.)))
        histogram += np.histogram(theta, bins=np.r_[-1., np.nextafter(angles, np.inf)], weights=w)[0]
    if hits != count or total <= 0:
        raise RuntimeError(f'Mitsuba haze coverage failed: {hits}/{count} rays hit, total={total}')
    cumulative = np.clip(np.cumsum(histogram)/total, 0., 1.)
    curve = [{'half_angle_deg': float(a), 'within_cone_percent': float(100*c),
              'outside_cone_percent': float(100*(1-c))} for a, c in zip(angles, cumulative)]
    result = {'haze_percent': 100*outside/total, 'wavelength_nm': wavelength_m*1e9,
              'acceptance_half_angle_deg': acceptance_deg, 'n_in': n_in, 'n_out': n_out,
              'physics_group': 'geometric_rays', 'height_grid_yx': [ny, nx],
              'height_pitch_um': pitch_m*1e6, 'field_um_xy': [nx*pitch_m*1e6, ny*pitch_m*1e6],
              'incident_rays': count, 'hit_rays': hits, 'transmitted_rays': transmitted,
              'mesh_vertices': vertices, 'mesh_faces': faces,
              'transmission_fraction': total/count, 'reflection_fraction': reflected/count,
              'energy_balance_error': abs((total+reflected)/count-1.),
              'definition': '100 * transmitted ray power outside acceptance cone / all forward transmitted ray power',
              'boundary': 'Periodically closed height mesh, N*dx projected area, two triangle-centroid rays per original grid cell',
              'model_status': 'Single-interface geometric refraction with native unpolarized Fresnel transmission; no diffraction or multiple scattering'}
    context = {'library': 'Mitsuba', 'library_version': mi.__version__, 'variant': mi.variant(),
               'native_operations': 'create_mitsuba_mesh -> ray_intersect -> fresnel -> refract',
               'ray_chunk': ray_chunk, 'normal': 'geometric triangle normal'}
    return result, curve, context


def load_surface(path, pitch_um=None, height_range_um=None, assume_periodic=False):
    """Read the existing PNG calibration and validate its optional sidecar."""
    path = Path(path).expanduser().resolve()
    sidecar = path.with_suffix('.source.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get('sha256', digest) != digest:
        raise ValueError(f'Height PNG does not match its source hash: {path}')
    for explicit, key in ((pitch_um, 'assigned_texel_um'), (height_range_um, 'height_range_um')):
        if explicit is not None and key in metadata and not math.isclose(explicit, metadata[key]):
            raise ValueError(f'Explicit {key} conflicts with the sidecar; provide a correctly calibrated input')
    pitch_um = pitch_um if pitch_um is not None else metadata.get('assigned_texel_um')
    height_range_um = height_range_um if height_range_um is not None else metadata.get('height_range_um')
    if any(v is None or not math.isfinite(v) or v <= 0 for v in (pitch_um, height_range_um)):
        raise ValueError('Provide height pitch and black-to-white range, or a valid .source.json sidecar')
    height = read_height(path, height_range_um)
    dimensions = np.array(height.shape[::-1])*pitch_um
    periodic_domain = metadata.get('periodic_domain_um')
    if periodic_domain is None:
        if not assume_periodic:
            raise ValueError('This estimator requires periodic continuation; use --assume-periodic only for a justified periodic input')
    elif not np.allclose(dimensions, periodic_domain):
        raise ValueError('Sidecar periodic domain disagrees with height dimensions and pitch')
    provenance = {'path': str(path), 'sha256': digest, 'height_range_um': height_range_um,
                  'height_pitch_um': pitch_um, 'periodicity': 'source metadata' if periodic_domain else 'explicit user assumption',
                  'roughness': roughness(height.astype(np.float64)*1e6, pitch_um)}
    if sidecar.exists():
        provenance.update({'sidecar_path': str(sidecar), 'sidecar_sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest()})
    return height, pitch_um, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height', type=Path, nargs='+', default=[DEFAULT_INPUTS[1.]/'heightfield.png'])
    parser.add_argument('--height-pitch-um', type=float, help='Read from sidecar by default')
    parser.add_argument('--height-range-um', type=float, help='PNG black-to-white range, not sampled Sz; read from sidecar by default')
    parser.add_argument('--wavelength-nm', type=float, nargs='+', default=[530.], help='Independent monochromatic calculations, not a white-light average')
    parser.add_argument('--n-in', type=float, default=1.5)
    parser.add_argument('--n-out', type=float, default=1.)
    parser.add_argument('--acceptance-deg', type=float, default=2.5, help='Half-angle about the direct beam')
    parser.add_argument('--assume-periodic', action='store_true', help='Explicitly justify periodic continuation when metadata does not declare it')
    parser.add_argument('--backend', choices=('numpy', 'torch', 'torchoptics', 'chromatix', 'mitsuba', 'all'), default='numpy',
                        help='numpy: original estimate; all: actual four native backends plus the NumPy reference')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='Native Torch/TorchOptics device')
    parser.add_argument('--propagation-um', type=float, default=100., help='Actual homogeneous output propagation before wave angular-spectrum measurement')
    parser.add_argument('--mitsuba-variant', default='cuda_ad_rgb')
    parser.add_argument('--ray-chunk', type=int, default=65536)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs'/(DEFAULT_CASE+'-haze'))
    args = parser.parse_args()
    records = []
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    identifiers = [path.resolve().parent.name for path in args.height]
    if len(set(identifiers)) != len(identifiers):
        parser.error('Input height maps need distinct parent directory names for result identifiers')
    native = args.backend != 'numpy'
    if native:
        protocol_path = args.out/'protocol.json'
        if protocol_path.exists():
            parser.error('A native comparison protocol already exists; use a new output directory')
        surfaces = []
        for path in args.height:
            _, _, provenance = load_surface(path, args.height_pitch_um, args.height_range_um, args.assume_periodic)
            surfaces.append(provenance)
        protocol = {'kind': 'ag-haze-four-backend', 'declared_at_utc': datetime.now(timezone.utc).isoformat(),
                    'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'height'},
                    'inputs': surfaces, 'illumination': 'Uniform normally incident plane wave from n_in to n_out',
                    'wave_model': 'Existing scalar thin phase-screen boundary, native homogeneous ASM, no camera pupil',
                    'ray_model': 'Existing height mesh with periodic endpoints; triangle-centroid projected-area quadrature; native Mitsuba Fresnel/refraction',
                    'reference': 'Original NumPy phase-screen angular estimate, recomputed on the identical height samples',
                    'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                                    ('calculate_haze.py', 'sparkle_multibackend.py', 'chromatix_wsl.py')}}
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    for path, identifier in zip(args.height, identifiers):
        height, pitch, provenance = load_surface(path, args.height_pitch_um, args.height_range_um, args.assume_periodic)
        for wavelength in args.wavelength_nm:
            result, curve = transmission_haze(height, pitch*1e-6, wavelength*1e-9,
                                              args.n_in, args.n_out, args.acceptance_deg)
            if native:
                backends = ('torch', 'torchoptics', 'chromatix', 'mitsuba') if args.backend == 'all' else (args.backend,)
                reference = result
                phase = 2*np.pi*(args.n_in-args.n_out)*(height.astype(np.float64)-height.mean(dtype=np.float64))/(wavelength*1e-9)
                reference_power = np.abs(np.fft.fft2(np.exp(1j*phase), norm='forward'))**2
                for backend in backends:
                    folder = args.out/identifier/backend/f'{wavelength:g}nm'
                    folder.mkdir(parents=True, exist_ok=True)
                    native_started_at = datetime.now(timezone.utc).isoformat()
                    native_start = time.perf_counter()
                    if backend == 'mitsuba':
                        result, curve, context = solve_mitsuba_haze(height, pitch*1e-6, wavelength*1e-9,
                            args.n_in, args.n_out, args.acceptance_deg, args.mitsuba_variant, args.ray_chunk)
                        seconds = time.perf_counter()-native_start
                        flat, _, _ = solve_mitsuba_haze(np.zeros_like(height), pitch*1e-6, wavelength*1e-9,
                            args.n_in, args.n_out, args.acceptance_deg, args.mitsuba_variant, args.ray_chunk)
                    else:
                        power, context = solve_wave_haze(height, pitch*1e-6, wavelength*1e-9, args.n_in, args.n_out,
                                                         backend, args.device, args.propagation_um)
                        seconds = time.perf_counter()-native_start
                        np.save(folder/'angular_power.npy', power, allow_pickle=False)
                        result, curve = transmission_haze(height, pitch*1e-6, wavelength*1e-9,
                            args.n_in, args.n_out, args.acceptance_deg, angular_power=power)
                        result['physics_group'] = 'scalar_phase_screen_wave'
                        result['angular_power_relative_L2_vs_numpy'] = float(np.linalg.norm(power-reference_power)/np.linalg.norm(reference_power))
                        flat_power, _ = solve_wave_haze(np.zeros_like(height), pitch*1e-6, wavelength*1e-9,
                            args.n_in, args.n_out, backend, args.device, args.propagation_um)
                        flat, _ = transmission_haze(np.zeros_like(height), pitch*1e-6, wavelength*1e-9,
                            args.n_in, args.n_out, args.acceptance_deg, angular_power=flat_power)
                    assert flat['haze_percent'] < 1e-12
                    result.update({'backend': backend, 'surface_id': identifier, 'input': provenance,
                        'reference_numpy_percent': reference['haze_percent'],
                        'difference_vs_numpy_percentage_points': result['haze_percent']-reference['haze_percent'],
                        'execution_context': context, 'solver_seconds': seconds,
                        'started_at_utc': native_started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                        'flat_control_percent': flat['haze_percent'],
                        'angular_curve_csv': str((folder/'angular.csv').relative_to(args.out)).replace('\\', '/')})
                    with (folder/'angular.csv').open('w', encoding='utf-8', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
                        writer.writeheader()
                        writer.writerows(curve)
                    (folder/'stats.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
                    records.append(result)
                    print(f'{identifier} | {backend} | {wavelength:g} nm | haze = {result["haze_percent"]:.6f}% | NumPy = {reference["haze_percent"]:.6f}% | {seconds:.3f} s', flush=True)
                continue
            flat, _ = transmission_haze(np.zeros_like(height), pitch*1e-6, wavelength*1e-9,
                                        args.n_in, args.n_out, args.acceptance_deg)
            filename = f'{identifier}-{wavelength:g}nm-angular.csv'
            with (args.out/filename).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
                writer.writeheader()
                writer.writerows(curve)
            result.update({'surface_id': identifier, 'input': provenance,
                           'angular_curve_csv': filename, 'flat_control_percent': flat['haze_percent']})
            records.append(result)
            print(f'{identifier} | {wavelength:g} nm | haze = {result["haze_percent"]:.6f}% | flat = {flat["haze_percent"]:.3g}%')
    summary = {'kind': 'ag-haze-four-backend' if native else 'ag-transmission-haze-estimate', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.perf_counter()-started,
               'calculator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'height_reader_sha256': hashlib.sha256(Path(__file__).with_name('sparkle_multibackend.py').read_bytes()).hexdigest(),
               'records': records, 'references': REFERENCES,
               'limitations': [
                   'Normal-incidence plane wave at a single dielectric rough interface; identical local thin phase approximation to the sparkle wave model.',
                   'No OLED source pattern, camera NA, focal blur or pixel integration: haze is a separate angular transmission measurement.',
                   'No polarization, angle-dependent Fresnel factors, multiple reflections, bulk scattering, absorption or unresolved fine roughness.',
                   'Only represented propagating directions are integrated. Report angular coverage, spectral edge power and independently sampled grid convergence.',
                   'No claim of absolute total transmittance; haze is a ratio of transmitted scalar flux.',
                   'Separate wavelengths are not luminous haze. A white-light value requires source spectrum, V(lambda), dispersion and transmitted-power weighting, plus the instrument correction for a standard measurement.',
               ]}
    if native:
        summary.update({'protocol': protocol,
                        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                        'comparison_scope': 'Three native wave implementations of one phase-screen approximation versus a native geometric refraction model. Agreement among wave backends is numerical consistency, not independent Maxwell validation.'})
        summary['limitations'][0] = 'Normal-incidence plane wave at one rough dielectric interface. Wave backends use a thin phase screen; Mitsuba uses geometric mesh refraction with Fresnel weights.'
        summary['limitations'][2] = 'Wave backends omit polarization and Fresnel amplitudes; Mitsuba includes unpolarized Fresnel transmission but omits diffraction. Both omit volume scatter and multiple interactions.'
    (args.out/'haze.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('Results:', (args.out/'haze.json').resolve())


if __name__ == '__main__':
    main()
