"""Transfer a Windows Chromatix model to a CUDA-only WSL Python worker."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def _query(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Chromatix WSL missing: {error}. See references/windows-wsl.md') from error
    if result.returncode:
        raise RuntimeError(f'Chromatix WSL setup failed: {result.stderr.strip() or result.stdout.strip()}. '
                           'See references/windows-wsl.md')
    return result.stdout.strip()


def simulate_chromatix_wsl(model, max_sources):
    parameters = {field.name: getattr(model, field.name) for field in fields(model)
                  if field.name not in ('h_m', 'emission')}
    image, response = _execute_worker({'model': parameters, 'max_sources': max_sources},
                                      {'h_m': model.h_m, 'emission': model.emission}, 'brightness.npy')
    if image.shape != model.shape or not np.isfinite(image).all():
        raise RuntimeError('Chromatix WSL returned an invalid brightness image')
    model._chromatix_execution_context = response['execution_context']
    return image, response['source_count']


def simulate_haze_chromatix_wsl(height_m, parameters):
    """Use the same CUDA-only transport for the uniform-incidence haze case."""
    power, response = _execute_worker({'kind': 'haze', 'parameters': parameters},
                                      {'h_m': height_m}, 'angular_power.npy')
    if power.shape != height_m.shape or not np.isfinite(power).all() or np.any(power < 0):
        raise RuntimeError('Chromatix WSL returned invalid haze angular power')
    return power, response['execution_context']


def _execute_worker(request, arrays, output_filename):
    distro = os.environ.get('SPARKLE_WSL_DISTRO', 'Ubuntu-22.04')
    prefix = ['wsl.exe', '--distribution', distro, '--exec']
    python = os.environ.get('SPARKLE_WSL_PYTHON')
    if not python:
        home = _query(prefix + ['printenv', 'HOME'])
        python = home.rstrip('/') + '/.venvs/oled-sparkle-chromatix/bin/python'
    worker = _query(prefix + ['wslpath', '-a', '-u', str(Path(__file__).resolve())])
    with tempfile.TemporaryDirectory(prefix='chromatix-wsl-') as directory:
        root = Path(directory)
        np.savez(root / 'inputs.npz', **arrays)
        (root / 'request.json').write_text(json.dumps(request), encoding='utf-8')
        linux_root = _query(prefix + ['wslpath', '-a', '-u', str(root.resolve())])
        command = prefix + ['env', '-u', 'PYTHONPATH', '-u', 'LD_LIBRARY_PATH',
                            'JAX_PLATFORMS=cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
                            'XLA_PYTHON_CLIENT_MEM_FRACTION=' + os.environ.get(
                                'XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10'),
                            'PYTHONUNBUFFERED=1', python, worker, '--request-dir', linux_root]
        print(f'  Chromatix WSL: {distro}, JAX_PLATFORMS=cuda', flush=True)
        try:
            result = subprocess.run(command)
        except OSError as error:
            raise RuntimeError(f'Chromatix WSL launch failed: {error}') from error
        if result.returncode:
            raise RuntimeError(f'Chromatix WSL worker failed (exit {result.returncode}); '
                               'see worker output and references/windows-wsl.md')
        response = json.loads((root / 'result.json').read_text(encoding='utf-8'))
        image = np.load(root / output_filename, allow_pickle=False)
        context = response['execution_context']
        if context['default_backend'] != 'gpu':
            raise RuntimeError('Chromatix WSL worker did not use the GPU')
        context.update({'runtime': 'wsl', 'wsl_distribution': distro,
                        'worker_script': worker, 'worker_command': command})
        return image, response


def run_worker(root):
    import jax
    from reporting import environment_snapshot, execution_context
    from sparkle_multibackend import Model, simulate_chromatix

    if jax.default_backend() != 'gpu':
        raise RuntimeError(f'WSL worker requires the CUDA backend; got {jax.devices()}')
    request = json.loads((root / 'request.json').read_text(encoding='utf-8'))
    if request.get('kind') == 'haze':
        from calculate_haze import solve_wave_haze
        with np.load(root/'inputs.npz', allow_pickle=False) as arrays:
            height = arrays['h_m']
        start = time.perf_counter()
        power, context = solve_wave_haze(height, backend='chromatix', **request['parameters'])
        context.update({'environment': environment_snapshot(),
                        'worker_solver_elapsed_seconds': time.perf_counter()-start})
        np.save(root/'angular_power.npy', power, allow_pickle=False)
        (root/'result.json').write_text(json.dumps({'execution_context': context}, indent=2), encoding='utf-8')
        return
    with np.load(root / 'inputs.npz', allow_pickle=False) as arrays:
        model = Model(h_m=arrays['h_m'], emission=arrays['emission'], **request['model'])
    start = time.perf_counter()
    image, source_count = simulate_chromatix(model, request['max_sources'])
    elapsed = time.perf_counter() - start
    context = execution_context('chromatix', model)
    context.update({'default_backend': jax.default_backend(),
                    'gpu_names': [device.device_kind for device in jax.devices()],
                    'environment': environment_snapshot(),
                    'worker_solver_elapsed_seconds': elapsed})
    np.save(root / 'brightness.npy', image, allow_pickle=False)
    (root / 'result.json').write_text(json.dumps({
        'source_count': source_count, 'execution_context': context}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-dir', required=True, type=Path)
    run_worker(parser.parse_args().request_dir)
