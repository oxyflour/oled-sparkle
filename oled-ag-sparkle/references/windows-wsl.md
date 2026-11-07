# Windows Chromatix through WSL2 CUDA

The Windows solver automatically transfers the model's height/emission arrays and
physical parameters to `scripts/chromatix_wsl.py` in WSL. The worker returns raw
brightness and execution metadata; Windows performs the same postprocessing and
builds the combined report. This applies to `--backend chromatix`, `--backend all`,
and direct calls to `simulate_chromatix`. Linux uses its local implementation.

## Environment

Default distribution: `Ubuntu-22.04`. Default Python:
`$HOME/.venvs/oled-sparkle-chromatix/bin/python` inside that distribution. Override
these from PowerShell when another distribution or environment is needed:

```powershell
$env:SPARKLE_WSL_DISTRO = 'Ubuntu-22.04'
$env:SPARKLE_WSL_PYTHON = '/home/your-user/.venvs/oled-sparkle-chromatix/bin/python'
```

Python overrides are absolute Linux paths. The tested environment on this machine
is `/home/oxyflour/.venvs/oled-sparkle-chromatix/bin/python` in `Ubuntu-22.04`.

For a fresh environment, run in the selected WSL distribution (with `uv` available):

```bash
uv venv --python 3.12 "$HOME/.venvs/oled-sparkle-chromatix"
uv pip install --python "$HOME/.venvs/oled-sparkle-chromatix/bin/python" \
  'jax[cuda13]==0.12.0.1' numpy pillow \
  'chromatix @ git+https://github.com/chromatix-team/chromatix.git@ce1482906fc663298613bb252bbf425e3be59839'
```

If the default package source is slow, append
`--index-url https://pypi.tuna.tsinghua.edu.cn/simple` to the install command.
Verify `nvidia-smi` inside WSL before running the solver. The tested machine uses
an RTX 3090 Ti and Windows NVIDIA driver 596.49. CUDA libraries come from the JAX
pip dependencies; this setup does not require installing a Linux NVIDIA driver.

## Run and inspect

Run the normal Windows commands from the skill folder:

```powershell
python scripts/run_legacy_demo.py --example quick --backend chromatix --out outputs/quick-wsl
python scripts/run_oled_demo.py --backend all --out outputs/oled-1mm-etched-ag-periodic-50um
```

`--device` selects the Windows Torch device. Chromatix's WSL worker explicitly
sets `JAX_PLATFORMS=cuda`, even when the Windows parent has `JAX_PLATFORMS=cpu`.
It clears the worker's inherited `PYTHONPATH` and `LD_LIBRARY_PATH` to keep Windows
library paths out of the Linux environment. A failed worker is reported as failed;
there is no automatic fallback to Windows CPU JAX.

Each `chromatix/stats.json` includes `execution_context.runtime = "wsl"`,
`devices`, `gpu_names`, `wsl_distribution`, and the actual Linux
`environment.executable`, platform and package versions. Parent solver elapsed
time includes WSL launch and array transfer; `worker_solver_elapsed_seconds`
records the computation separately. Initialization and JIT are included.

The worker disables JAX GPU preallocation and defaults to a 0.10 allocator
fraction for concurrent backend runs. Override `XLA_PYTHON_CLIENT_MEM_FRACTION`
in the Windows process if a larger model needs more memory. Shared Windows
paths are converted using `wslpath`; the script and temporary directory must be
accessible from the selected distribution.

The Windows bridge was verified on the complete 1000x1000 grid and 76,550 source
modes: 61.39 s including launch/transfer, with 58.76 s inside the worker. Relative
L2 brightness error against the saved Windows CPU reference was 1.43e-6. All 23
tests passed, including mixed-runtime backend comparisons, Unicode/space paths,
and worker failure reporting. `VALIDATION.json` records these bridge measurements
separately from the historical four-backend reference run.
