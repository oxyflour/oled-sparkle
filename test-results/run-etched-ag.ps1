$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$taskCudaRuntime = Join-Path ((& uv cache dir).Trim()) 'archive-v0\4WYk6cbWPUV3KB3ITqhyH'
if (Test-Path -LiteralPath (Join-Path $taskCudaRuntime 'torch\lib\torch_cuda.dll')) {
    $env:PYTHONPATH = $taskCudaRuntime
}
$taskRunner = Join-Path $taskRoot 'oled-ag-sparkle\scripts\run_oled_demo.py'
& $taskPython $taskRunner --height-surface etched-ag --device cuda --dry-run --out (Join-Path $PSScriptRoot 'oled-etched-ag-dry-run')
if ($LASTEXITCODE -ne 0) { throw 'Etched AG dry-run failed' }
& $taskPython $taskRunner --height-surface etched-ag --device cuda --backend all --out (Join-Path $PSScriptRoot 'oled-etched-ag\1um')
if ($LASTEXITCODE -ne 0) { throw 'Etched AG four-backend run failed' }
& $taskPython $taskRunner --height-surface etched-ag --device cuda --pitch-um .5 --backend torch --out (Join-Path $PSScriptRoot 'oled-etched-ag\0p5um')
if ($LASTEXITCODE -ne 0) { throw 'Etched AG finer-grid run failed' }
& $taskPython (Join-Path $PSScriptRoot 'plot-etched-ag.py')
if ($LASTEXITCODE -ne 0) { throw 'Etched AG report audit failed' }
