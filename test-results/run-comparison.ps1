$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:JAX_PLATFORMS = 'cpu'
$taskCudaRuntime = Join-Path ((& uv cache dir).Trim()) 'archive-v0\4WYk6cbWPUV3KB3ITqhyH'
if (Test-Path -LiteralPath (Join-Path $taskCudaRuntime 'torch\lib\torch_cuda.dll')) {
    $env:PYTHONPATH = $taskCudaRuntime
}
& $taskPython (Join-Path $taskRoot 'oled-ag-sparkle\scripts\run_oled_demo.py') --out (Join-Path $PSScriptRoot 'oled-1mm-height1um-50um')
if ($LASTEXITCODE -ne 0) { throw 'OLED brightness simulation failed' }
& $taskPython (Join-Path $PSScriptRoot 'plot-direct-oled.py')
if ($LASTEXITCODE -ne 0) { throw 'Brightness plotting failed' }
& $taskPython (Join-Path $taskRoot 'oled-ag-sparkle\scripts\generate_report.py') --results (Join-Path $PSScriptRoot 'oled-1mm-height1um-50um') --out (Join-Path $PSScriptRoot 'report.html')
if ($LASTEXITCODE -ne 0) { throw 'HTML generation failed' }
