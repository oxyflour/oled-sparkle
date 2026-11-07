$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$taskCudaRuntime = Join-Path ((& uv cache dir).Trim()) 'archive-v0\4WYk6cbWPUV3KB3ITqhyH'
if (Test-Path -LiteralPath (Join-Path $taskCudaRuntime 'torch\lib\torch_cuda.dll')) {
    $env:PYTHONPATH = $taskCudaRuntime
}
& $taskPython (Join-Path $taskRoot 'oled-ag-sparkle\scripts\run_oled_demo.py') --height-surface original --backend all --out (Join-Path $PSScriptRoot 'oled-1mm-original-periodic-50um')
if ($LASTEXITCODE -ne 0) { throw 'Original-surface tiled-emission run failed' }
& $taskPython (Join-Path $taskRoot 'oled-ag-sparkle\scripts\run_oled_demo.py') --backend all --out (Join-Path $PSScriptRoot 'oled-1mm-pits3to5um-periodic-50um')
if ($LASTEXITCODE -ne 0) { throw 'Random-pit four-backend run failed' }
& $taskPython (Join-Path $PSScriptRoot 'check-pit-sampling.py')
if ($LASTEXITCODE -ne 0) { throw 'Random-pit sampling check failed' }
& $taskPython (Join-Path $PSScriptRoot 'compare-random-pits.py')
if ($LASTEXITCODE -ne 0) { throw 'Random-pit comparison and audit failed' }
$env:SPARKLE_TEST_OPTIONAL = '1'
$env:SPARKLE_TEST_MITSUBA_VARIANT = 'cuda_ad_rgb'
& $taskPython -m unittest discover -s (Join-Path $taskRoot 'oled-ag-sparkle\tests') -v *> (Join-Path $PSScriptRoot 'logs\validation-periodic.log')
if ($LASTEXITCODE -ne 0) { throw 'Tiled-emission regression tests failed' }
& $taskPython (Join-Path $PSScriptRoot 'finalize-random-pits.py')
if ($LASTEXITCODE -ne 0) { throw 'Random-pit report integrity or skill packaging failed' }
