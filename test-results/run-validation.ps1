param([switch]$IncludeChromatix)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskSkill = Join-Path $taskRoot 'oled-ag-sparkle'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:SPARKLE_TEST_OPTIONAL = if ($IncludeChromatix) { '1' } else { '0' }
$env:SPARKLE_TEST_MITSUBA_VARIANT = 'cuda_ad_rgb'

& $taskPython -m unittest discover -s (Join-Path $taskSkill 'tests') -v
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }

& $taskPython (Join-Path $taskSkill 'scripts\run_legacy_demo.py') --example quick --backend all --skip-unavailable --device cpu --out (Join-Path $PSScriptRoot 'quick-all') --extra --mitsuba-variant cuda_ad_rgb
if ($LASTEXITCODE -ne 0) { throw 'Multi-backend quick demo failed' }

& $taskPython (Join-Path $taskSkill 'scripts\run_legacy_demo.py') --example 1mm --backend all --device cpu --dry-run
if ($LASTEXITCODE -ne 0) { throw '1 mm dry run failed' }

& $taskPython (Join-Path $taskSkill 'scripts\run_legacy_demo.py') --example 1mm --emission-variant subpixel --backend all --device cpu --dry-run
if ($LASTEXITCODE -ne 0) { throw '1 mm subpixel dry run failed' }
