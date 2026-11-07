[CmdletBinding(DefaultParameterSetName = 'Full')]
param(
    [Parameter(ParameterSetName = 'ReportOnly')][switch]$ReportOnly,
    [Parameter(ParameterSetName = 'PrepareOnly')][switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Missing .venv Python. Follow the environment setup in README.md.'
}
$taskOriginalEnvironment = @{}
foreach ($taskKey in @('PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')) {
    $taskOriginalEnvironment[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process')
}
try {
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    if (-not $ReportOnly) {
        & $taskPython -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
        if ($LASTEXITCODE -ne 0) {
            $taskCudaRuntime = $env:SPARKLE_TORCH_RUNTIME
            if (-not $taskCudaRuntime) {
                # Compatibility with the original machine; fresh installs use
                # the CUDA-enabled Torch in .venv and do not enter this branch.
                $taskCudaRuntime = Join-Path ((& uv cache dir).Trim()) 'archive-v0\4WYk6cbWPUV3KB3ITqhyH'
            }
            if (Test-Path -LiteralPath (Join-Path $taskCudaRuntime 'torch\lib\torch_cuda.dll')) {
                $env:PYTHONPATH = $taskCudaRuntime
            }
            & $taskPython -c 'import sys, torch; print(torch.__version__, "CUDA:", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)'
            if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch unavailable. Install the cu130 build described in README.md.' }
        }
        foreach ($taskPitch in @('1', '.5')) {
            $taskAssetName = if ($taskPitch -eq '1') { 'oled-1mm-etched-ag' } else { 'oled-1mm-etched-ag-pitch0p5' }
            & $taskPython (Join-Path $taskRoot 'oled-ag-sparkle\scripts\prepare_etched_ag.py') --pitch-um $taskPitch --out (Join-Path $taskRoot "oled-ag-sparkle\assets\$taskAssetName")
            if ($LASTEXITCODE -ne 0) { throw "Height generation failed at pitch $taskPitch" }
        }
        $taskRunner = Join-Path $taskRoot 'oled-ag-sparkle\scripts\run_sms_reference.py'
        & $taskPython $taskRunner --stage prepare
        if ($LASTEXITCODE -ne 0) { throw 'SMS reference preparation/dry-run failed' }
        if ($PrepareOnly) { return }
        foreach ($taskStage in @('main', 'fine', 'old-na')) {
            & $taskPython $taskRunner --stage $taskStage
            if ($LASTEXITCODE -ne 0) { throw "SMS reference stage failed: $taskStage" }
        }
    }
    & $taskPython (Join-Path $PSScriptRoot 'report-sms-reference.py')
    if ($LASTEXITCODE -ne 0) { throw 'SMS reference analysis/report failed' }
} finally {
    foreach ($taskKey in $taskOriginalEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskKey, $taskOriginalEnvironment[$taskKey], 'Process')
    }
}
