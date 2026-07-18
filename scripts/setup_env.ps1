param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $project ".venv"

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    python -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
function Invoke-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Python -m pip install --upgrade pip
if ($CpuOnly) {
    Invoke-Python -m pip install "torch==2.12.0"
}
else {
    Invoke-Python -m pip install "torch==2.12.0" --index-url "https://download.pytorch.org/whl/cu130"
}
Invoke-Python -m pip install --editable "${project}[dev]"

Invoke-Python -c "import torch; print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'bf16': torch.cuda.is_available() and torch.cuda.is_bf16_supported()})"
