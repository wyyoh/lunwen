param(
    [switch]$PrepareData
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $project ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $project "src"

function Invoke-KeyedGram {
    & $python -m keyed_gram @args
    if ($LASTEXITCODE -ne 0) {
        throw "keyed-gram failed with exit code $LASTEXITCODE"
    }
}

Push-Location $project
try {
    if ($PrepareData) {
        Invoke-KeyedGram prepare-data --mode api-sample --destination data/stories-smoke --workers 4 --requests-per-second 1.5
    }
    Invoke-KeyedGram train --config configs/smoke.yaml --output-dir artifacts/checkpoints/smoke
    Invoke-KeyedGram evaluate-localization --checkpoint artifacts/checkpoints/smoke/gram_original.pt --data-dir data/stories-smoke --core-sequences 20 --private-sequences 200 --output artifacts/results/smoke/localization.json
    $localization = Get-Content artifacts/results/smoke/localization.json -Raw | ConvertFrom-Json
    if (-not $localization.passed) {
        Write-Warning "Localization gate failed; lock and attack stages were not started."
        return
    }
    Invoke-KeyedGram keygen --checkpoint artifacts/checkpoints/smoke/gram_original.pt --output artifacts/keys/private_key.json --group-size 8 --seed 42
    Invoke-KeyedGram lock --checkpoint artifacts/checkpoints/smoke/gram_original.pt --key artifacts/keys/private_key.json --output artifacts/checkpoints/smoke/gram_locked.pt
    Invoke-KeyedGram restore --checkpoint artifacts/checkpoints/smoke/gram_locked.pt --key artifacts/keys/private_key.json --output artifacts/checkpoints/smoke/gram_unlocked.pt
    Invoke-KeyedGram evaluate --original artifacts/checkpoints/smoke/gram_original.pt --locked artifacts/checkpoints/smoke/gram_locked.pt --key artifacts/keys/private_key.json --data-dir data/stories-smoke --output-dir artifacts/results/smoke
}
finally {
    Pop-Location
}
