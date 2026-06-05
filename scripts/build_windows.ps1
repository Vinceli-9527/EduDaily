param(
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not $SkipModelDownload -and -not (Test-Path "models\bge-small-zh-v1.5\config.json")) {
    & ".\.venv\Scripts\python.exe" scripts\download_embedding_model.py
}

if (-not (Test-Path "frontend\assets\vendor\vue.global.prod.js") -or
    -not (Test-Path "frontend\assets\vendor\marked.min.js")) {
    throw "Missing frontend vendor files. Expected frontend/assets/vendor/vue.global.prod.js and marked.min.js."
}

& ".\.venv\Scripts\pyinstaller.exe" packaging\windows\EduDaily.spec --noconfirm --clean

Write-Host ""
Write-Host "Build complete: dist\EduDaily\EduDaily.exe"
