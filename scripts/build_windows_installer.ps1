$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path "dist\EduDaily\EduDaily.exe")) {
    & powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
}

$iscc = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $iscc = $candidate
            break
        }
    }
}

if (-not $iscc) {
    throw "Inno Setup 6 was not found. Install it, then rerun this script."
}

& $iscc packaging\windows\EduDaily.iss
Write-Host ""
Write-Host "Installer complete: dist\installer"
