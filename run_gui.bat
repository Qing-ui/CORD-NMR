@echo off
setlocal
cd /d "%~dp0"

set "RUNTIME_CONFIG=%~dp0external\NMR-Predictor-Portable\runtime-paths.json"
if not exist "%RUNTIME_CONFIG%" goto :not_installed

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $p = (Get-Content -LiteralPath $env:RUNTIME_CONFIG -Raw -Encoding UTF8 | ConvertFrom-Json).application_python; if (-not $p -or -not (Test-Path -LiteralPath $p -PathType Leaf)) { Write-Error 'CORD-NMR Python is missing. Run Install-CORD-NMR.bat first.' }; & $p (Join-Path (Get-Location).Path 'gui.py'); exit $LASTEXITCODE"
exit /b %ERRORLEVEL%

:not_installed
echo CORD-NMR is not installed or its runtime configuration is missing.
echo Run Install-CORD-NMR.bat first.
pause
exit /b 1
