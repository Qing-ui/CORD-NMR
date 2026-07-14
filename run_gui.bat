@echo off
setlocal
cd /d "%~dp0"

set "RUNTIME_CONFIG=%~dp0external\NMR-Predictor-Portable\runtime-paths.json"
if not exist "%RUNTIME_CONFIG%" goto :not_installed

for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$p = (Get-Content -LiteralPath '%RUNTIME_CONFIG%' -Raw -Encoding UTF8 | ConvertFrom-Json).application_python; if ($p) { [Console]::Write($p) }"`) do set "CORD_NMR_PYTHON=%%I"

if not defined CORD_NMR_PYTHON goto :not_installed
if not exist "%CORD_NMR_PYTHON%" goto :not_installed

"%CORD_NMR_PYTHON%" "%~dp0gui.py"
exit /b %ERRORLEVEL%

:not_installed
echo CORD-NMR is not installed or its runtime configuration is missing.
echo Run Install-CORD-NMR.bat first.
pause
exit /b 1
