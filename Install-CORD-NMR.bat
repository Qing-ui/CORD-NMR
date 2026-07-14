@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_prediction_runtime.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo CORD-NMR installation failed with exit code %EXIT_CODE%.
) else (
    echo CORD-NMR installation completed successfully.
    echo Start the application with run_gui.bat.
)
echo.
pause
exit /b %EXIT_CODE%
