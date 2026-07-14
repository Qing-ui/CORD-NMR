@echo off
setlocal

if "%~3"=="" (
    echo Usage:
    echo   run_unified_nmr.bat ^<mode^> ^<input_type^> ^<input_file^> [output_dir] [c_engine] [max_conformers] [max_iters] [forcefield] [coord_route]
    echo.
    echo mode:
    echo   C
    echo   H
    echo   CH
    echo.
    echo input_type:
    echo   sdf
    echo   csv
    echo.
    echo c_engine:
    echo   nmrnet
    echo   cascade2
    echo.
    echo Examples:
    echo   run_unified_nmr.bat CH sdf "molecules.sdf" "unified_out" cascade2 9 300 auto standard
    echo   run_unified_nmr.bat CH csv "molecules.csv" "unified_out" nmrnet 9 300 auto staged27
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%.."
set "BUNDLE_ROOT=%REPO_DIR%\.."
set "MODE=%~1"
set "INPUT_TYPE=%~2"
set "INPUT_FILE=%~3"

if "%~4"=="" (
    for %%I in ("%INPUT_FILE%") do set "OUTPUT_DIR=%%~dpnI_unified_nmr_out"
) else (
    set "OUTPUT_DIR=%~4"
)

if "%~5"=="" (
    set "C_ENGINE=nmrnet"
) else (
    set "C_ENGINE=%~5"
)

if "%~6"=="" (
    set "MAX_CONFORMERS=9"
) else (
    set "MAX_CONFORMERS=%~6"
)

if "%~7"=="" (
    set "MAX_ITERS=300"
) else (
    set "MAX_ITERS=%~7"
)

if "%~8"=="" (
    set "FORCEFIELD=auto"
) else (
    set "FORCEFIELD=%~8"
)

if "%~9"=="" (
    set "COORD_ROUTE=standard"
) else (
    set "COORD_ROUTE=%~9"
)

cd /d "%REPO_DIR%"
set "NMR_PREDICTOR_HOME=%BUNDLE_ROOT%"
set "NMRNET_PYTHON=%BUNDLE_ROOT%\envs\nmrnet\python.exe"
set "CASCADE2_PYTHON=%BUNDLE_ROOT%\envs\cascade2\python.exe"
"%NMRNET_PYTHON%" "%SCRIPT_DIR%predict_nmr_unified.py" --mode "%MODE%" --input-type "%INPUT_TYPE%" --input "%INPUT_FILE%" --output-dir "%OUTPUT_DIR%" --c-engine "%C_ENGINE%" --h-engine nmrnet --max-conformers %MAX_CONFORMERS% --max-iters %MAX_ITERS% --forcefield %FORCEFIELD% --coord-route %COORD_ROUTE%

echo.
echo Done.
echo Output dir: %OUTPUT_DIR%
endlocal
