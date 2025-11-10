@echo off
REM Wire Loop Annotation Tool - One-Click Launcher

echo Starting Wire Loop Annotation Tool...
echo.

REM Clean cache
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul

REM Set Python path
set PYTHON=C:\Users\lin46\.conda\envs\wire_sag\python.exe

REM Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo.
    pause
    exit /b 1
)

REM Run application
"%PYTHON%" -m src.main

pause
