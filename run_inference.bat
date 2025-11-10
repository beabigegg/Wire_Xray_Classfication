@echo off
REM Run Wire Loop X-ray Inference GUI Tool

echo ========================================
echo Wire Loop X-ray Inference Tool
echo ========================================
echo.

REM Activate conda environment
call conda activate wire_sag

REM Run inference GUI
python -m src.gui.inference_tool

pause
