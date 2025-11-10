@echo off
chcp 65001 >nul
REM Wire Loop Training Pipeline

echo ============================================================
echo Wire Loop Training Pipeline
echo ============================================================
echo.

REM Clean cache
echo [1/5] Cleaning cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo       Done!
echo.

REM Set Python path
set PYTHON=C:\Users\lin46\.conda\envs\wire_sag\python.exe

REM Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo.
    pause
    exit /b 1
)

echo [2/5] Checking Python environment...
"%PYTHON%" --version
if errorlevel 1 (
    echo Python execution failed
    pause
    exit /b 1
)
echo       Done!
echo.

REM Check database exists
if not exist "annotations.db" (
    echo ERROR: annotations.db not found
    echo Please run run.bat first to create annotations
    echo.
    pause
    exit /b 1
)

echo [3/5] Checking database...
echo       annotations.db exists
echo       Done!
echo.

REM Display menu
echo ============================================================
echo Select operation:
echo ============================================================
echo.
echo === DATA PREPARATION ===
echo [1] Data Preparation and Analysis
echo     - Stratified split (80/20)
echo     - Export training datasets
echo     - Show class distribution
echo.
echo === MODEL TRAINING ===
echo [5] Train View Classifier (TOP/SIDE)
echo     - Epochs: 50, Batch: 32
echo     - Balanced data, target accuracy ^> 0.90
echo.
echo [6] Train Defect Classifier (PASS handling)
echo     - Epochs: 100, Batch: 16
echo     - CRITICAL: PASS class only 5 samples
echo     - Target PASS recall ^> 0.70
echo.
echo [7] Train YOLO Detection Model
echo     - Epochs: 100, Batch: 16
echo     - Target mAP@0.5 ^> 0.80
echo.
echo [8] Train ALL Models (Sequential)
echo     - Train all three models in sequence
echo.
echo === UTILITIES ===
echo [2] View Dataset Statistics
echo [3] Run Test Suite
echo [4] View Training History
echo [9] Launch TensorBoard
echo.
echo [0] Exit
echo.
echo ============================================================
set /p choice="Enter choice [0-9]: "
echo.

if "%choice%"=="0" (
    echo Cancelled
    exit /b 0
)

if "%choice%"=="1" (
    echo [4/5] Running data preparation...
    echo ============================================================
    echo.
    "%PYTHON%" demo_training_workflow.py
    if errorlevel 1 (
        echo.
        echo Data preparation failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="2" (
    echo [4/5] Analyzing dataset...
    echo ============================================================
    echo.
    "%PYTHON%" -m src.training.dataset_analyzer
    if errorlevel 1 (
        echo.
        echo Analysis failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="3" (
    echo [4/5] Running test suite...
    echo ============================================================
    echo.
    "%PYTHON%" -m pytest tests/training/ -v --disable-warnings
    if errorlevel 1 (
        echo.
        echo Tests failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="4" (
    echo [4/5] Viewing training history...
    echo ============================================================
    echo.
    "%PYTHON%" -c "from src.core.database import Database; db = Database('annotations.db'); history = db.get_training_history(limit=10); print('\n=== Training History ==='); print('ID  | Model Type | Status     | Start Time'); print('-' * 60); [print(f'{h[\"id\"]:3} | {h[\"model_type\"]:10} | {h[\"status\"]:10} | {h[\"start_time\"]}') for h in history] if history else print('No training records yet'); db.close()"
    if errorlevel 1 (
        echo.
        echo Query failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="5" (
    echo [4/5] Training View Classifier...
    echo ============================================================
    echo.
    "%PYTHON%" train_view.py
    if errorlevel 1 (
        echo.
        echo Training failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="6" (
    echo [4/5] Training Defect Classifier...
    echo ============================================================
    echo.
    "%PYTHON%" train_defect.py
    if errorlevel 1 (
        echo.
        echo Training failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="7" (
    echo [4/5] Training YOLO Detection Model...
    echo ============================================================
    echo.
    "%PYTHON%" train_yolo.py
    if errorlevel 1 (
        echo.
        echo Training failed!
        pause
        exit /b 1
    )
    goto success
)

if "%choice%"=="8" (
    echo [4/5] Training ALL Models...
    echo ============================================================
    echo.
    echo Step 1/3: Training View Classifier...
    "%PYTHON%" train_view.py
    if errorlevel 1 (
        echo View classifier training failed!
        pause
        exit /b 1
    )
    echo.
    echo Step 2/3: Training YOLO Detection...
    "%PYTHON%" train_yolo.py
    if errorlevel 1 (
        echo YOLO training failed!
        pause
        exit /b 1
    )
    echo.
    echo Step 3/3: Training Defect Classifier...
    "%PYTHON%" train_defect.py
    if errorlevel 1 (
        echo Defect classifier training failed!
        pause
        exit /b 1
    )
    echo.
    echo ============================================================
    echo All models trained successfully!
    echo ============================================================
    goto success
)

if "%choice%"=="9" (
    echo [4/5] Launching TensorBoard...
    echo ============================================================
    echo.
    echo TensorBoard will open at: http://localhost:6006
    echo Press Ctrl+C to stop
    echo.
    "%PYTHON%" -m tensorboard.main --logdir runs/ --port 6006
    goto success
)

echo Invalid choice: %choice%
pause
exit /b 1

:success
echo.
echo ============================================================
echo [5/5] Operation completed!
echo ============================================================
echo.
echo Next steps:
echo 1. Check training results in runs/
echo 2. Launch TensorBoard: tensorboard --logdir runs/
echo 3. Verify target metrics achieved
echo.
pause
