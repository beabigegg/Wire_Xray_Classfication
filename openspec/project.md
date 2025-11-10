# Project Context

## Purpose
Wire Loop X-ray Classification 是一個用於半導體封裝製程的桌面 GUI 應用程式，專注於 **Wire Loop** (線弧) 的自動化缺陷檢測。系統採用 PyTorch 深度學習模型，透過**人機協作的迭代訓練流程**，最終實現自動化的 Wire Loop 偵測、視角判斷及缺陷分類。

**注意**: 本系統檢測的是 **Wire Loop** (兩個打點之間的線弧形狀)，而非 Wire Bond 的打點 (Ball Bond / Stitch)。

### 核心工作流程
1. **初始人工標註階段**: 檢驗人員手動標註 Wire Loop 的位置、視角 (TOP/SIDE) 及缺陷分類 (PASS/沖線/晃動/碰觸)
2. **模型訓練階段**: 使用標註資料訓練 PyTorch 模型
3. **半自動標註階段**: 模型自動進行 Wire Loop 偵測、視角判斷和缺陷分類，人員審查並修正
4. **持續改進**: 將修正後的資料回饋訓練，提升模型準確度
5. **最終目標**: 產生可全自動執行 Wire Loop 偵測、視角分類 (TOP/SIDE)、缺陷判斷 (PASS/沖線/晃動/碰觸) 的 AI 系統

## Tech Stack
- **Operating System**: Windows 10/11 (主要) 或 Docker 容器化開發
- **Language**: Python 3.10+
- **GUI Framework**: PyQt6 / PySide6
- **ML Framework**: PyTorch (支援物件偵測、分類)
- **Computer Vision**: OpenCV, Pillow, albumentations (資料增強)
- **標註工具整合**: LabelImg / CVAT 格式相容 (YOLO, COCO)
- **Data Storage**: SQLite (標註記錄、訓練歷史、模型版本)
- **Configuration**: YAML 或 JSON
- **Packaging**: PyInstaller (Windows .exe) 或 Docker Image
- **Containerization** (可選): Docker + Docker Compose (用於開發環境隔離)

## Project Conventions

### Code Style
- Follow PEP 8 for Python code formatting
- Use type hints for function signatures
- Use descriptive variable names (e.g., `xray_image_path` instead of `img`)
- Maximum line length: 100 characters
- Use docstrings for all public classes and functions (Google style)
- Naming conventions:
  - Classes: `PascalCase` (e.g., `WireClassifier`, `XrayImageLoader`)
  - Functions/methods: `snake_case` (e.g., `load_model`, `preprocess_image`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_IMAGE_SIZE`, `MODEL_PATH`)
  - Private methods: `_leading_underscore` (e.g., `_validate_input`)

### Architecture Patterns
- **Model-View-Controller (MVC)** separation:
  - Model: PyTorch models, data handling, business logic
  - View: PyQt6 UI components
  - Controller: Event handlers, user interaction logic
- **Modular design**: Separate modules for:
  - Image preprocessing pipeline
  - Model inference engine
  - Database operations
  - UI components
- **Configuration-driven**: Model paths, thresholds, and settings in config files
- **Plugin architecture** for future model additions (optional)

### Testing Strategy
- **Unit tests** for core logic (preprocessing, data validation)
- **Integration tests** for model inference pipeline
- **GUI tests** for critical user workflows (using pytest-qt)
- **Test data**: Include sample X-ray images in `tests/fixtures/`
- **Mocking**: Mock ML models for faster unit tests
- **Coverage target**: Aim for >80% code coverage for non-GUI code
- **Test naming**: `test_<function_name>_<scenario>` (e.g., `test_load_image_invalid_path`)

### Git Workflow
- **Branching strategy**: GitHub Flow (feature branches)
  - `main` branch: stable, deployable code
  - Feature branches: `feature/<change-id>` or `fix/<issue>`
- **Commit conventions**: Conventional Commits format
  - `feat:` new features
  - `fix:` bug fixes
  - `refactor:` code refactoring
  - `docs:` documentation changes
  - `test:` test additions/modifications
  - Example: `feat: add batch processing for X-ray images`
- **Pull requests**: Required for merging to main
- **Tags**: Version tags on main (e.g., `v1.0.0`, `v1.1.0`)

## Domain Context

### 半導體封裝 Wire Loop 特性
- **應用場景**: IC 封裝製程中的 Wire Loop (線弧) 品質檢驗
- **檢驗對象**: Wire Loop - 兩個 Bond 打點之間的線弧形狀 (金線/銅線)
- **X-ray 影像規格**:
  - **格式**: PNG, JPEG, BMP (來自 X-ray 設備)
  - **解析度**: 1004 x 1004 像素 (固定)
  - **DPI**: 96 dpi
  - **位元深度**: 24-bit RGB (儲存格式)
  - **實際內容**: 灰階影像 (R=G=B)，轉換時可直接取單一通道

- **影像特徵** (基於實際樣本):
  - Wire Loop 呈現較暗色調，與背景有良好對比
  - 典型場景: 兩個矩形封裝體 + 中間的 Wire Loop
  - Wire Loop 線條較細，需要精確的物件偵測
  - 背景相對乾淨，但可能有些微紋理或雜訊

- **產品類型**:
  - **目前**: 單線產品 (一張影像一條 Wire Loop)
  - **未來**: 多線產品 (一張影像多條 Wire Loop 同時出現)
  - **設計考量**: 系統架構需支援多物件偵測擴充性

### Wire Loop 視角分類 (View Angle) - 2 類
系統需要識別兩種拍攝視角:
- **TOP** (俯視): 從上方垂直拍攝，觀察 Wire Loop 的平面投影
- **SIDE** (側視): 從側面角度拍攝，觀察 Wire Loop 的高度和弧度

### Wire Loop 缺陷分類 (Defect Types) - 4 類
需要檢測的 Wire Loop 缺陷類型 (固定 4 類):
1. **PASS** (良品): Wire Loop 形狀正常，無缺陷
2. **沖線**: Wire Loop 與其他元件或線路發生干涉/衝突
3. **晃動**: Wire Loop 形狀不穩定或位置偏移
4. **碰觸**: Wire Loop 接觸到不應接觸的表面或元件

### 標註資料格式
- **Wire Loop 定位**: Bounding Box (x, y, width, height)
- **視角標籤**: 單一類別標籤 (TOP 或 SIDE)
- **缺陷標籤**: 單一類別標籤 (PASS / 沖線 / 晃動 / 碰觸)
- **儲存格式**: YOLO format (.txt) 或 COCO format (.json)
- **標註中繼資料**: 標註者 ID、標註時間、信心度
- **影像尺寸**: 固定 1004x1004 像素 (無需 resize 驗證)

### 人機協作工作流程

#### 階段 1: 初始人工標註
1. 載入 X-ray 影像 (1004x1004, 24-bit)
2. 人員使用標註工具框選 Wire Loop 位置 (bounding box)
3. 為每個 Wire Loop 標記:
   - 視角: TOP 或 SIDE
   - 缺陷類型: PASS / 沖線 / 晃動 / 碰觸
4. 儲存標註資料至資料庫

#### 階段 2: 模型訓練
1. 從資料庫載入標註資料集
2. 資料前處理 (24-bit RGB 轉灰階或直接使用) 和增強 (data augmentation)
3. 訓練三個子模型:
   - **Wire Loop Detection**: 偵測 Wire Loop 位置 (YOLOv8/v10)
   - **View Classification**: 分類視角 TOP/SIDE (2 類)
   - **Defect Classification**: 分類缺陷 PASS/沖線/晃動/碰觸 (4 類)
4. 評估模型效能 (mAP, accuracy, precision, recall)
5. 儲存模型檢查點和訓練記錄

#### 階段 3: 半自動標註
1. 載入訓練好的三個模型
2. 模型自動推論:
   - 偵測 Wire Loop 位置 (bounding box)
   - 判斷視角 (TOP/SIDE)
   - 分類缺陷 (PASS/沖線/晃動/碰觸)
3. 在 GUI 顯示預測結果 (含信心分數)
4. 人員審查並修正錯誤標註
5. 將修正後的資料存回資料庫
6. 累積更多標註資料

#### 階段 4: 迭代改進
1. 使用修正後的資料重新訓練模型
2. 比較新舊模型效能
3. 部署效能更佳的模型版本
4. 重複半自動標註流程

#### 階段 5: 全自動推論 (最終目標)
1. 批次載入待檢驗影像 (1004x1004, 24-bit)
2. 自動執行:
   - Wire Loop 偵測
   - 視角判斷 (TOP/SIDE)
   - 缺陷分類 (PASS/沖線/晃動/碰觸)
3. 產生檢驗報告 (含統計圖表)
4. 匯出結果 (CSV/Excel)

### 效能需求
- **影像規格**: 1004x1004 像素, 96 dpi, 24-bit RGB
- **推論速度**: <500ms per image (GPU) / <2s (CPU)
- **偵測準確度**: mAP@0.5 > 0.9 (Wire Loop detection)
- **視角分類準確度**: Accuracy > 0.95 (TOP/SIDE 二分類)
- **缺陷分類準確度**: Accuracy > 0.95 (PASS/沖線/晃動/碰觸 四分類)
- **批次處理**: 支援 100+ 影像批次推論
- **模型大小**: <500MB (含三個子模型)

## Important Constraints

### Technical Constraints
- **Operating System**:
  - **主要運行環境**: Windows 10/11
  - **開發環境**: Windows 10/11 或 Docker 容器
- **Offline operation**: 必須支援離線運作 (無網路連線環境)
- **GPU optional**: 必須能優雅降級至 CPU (若無 CUDA 可用)
- **Model compatibility**: 支援 PyTorch 2.x 模型格式
- **Resource limits**: 目標機器可能僅有 8GB RAM, 有限的 GPU 記憶體
- **Image specs**: 固定處理 1004x1004, 24-bit RGB 影像

### Regulatory Constraints
- **資料隱私**: X-ray 影像包含半導體製程機密資訊，不得外流
- **稽核軌跡**: 記錄所有標註和分類決策 (含時間戳記、操作者、模型版本)
- **模型版本控制**: 追蹤每次推論使用的模型版本及訓練資料集版本
- **可追溯性**: 支援從最終結果回溯到原始影像和標註資料

### Business Constraints
- **Easy deployment**:
  - Windows: 單一執行檔安裝程式 (PyInstaller .exe)
  - Docker: 提供 Docker Image 供容器化部署
- **Minimal configuration**: 自動偵測硬體能力 (GPU/CPU)
- **Export formats**: CSV 或 Excel 匯出檢驗報表
- **User-friendly**: 非技術人員也能操作標註和審查介面

## External Dependencies

### Machine Learning
- **PyTorch**: 核心深度學習框架 (v2.0+)
- **torchvision**: 影像轉換及預訓練模型
- **YOLOv8 / YOLOv10** (Ultralytics): Wire Loop 物件偵測（原生支援多物件）
- **timm** (PyTorch Image Models): 視角和缺陷分類的預訓練模型
- **albumentations**: 資料增強庫
- **TensorBoard**: 訓練過程即時監控（必備）
- **Optuna** (可選): 自動超參數調優（效能優化階段）
- **ONNX** (可選): 模型互通性和加速推論

### Image Processing
- **OpenCV**: 影像前處理、增強、視覺化
- **Pillow (PIL)**: 影像 I/O 支援
- **scikit-image**: 進階影像處理演算法

### GUI and Utilities
- **PyQt6**: 桌面 GUI 框架
- **matplotlib**: 結果視覺化、信心分數圖表、訓練曲線
- **pandas**: 資料操作、統計分析、報表生成
- **openpyxl**: Excel 匯出功能
- **pyqtgraph** (可選): 高效能即時圖表顯示

### Annotation Tools
- **LabelImg** (推薦): 輕量 Bounding Box 標註工具
  - YOLO 格式原生支援
  - 離線使用，符合資料隱私
  - Windows 相容性佳
  - 鍵盤快捷鍵提升效率
- **CVAT** (可選): 進階協作標註平台（若需多人標註）

### Annotation Format (推薦)
- **格式**: YOLO format (.txt)
- **座標**: 歸一化 (0-1) 中心座標 + 寬高
- **範例**: `<class_id> <x_center> <y_center> <width> <height>`
- **優勢**: 與 YOLOv8/v10 無縫整合，支援多物件標註

### Database
- **SQLite**: 嵌入式資料庫，儲存:
  - 影像中繼資料 (路徑、尺寸、拍攝參數)
  - 標註資料 (wire 座標、視角、缺陷類型、標註者)
  - 訓練記錄 (超參數、損失曲線、評估指標)
  - 模型版本 (檔案路徑、訓練時間、效能指標)
  - 推論記錄 (影像 ID、模型版本、預測結果、信心分數)

### Deployment
- **PyInstaller**: Package Python app as standalone .exe
- **NSIS** (optional): Create Windows installer

### Development Tools
- **pytest**: Testing framework
- **pytest-qt**: PyQt testing plugin
- **black**: Code formatter
- **mypy**: Static type checker
- **flake8**: Linter

### Deployment Options
1. **Windows Native**:
   - **PyInstaller**: 打包為單一 .exe 執行檔
   - **NSIS** (可選): 建立 Windows 安裝程式

2. **Docker Container**:
   - **Docker**: 容器化開發和部署環境
   - **Docker Compose**: 多服務編排 (若需資料庫、Web API 等)
   - **Base Image**: python:3.10-slim 或 pytorch/pytorch:2.0-cuda11.7
