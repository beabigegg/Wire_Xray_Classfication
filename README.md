# Wire Loop X-ray Classification System

線弧X光影像智能分類系統 - 用於半導體封裝的線弧缺陷自動檢測與分類

---

## 📋 專案概述

本系統提供完整的線弧X光影像標註、訓練和推理功能，包括：

### 核心功能

1. **標註系統** ✅
   - 圖形化標註介面（PyQt6）
   - 視角標註（TOP/SIDE）
   - 缺陷標註（PASS/沖線/晃動/碰觸）
   - YOLO格式邊界框標註
   - SQLite數據庫存儲

2. **訓練管線** ✅
   - 數據準備和分層分割
   - **TOP/SIDE 配對感知切分** 🆕 - 避免 data leakage
   - **VIEW-aware 架構訓練** 🆕 - 分別訓練 TOP/SIDE 專屬模型
   - 三階段模型訓練（視角分類 + YOLO檢測 + 缺陷分類）
   - **暫停/恢復/取消訓練** 🆕
   - **檢查點管理與自動恢復** 🆕
   - **TensorBoard深度整合** 🆕
   - 模型版本管理與比較
   - 向後兼容舊版統一架構

3. **推理系統** ✅
   - **VIEW-aware 三階段推理** 🆕 - 視角分類 → 專屬檢測 → 專屬分類
   - **TOP/SIDE 配對推論** 🆕 - 綜合雙視角判斷
   - 單張/批次推理
   - 自動偵測並載入 VIEW-aware 或統一架構模型
   - CSV/JSON結果導出
   - CLI工具
   - 批次推理結果 GUI 查看器 🆕

---

## 🚀 快速開始

### 1. 標註數據

```bash
run_annotation.bat
```

- 打開影像文件夾
- 繪製邊界框（按 `W`）
- 選擇視角和缺陷類型
- 自動保存到數據庫

### 2. 訓練模型

#### 方法 A: GUI 訓練對話框（推薦）

從標註工具直接訓練：

1. 打開標註工具：`run_annotation.bat`
2. 點擊選單：**Training** → **Train Detection Model** (或其他模型)
3. 配置訓練參數：
   - Epochs: 訓練輪數（默認100）
   - Batch Size: 批次大小（默認16，GPU記憶體不足時減小）
   - Learning Rate: 學習率（默認0.01）
   - Device: Auto（自動選GPU/CPU）
4. 點擊 **Start Training** 開始訓練
5. 實時監控：
   - 進度條顯示訓練進度
   - 實時顯示損失值和準確率
   - 訓練日誌自動滾動
6. 點擊 **Open TensorBoard** 查看詳細圖表

**VIEW-aware 架構訓練建議順序：**
1. **視角分類器** (View Classifier) - 先訓練以分類 TOP/SIDE
2. **YOLO檢測器 - TOP** (Detection Model YOLO - TOP) - 訓練 TOP 視角專屬模型
3. **YOLO檢測器 - SIDE** (Detection Model YOLO - SIDE) - 訓練 SIDE 視角專屬模型
4. **缺陷分類器 - TOP** (Defect Classifier - TOP) - 訓練 TOP 視角專屬模型
5. **缺陷分類器 - SIDE** (Defect Classifier - SIDE) - 訓練 SIDE 視角專屬模型

**模型類型選擇：**
- 🎯 **推薦**: VIEW-aware 模型（Detection/Defect - TOP/SIDE）
- 📦 **舊版**: 統一模型（Unified）僅供向後兼容

**快捷鍵：**
- `Ctrl+Shift+D` - 訓練檢測模型
- `Ctrl+Shift+V` - 訓練視角分類器
- `Ctrl+Shift+F` - 訓練缺陷分類器
- `Ctrl+Shift+T` - 打開TensorBoard

#### 方法 B: 命令行訓練

```bash
train.bat
```

**訓練流程：**
1. 選擇 `[1]` - 數據準備和分析
2. 選擇 `[8]` - 訓練所有模型（自動依序訓練）
3. 選擇 `[9]` - 啟動TensorBoard監控

**模型訓練時間（RTX 4060 GPU）：**
- 視角分類器：15-20分鐘
- YOLO檢測器 (TOP/SIDE 各一個)：每個 1-1.5小時
- 缺陷分類器 (TOP/SIDE 各一個)：每個 30-40分鐘

**總訓練時間（VIEW-aware 5個模型）：約 3.5-5 小時**

### 3. 推理分類

訓練完成後，使用推理系統對新影像進行自動分類：

#### 方法 A: GUI 工具（推薦）

```bash
run_inference.bat
```

**GUI 功能：**
- 載入訓練好的模型
- 打開影像文件夾
- 單張或批次推理
- 實時顯示結果（邊界框、視角、缺陷）
- 查看統計信息
- 導出 CSV/JSON 結果

#### 方法 B: 命令行工具

```bash
# 單張影像推理
python infer.py --image path/to/image.png

# 批次推理（整個文件夾）
python infer.py --batch path/to/images/ --output results.csv

# JSON格式輸出
python infer.py --batch path/to/images/ --output results.json

# 使用配置文件
python infer.py --batch path/to/images/ --config inference_config.yaml --output results.csv
```

**推理速度：**
- GPU：< 500ms/張
- CPU：< 2秒/張

---

## 💻 系統需求

### 硬體需求
- **CPU**: Intel i5或更高
- **RAM**: 8GB最低，16GB推薦
- **GPU**: NVIDIA GPU with CUDA 11.8（推薦，CPU也可運行但速度慢）
- **儲存空間**: 10GB以上

### 軟體需求
- **作業系統**: Windows 10/11
- **Python**: 3.10
- **CUDA**: 11.8（如使用GPU）

---

## 📦 安裝

### 1. 安裝Conda環境

```bash
# 創建conda環境
conda create -n wire_sag python=3.10
conda activate wire_sag
```

### 2. 安裝依賴

```bash
# 安裝PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements.txt
```

### 3. 驗證安裝

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📖 使用指南

### 完整 GUI 工作流程（推薦新手）

本系統現在支持完全基於 GUI 的工作流程，無需使用命令行：

#### 1. 標註階段
```bash
run_annotation.bat
```
- 打開影像文件夾 (`Ctrl+O`)
- 繪製邊界框 (`W`)
- 選擇視角和缺陷類型
- 按 `D` 切換到下一張（自動保存）

#### 2. 訓練階段
在標註工具中直接訓練：
- **Training** → **Train Detection Model** (`Ctrl+Shift+D`)
- **Training** → **Train View Classifier** (`Ctrl+Shift+V`)
- **Training** → **Train Defect Classifier** (`Ctrl+Shift+F`)

或一次訓練全部：
- **Training** → **Train All Models** (`Ctrl+Shift+A`)

訓練對話框功能：
- ✅ 自動驗證數據充足性
- ✅ 實時進度顯示
- ✅ 損失值和準確率監控
- ✅ 預估完成時間 (ETA)
- ✅ 訓練日誌查看
- ✅ 一鍵打開 TensorBoard

#### 3. 模型管理
訓練完成後管理模型版本：
- **Training** → **Manage Models...** (`Ctrl+Shift+M`)
- 查看所有訓練好的模型
- 設置當前使用的模型版本
- 刪除舊模型
- 查看模型性能指標

#### 4. 推理階段
```bash
run_inference.bat
```
- 載入訓練好的模型
- 打開測試影像文件夾
- 單張或批次推理
- 導出結果 (CSV/JSON)

**完整流程時間估算（100張影像）：**
1. 標註：2-3 小時
2. 訓練：2-3 小時（GPU）
3. 推理：1-2 分鐘（批次處理）

---

### 標註工具 (run_annotation.bat)

**快捷鍵：**
- `Ctrl+O` - 打開文件夾
- `W` - 啟用繪圖模式
- `S` - 保存標註
- `D` - 下一張影像
- `A` - 上一張影像
- `Delete` - 刪除選中的邊界框

**工作流程：**
1. 打開包含X光影像的文件夾（1004x1004像素）
2. 按 `W` 啟用繪圖，拖曳繪製邊界框
3. 選擇視角類型（TOP/SIDE）
4. 選擇缺陷類型（PASS/沖線/晃動/碰觸）
5. 按 `D` 前往下一張影像（自動保存）

**數據存儲：**
- SQLite數據庫：`annotations.db`
- YOLO標籤：`labels/` 目錄

**完整快捷鍵列表：**

| 功能 | 快捷鍵 | 說明 |
|------|--------|------|
| **檔案操作** | | |
| 打開文件夾 | `Ctrl+O` | 選擇影像文件夾 |
| 保存 | `S` | 保存當前標註 |
| 退出 | `Ctrl+Q` | 關閉程式 |
| **標註操作** | | |
| 啟用繪圖 | `W` | 繪製邊界框模式 |
| 刪除標註 | `Delete` | 刪除選中的框 |
| **導航** | | |
| 下一張 | `D` | 下一張影像（自動保存） |
| 上一張 | `A` | 上一張影像 |
| **訓練** | | |
| 訓練檢測模型 | `Ctrl+Shift+D` | 打開檢測模型訓練對話框 |
| 訓練視角分類器 | `Ctrl+Shift+V` | 打開視角分類器訓練對話框 |
| 訓練缺陷分類器 | `Ctrl+Shift+F` | 打開缺陷分類器訓練對話框 |
| 訓練所有模型 | `Ctrl+Shift+A` | 依序訓練所有模型 |
| 打開 TensorBoard | `Ctrl+Shift+T` | 啟動 TensorBoard |
| 管理模型 | `Ctrl+Shift+M` | 打開模型管理器 |
| **推理** | | |
| 當前影像推理 | `Ctrl+I` | 對當前影像執行推理 |
| 批次推理 | `Ctrl+Shift+I` | 打開批次推理對話框 |

---

### 訓練系統 (train.bat)

**選單選項：**

**數據準備：**
- `[1]` Data Preparation and Analysis
  - 分層分割（80%訓練，20%驗證）
  - 導出YOLO格式數據集
  - 顯示類別分佈統計

**模型訓練：**
- `[5]` Train View Classifier（視角分類器）
  - 模型：ResNet18
  - 目標：準確率 > 0.90

- `[6]` Train Defect Classifier（缺陷分類器）
  - 模型：EfficientNet-B0
  - 目標：PASS召回率 > 0.70
  - 特殊處理：PASS類別20倍數據增強

- `[7]` Train YOLO Detection（檢測模型）
  - 模型：YOLOv8n
  - 目標：mAP@0.5 > 0.80

- `[8]` Train ALL Models（推薦）
  - 依序訓練所有三個模型

**工具：**
- `[2]` View Dataset Statistics
- `[3]` Run Test Suite
- `[4]` View Training History
- `[9]` Launch TensorBoard

**自定義訓練參數：**

```bash
# 視角分類器
python train_view.py --epochs 50 --batch 32 --lr 0.001

# 缺陷分類器
python train_defect.py --epochs 100 --batch 16 --lr 0.001

# YOLO檢測
python train_yolo.py --epochs 100 --batch 16 --imgsz 1004
```

**監控訓練：**

```bash
# 啟動TensorBoard
train.bat -> [9] Launch TensorBoard

# 或直接執行
tensorboard --logdir runs/ --port 6006
```

然後在瀏覽器開啟：http://localhost:6006

---

### 推理系統 (infer.py)

**基本使用：**

```bash
# 查看幫助
python infer.py --help

# 單張影像推理
python infer.py --image data/sample.png

# 批次推理
python infer.py --batch data/images/ --output results.csv

# 使用JSON格式輸出
python infer.py --batch data/images/ --output results.json --summary summary.txt
```

**進階選項：**

```bash
# 指定模型路徑
python infer.py --image data/sample.png \
  --yolo-model models/detection/best.pt \
  --view-model models/view/best.pth \
  --defect-model models/defect/best.pth

# 指定設備
python infer.py --batch data/images/ --device cuda --output results.csv

# 調整信心閾值
python infer.py --batch data/images/ --confidence 0.6 --output results.csv

# 詳細輸出
python infer.py --image data/sample.png --verbose

# 靜默模式（僅錯誤）
python infer.py --batch data/images/ --output results.csv --quiet
```

**配置文件（inference_config.yaml）：**

```yaml
models:
  detection: "models/detection/best.pt"
  view_classifier: "models/view/best.pth"
  defect_classifier: "models/defect/best.pth"

inference:
  device: "auto"                  # auto, cuda, cpu
  confidence_threshold: 0.5       # 0.0-1.0

output:
  format: "csv"                   # csv or json
```

**輸出格式：**

CSV格式：
```
image_name,bbox_x1,bbox_y1,bbox_x2,bbox_y2,view,view_confidence,defect,defect_confidence,success,error
sample.png,100,120,900,880,TOP,0.9876,PASS,0.8765,True,
```

JSON格式：
```json
[
  {
    "image_name": "sample.png",
    "success": true,
    "bbox": {"x1": 100, "y1": 120, "x2": 900, "y2": 880, "confidence": 0.95},
    "view": {"label": "TOP", "confidence": 0.9876},
    "defect": {"label": "PASS", "confidence": 0.8765},
    "processing_time": 0.456
  }
]
```

---

## 🔗 Wire TOP/SIDE 配對功能

### 功能說明

本系統支援 TOP/SIDE 配對功能，確保訓練和推論時正確處理同一顆 wire 的兩個視角：

#### 訓練階段 - 配對感知切分
- **避免 Data Leakage**: 確保同一顆 wire 的 TOP/SIDE 圖片都在同一集合（train 或 val）
- **自動啟用**: 預設在訓練 GUI 中啟用「Preserve Wire Pairs」選項
- **標籤一致性檢查**: 自動驗證 TOP/SIDE 是否有相同的缺陷分類

#### 推論階段 - 配對推論
- **綜合雙視角**: 結合 TOP/SIDE 兩個視角進行最終判斷
- **兩種策略**:
  - `Worst Case` (預設)：任一視角有缺陷即判定為缺陷
  - `Confidence`：選擇信心度較高的預測
- **GUI 支援**: 推論工具提供配對模式選擇

### 檔名格式要求

配對功能依賴正確的檔名格式：

```
{wire_id}_TOP.jpg
{wire_id}_SIDE.jpg
```

**範例**:
```
001_TOP.jpg  ←→  001_SIDE.jpg
042_TOP.jpg  ←→  042_SIDE.jpg
```

**重要**: 分類資訊存在於資料庫的 `defect_type` 欄位和資料夾路徑中，不在檔名中。

### 使用方式

#### GUI 訓練（推薦）
1. 開啟訓練對話框
2. 確認 ☑ **Preserve Wire Pairs** 已勾選（預設啟用）
3. 開始訓練

#### GUI 推論（推薦）
1. 啟動推論工具：`run_inference.bat`
2. 選擇 **Inference Mode**: `Paired TOP/SIDE (Recommended)`
3. 選擇 **Combination Strategy**: `Worst Case` (預設)
4. 執行批次推論

---

## 📊 專案結構

```
Wire_Xray_Classfication/
├── src/                          # 源代碼
│   ├── core/                     # 核心模組
│   │   ├── database.py           # 數據庫管理
│   │   ├── pairing_utils.py      # TOP/SIDE 配對工具 🆕
│   │   └── image_utils.py        # 圖像處理
│   ├── gui/                      # 圖形界面
│   │   ├── annotation_tool.py    # 標註工具
│   │   ├── inference_tool.py     # 推理GUI工具
│   │   └── canvas.py             # 繪圖畫布
│   ├── training/                 # 訓練模組
│   │   ├── data_preparation.py   # 數據準備
│   │   ├── view_classifier.py    # 視角分類器
│   │   ├── defect_classifier.py  # 缺陷分類器
│   │   ├── yolo_trainer.py       # YOLO訓練器
│   │   ├── config_manager.py     # 配置管理
│   │   ├── model_manager.py      # 模型管理
│   │   └── training_utils.py     # 訓練工具
│   └── inference/                # 推理模組
│       ├── model_loader.py       # 模型載入器
│       ├── preprocessor.py       # 圖像預處理
│       ├── inference_pipeline.py # 三階段推理管線
│       ├── batch_processor.py    # 批次處理器
│       └── result_exporter.py    # 結果導出器
├── tests/                        # 測試代碼
│   └── core/
│       └── test_pairing_utils.py # 配對工具測試 (24 tests) 🆕
├── docs/                         # 文檔
│   └── PROJECT_STATUS.md         # 專案狀態報告
├── openspec/                     # OpenSpec規範
├── run_annotation.bat            # 標註工具啟動腳本
├── run_inference.bat             # 推理GUI工具啟動腳本
├── train.bat                     # 訓練工具啟動腳本
├── train_view.py                 # 視角分類器訓練腳本
├── train_defect.py               # 缺陷分類器訓練腳本
├── train_yolo.py                 # YOLO訓練腳本
├── infer.py                      # 推理CLI工具
├── inference_config.yaml         # 推理配置文件
├── cleanup_project.bat           # 專案清理工具
├── test_pairing_simple.py        # 配對功能測試腳本 🆕
├── annotations.db                # 標註數據庫
├── config.yaml                   # 訓練配置文件
├── requirements.txt              # Python依賴
├── README.md                     # 本文件
├── USER_MANUAL.md                # 使用手冊（含配對功能說明）🆕
├── ANALYSIS_TOP_SIDE_PAIRING.md  # 配對功能技術分析 🆕
├── 訓練快速指南.md                # 訓練指南（中文）
├── 訓練腳本使用說明.md            # 詳細使用說明
└── 訓練管線實施狀態.md            # 技術實施細節
```

**訓練輸出目錄（訓練後生成）：**
```
├── runs/                         # TensorBoard日誌
│   ├── view_classifier/
│   ├── defect_classifier/
│   └── detection/
├── models/                       # 訓練好的模型
│   ├── view/
│   ├── defect/
│   └── detection/
└── training_data/                # 處理後的訓練數據
    ├── view_classifier/
    ├── defect_classifier/
    └── yolo_dataset/
```

---

## 🛠️ 技術棧

### 深度學習框架
- PyTorch 2.0.1
- TorchVision 0.15.2+cu118
- Ultralytics YOLOv8
- timm (pretrained models)

### 模型架構

#### VIEW-aware 架構 🆕 (推薦)
系統採用 **視角感知 (VIEW-aware)** 架構，針對 TOP 和 SIDE 視角訓練獨立模型，提升準確度：

**三階段推理流程：**
```
1. 視角分類 (全圖) → TOP 或 SIDE
       ↓
2. 視角專屬檢測 → 選擇 YOLO_TOP 或 YOLO_SIDE
       ↓
3. 視角專屬分類 → 選擇 Defect_TOP 或 Defect_SIDE
```

**模型清單 (5個)：**
- **視角分類器** (1個): ResNet18 - 分類 TOP/SIDE (使用全圖)
- **YOLO檢測器** (2個): YOLOv8n - 分別訓練 TOP 和 SIDE 專屬模型
- **缺陷分類器** (2個): EfficientNet-B0 - 分別訓練 TOP 和 SIDE 專屬模型

**優勢：**
- ✅ 每個模型專注於單一視角特徵，避免混淆
- ✅ TOP 和 SIDE 視角特性差異大，獨立模型更準確
- ✅ 更好的泛化能力和穩定性

#### 舊版統一架構 (向後兼容)
系統仍支援舊版統一架構（單一模型處理兩種視角）：
- **視角分類**: ResNet18 (ImageNet pretrained)
- **缺陷分類**: EfficientNet-B0 (ImageNet pretrained) - 統一模型
- **目標檢測**: YOLOv8n - 統一模型

### GUI & 數據處理
- PyQt6 (GUI)
- albumentations (data augmentation)
- Pillow (image processing)
- SQLite (database)

### 訓練監控
- TensorBoard
- Model versioning

---

## ⚙️ 配置文件 (config.yaml)

```yaml
annotation:
  view_types:
    - TOP
    - SIDE
  defect_types:
    - PASS
    - 沖線
    - 晃動
    - 碰觸
  default_annotator: unknown

gui:
  window_width: 1600
  window_height: 900
  bbox_color: '#00ff00'
  bbox_selected_color: '#ff0000'

paths:
  database: annotations.db
  data_dir: data
  labels_dir: labels
```

---

## 📚 詳細文檔

- **[USER_MANUAL.md](USER_MANUAL.md)** - 📖 完整使用手冊（含配對功能詳細說明）🆕
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - 📊 專案狀態總覽（必讀）
- **[ANALYSIS_TOP_SIDE_PAIRING.md](ANALYSIS_TOP_SIDE_PAIRING.md)** - 🔗 配對功能技術分析 🆕
- **[docs/訓練快速指南.md](docs/訓練快速指南.md)** - 訓練操作快速入門
- **[docs/訓練腳本使用說明.md](docs/訓練腳本使用說明.md)** - 詳細參數說明和故障排除
- **[docs/訓練管線實施狀態.md](docs/訓練管線實施狀態.md)** - 技術實施細節
- **[docs/推理工具使用說明.md](docs/推理工具使用說明.md)** - 推理GUI和CLI詳細說明
- **[docs/backend_api.md](docs/backend_api.md)** - 後端API文檔

---

## 🧹 專案維護

### 清理測試數據

當需要清理測試數據並準備生產訓練時：

```bash
cleanup_project.bat
選擇: [A] Clean ALL
```

**將刪除：**
- 測試訓練輸出（runs/, models/, training_data/）
- 測試腳本（test_*.py, demo_*.py）
- 臨時文件（__pycache__, *.pyc）
- 舊文檔報告

**保留：**
- 源代碼（src/）
- 標註數據庫（annotations.db）
- 訓練腳本
- YOLO預訓練模型
- 配置文件

---

## 🐛 故障排除

### GPU不可用

**問題：** `CUDA available: False`

**解決：**
```bash
# 重新安裝PyTorch with CUDA
pip uninstall torch torchvision
pip install torch==2.0.1 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### TorchVision NMS錯誤

**問題：** `Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`

**解決：** 已在訓練腳本中自動處理，確保torchvision版本匹配

### PASS召回率低

**問題：** PASS召回率 < 0.70

**解決：**
1. 收集更多PASS樣本（> 30個）
2. 增加PASS增強倍數（在train_defect.py中調整）
3. 啟用Focal Loss
4. 訓練更長時間（增加epochs）

### 訓練早停

**問題：** 訓練在很少的epoch後停止

**原因：** Early stopping機制，性能未改善時自動停止

**解決：**
- 如果性能已達標，這是正常的
- 如果性能未達標，調整學習率或增加patience參數

---

## 📈 使用範例

### 完整工作流程

```bash
# Step 1: 標註數據
run_annotation.bat
# 標註 > 100 張影像

# Step 2: 準備訓練數據
train.bat -> [1] Data Preparation

# Step 3: 訓練所有模型
train.bat -> [8] Train ALL Models
# 等待約 2-3 小時（GPU）

# Step 4: 監控訓練（另開終端）
train.bat -> [9] Launch TensorBoard

# Step 5: 查看訓練歷史
train.bat -> [4] View Training History

# Step 6: 評估模型
# 查看 TensorBoard 或 runs/ 目錄中的結果
```

---

## 🔮 下一步開發

### 推理系統 ✅ 已完成
- [x] 批次推理腳本
- [x] 單張影像推理
- [x] GUI推理工具
- [x] 結果可視化和導出

### 訓練系統增強（可選）
- [ ] GUI訓練對話框
- [ ] TensorBoard整合到GUI
- [ ] 訓練暫停/恢復功能
- [ ] 與標註系統深度整合

### 系統優化
- [ ] 模型性能優化
- [ ] 超參數自動調優
- [ ] 模型集成（Ensemble）
- [ ] 混淆矩陣可視化

### 生產部署
- [ ] Web API服務（FastAPI）
- [ ] Docker容器化
- [ ] 性能監控
- [ ] 完整文檔

---

## 📝 版本歷史

### v2.3 (Current - 2025-11-10)
- ✅ **TOP/SIDE 配對功能** 🆕
  - 配對感知的訓練資料切分（避免 data leakage）
  - 配對推論功能（綜合雙視角判斷）
  - 兩種結合策略（Worst Case / Confidence）
  - GUI 完整支援配對模式
  - 24 個單元測試全部通過

### v2.2 (2025-11-07)
- ✅ GUI推理工具（類似標註工具的介面）
- ✅ CLI推理工具
- ✅ 自動模型版本檢測
- ✅ 批次處理與結果匯出
- ✅ 修正TOP/SIDE分類順序問題

### v2.1 (2025-01-07)
- ✅ 完整訓練管線
- ✅ 三階段模型訓練
- ✅ TensorBoard監控
- ✅ 模型版本管理
- ✅ 數據增強和不平衡處理
- ✅ 專案清理工具

### v1.0 (2024)
- ✅ 標註系統
- ✅ YOLO格式支持
- ✅ SQLite數據庫

---

## 🤝 貢獻

歡迎提交Issue和Pull Request。

---

## 📞 聯絡方式

如有問題或建議，請聯繫專案維護者。

---

## 📄 授權

[待指定授權]

---

## 🙏 致謝

本專案用於半導體封裝品質檢測的線弧X光影像智能分類。
