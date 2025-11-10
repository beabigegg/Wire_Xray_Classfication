# Wire Loop X-ray Classification System - 使用手冊

## 📖 目錄

1. [系統簡介](#系統簡介)
2. [系統架構與工作流程](#系統架構與工作流程)
3. [標註工具使用](#標註工具使用)
4. [模型訓練指南](#模型訓練指南)
5. [推理工具使用](#推理工具使用)
6. [常見問題](#常見問題)
7. [最佳實踐建議](#最佳實踐建議)

---

## 系統簡介

Wire Loop X-ray Classification System 是一套完整的線弧X光影像智能分類系統，專為半導體封裝的線弧缺陷自動檢測與分類設計。

### 主要功能模組

1. **標註系統** - 圖形化標註介面，支援視角、缺陷類別和YOLO邊界框標註
2. **訓練管線** - 三階段模型訓練（視角分類、缺陷分類、線弧檢測）
3. **推理系統** - GUI和CLI兩種推理模式，支援單張和批量處理

---

## 系統架構與工作流程

### 完整工作流程

```
標註階段 → 訓練階段 → 推理階段
   ↓           ↓           ↓
數據標註    模型訓練    生產應用
```

### 三階段訓練流程

1. **視角分類器** (View Classifier) - 識別 TOP/SIDE 視角
2. **缺陷分類器** (Defect Classifier) - 分類 PASS/沖線/晃動/碰觸
3. **YOLO 檢測器** (Detection) - 定位線弧位置

---

## 標註工具使用

### 啟動標註工具

```bash
# 使用 Conda 環境
conda activate wire_sag
python -m src.main --database annotations.db --annotator your_name

# 或使用完整路徑
"C:\Users\lin46\.conda\envs\wire_sag\python.exe" -m src.main --database annotations.db --annotator your_name
```

### 標註介面功能

#### 1. 視角標註
- **快捷鍵**: `T` (TOP) / `S` (SIDE)
- **使用時機**: 每張影像首先標註視角
- **注意事項**: 視角標註會影響後續缺陷分類的訓練數據準備

#### 2. 缺陷標註
- **快捷鍵**:
  - `P` - PASS (良品)
  - `1` - 沖線
  - `2` - 晃動
  - `3` - 碰觸
- **使用時機**: 標註完視角後進行缺陷分類
- **注意事項**:
  - PASS 類別應確保影像品質正常
  - 缺陷類別應明確可見缺陷特徵

#### 3. YOLO 邊界框標註
- **操作**: 滑鼠拖曳繪製矩形框
- **快捷鍵**: `Delete` 刪除選中的框
- **使用時機**: 框選所有可見的線弧
- **注意事項**:
  - 邊界框應緊密貼合線弧
  - 避免包含過多背景
  - 標註所有可見線弧，不論是否有缺陷

#### 4. 導航與存檔
- **上一張**: `←` 或 `A`
- **下一張**: `→` 或 `D`
- **自動存檔**: 切換影像時自動保存
- **進度顯示**: 底部狀態欄顯示當前進度

### 標註數據查看

```python
import sqlite3

# 查看標註統計
conn = sqlite3.connect('annotations.db')
cursor = conn.cursor()

# 視角分佈
cursor.execute("SELECT view, COUNT(*) FROM annotations WHERE view IS NOT NULL GROUP BY view")
print("視角分佈:", cursor.fetchall())

# 缺陷分佈
cursor.execute("SELECT defect, COUNT(*) FROM annotations WHERE defect IS NOT NULL GROUP BY defect")
print("缺陷分佈:", cursor.fetchall())

# YOLO標註數量
cursor.execute("SELECT COUNT(*) FROM annotations WHERE yolo_boxes != '[]'")
print("YOLO標註數量:", cursor.fetchone()[0])

conn.close()
```

---

## 模型訓練指南

### 訓練準備

#### 1. 數據準備
確保標註數據充足：
- **視角分類器**: 每個類別至少 100 張影像
- **缺陷分類器**: 每個類別至少 50 張影像（PASS 類別建議 200+ 張）
- **YOLO 檢測器**: 至少 200 張標註邊界框的影像

#### 2. 環境檢查
```bash
# 檢查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 檢查訓練數據目錄
python -c "from pathlib import Path; print(list(Path('training_data').rglob('*'))[:10])"
```

### 使用 GUI 訓練

#### 啟動訓練對話框
在標註工具選單：**Training → Start Training...**

#### 訓練配置

##### 基本參數
- **Model Type**: 選擇訓練模型類型
  - View Classifier (視角分類)
  - Defect Classifier (缺陷分類)
  - Detection (YOLO 檢測)
- **Epochs**: 訓練輪數 (建議: View/Defect 20-50, YOLO 50-100)
- **Batch Size**: 批次大小 (建議: 16-32)
- **Learning Rate**: 學習率 (建議: 0.001-0.0001)

##### 進階參數 (Advanced Options)

**YOLO 專用**:
- `imgsz`: 影像大小 (預設 640)
- `optimizer`: 優化器 (SGD/Adam/AdamW)
- `patience`: 早停耐心值 (預設 50)
- `warmup_epochs`: 預熱輪數 (預設 3)
- `conf`: 信心閾值 (預設 0.25)
- `iou`: IoU 閾值 (預設 0.7)

**View Classifier 專用**:
- `backbone`: 骨幹網路 (resnet18/resnet50/efficientnet_b0)
- `pretrained`: 是否使用預訓練權重
- `scheduler`: 學習率調度器 (step/cosine/exponential)
- `dropout`: Dropout 比率 (預設 0.5)
- `weight_decay`: 權重衰減 (預設 1e-4)

**Defect Classifier 專用**:
- `loss_function`: 損失函數 (cross_entropy/focal/weighted)
- `focal_gamma`: Focal Loss gamma 參數 (預設 2.0)
- `class_weights`: 類別權重 (auto/balanced/manual)
- `pass_augmentation`: PASS 類別數據增強倍數 (預設 2.0)
- `balanced_sampling`: 是否使用平衡採樣

#### 訓練監控

##### TensorBoard 整合
- 勾選 **"Launch TensorBoard"** 自動啟動監控
- 瀏覽器自動開啟 `http://localhost:6006`
- 實時查看:
  - 訓練/驗證損失曲線
  - 準確率變化
  - 混淆矩陣 (每 5 個 epoch)
  - 樣本預測結果 (每 10 個 epoch)
  - 參數直方圖

##### 訓練控制
- **Pause**: 暫停訓練並保存檢查點
- **Resume**: 從檢查點恢復訓練
- **Cancel**: 取消訓練（確認對話框）

#### 檢查點管理

訓練過程中自動保存檢查點：
- 位置: `checkpoints/{model_type}_checkpoint.pth`
- 包含: 模型權重、優化器狀態、調度器狀態、當前 epoch、最佳指標
- 自動恢復: 重新開啟訓練對話框時自動檢測並詢問是否恢復

### 使用命令行訓練

```bash
# 訓練視角分類器
python train_view.py --epochs 30 --batch_size 16 --learning_rate 0.001

# 訓練缺陷分類器
python train_defect.py --epochs 50 --batch_size 16 --learning_rate 0.0001 \
    --loss_function focal --pass_augmentation 2.0

# 訓練 YOLO 檢測器
python train_yolo.py --epochs 100 --batch_size 16 --imgsz 640 \
    --optimizer AdamW --patience 50
```

### 模型比較與選擇

#### 使用模型比較工具
選單：**Training → Compare Models...**

功能：
- 選擇 2-4 個模型進行比較
- 自動計算指標差異（絕對值和相對百分比）
- 顏色標示改善/退步（綠色/紅色）
- AI 推薦最佳模型並提供理由
- 查看 TensorBoard 對比
- 設置為當前活動模型

#### 評估指標

**視角分類器**:
- Accuracy (準確率)
- Per-class Precision/Recall/F1

**缺陷分類器**:
- Balanced Accuracy (平衡準確率) - 主要指標
- PASS Class Recall (PASS 召回率) - 關鍵指標
- Per-class Precision/Recall/F1
- Confusion Matrix

**YOLO 檢測器**:
- mAP@0.5 - 主要指標
- mAP@0.5:0.95
- Precision/Recall
- Inference Speed (FPS)

---

## 推理工具使用

### GUI 推理工具

#### 啟動方式
在標註工具選單：**Inference → Run Inference...**

#### 使用流程

1. **選擇模型**
   - 自動檢測 `models/` 目錄下的最新模型
   - 或手動選擇指定版本

2. **選擇影像**
   - 單張影像：點擊 "Select Image"
   - 批量處理：點擊 "Select Folder"

3. **配置參數**
   - 信心閾值 (Confidence Threshold)
   - IoU 閾值 (IoU Threshold)
   - 是否保存結果

4. **執行推理**
   - 點擊 "Run Inference"
   - 查看即時結果顯示
   - 結果自動保存到 `inference_results/`

#### 結果查看

結果目錄結構：
```
inference_results/
├── YYYYMMDD_HHMMSS/
│   ├── annotated/          # 標註後的影像
│   ├── results.json        # JSON 格式結果
│   ├── results.csv         # CSV 格式結果
│   └── summary.txt         # 文字摘要
```

### 命令行推理

```bash
# 單張影像推理
python infer.py --image path/to/image.jpg

# 批量推理
python infer.py --input_dir path/to/images/ --output_dir results/

# 指定模型版本
python infer.py --image test.jpg \
    --detection_model models/detection/yolo_v2.pt \
    --view_model models/view/view_v2.pt \
    --defect_model models/defect/defect_v2.pt

# 調整閾值
python infer.py --image test.jpg --conf_threshold 0.3 --iou_threshold 0.5
```

### 推理結果格式

#### JSON 格式
```json
{
  "image_path": "test.jpg",
  "timestamp": "2025-01-15T10:30:00",
  "view": "TOP",
  "view_confidence": 0.95,
  "overall_defect": "PASS",
  "defect_confidence": 0.88,
  "wire_loops": [
    {
      "bbox": [100, 150, 200, 250],
      "confidence": 0.92,
      "defect": "PASS",
      "defect_confidence": 0.88
    }
  ]
}
```

#### CSV 格式
```csv
image_path,view,view_conf,overall_defect,defect_conf,num_wires,inference_time_ms
test.jpg,TOP,0.95,PASS,0.88,3,45.2
```

---

## Wire TOP/SIDE 配對功能

### 配對功能概述

Wire Loop X-ray 系統使用兩個視角（TOP 和 SIDE）對同一顆 wire 進行檢測。配對功能確保：

1. **訓練階段**: 同一顆 wire 的 TOP/SIDE 圖片保持在同一集合（避免 data leakage）
2. **推論階段**: 綜合 TOP/SIDE 兩個視角進行最終判斷

### 檔名格式要求

**必須遵循的格式**:
```
{wire_id}_TOP.{ext}
{wire_id}_SIDE.{ext}
```

**有效範例**:
```
001_TOP.jpg      ←→  001_SIDE.jpg
042_TOP.jpg      ←→  042_SIDE.jpg
ABC123_TOP.png   ←→  ABC123_SIDE.png
```

**無效範例**:
```
TOP_001.jpg      # 視角在前面（錯誤）
001-TOP.jpg      # 使用 - 分隔（錯誤）
001.jpg          # 缺少視角標識（錯誤）
```

**重要說明**:
- ⚠️ **分類資訊的位置**: 分類（PASS/沖線/晃動/碰觸）**不在檔名中**
- ✅ 分類存在於：
  1. 資料庫標註（`defect_type` 欄位）
  2. 資料夾路徑（例如：`photo/Top/沖線/042_TOP.jpg`）
- ✅ 同一顆 wire 的 TOP/SIDE 應該有相同的分類

### 訓練階段 - 配對感知切分

#### 使用 GUI 訓練（推薦）

1. **開啟訓練對話框**
   - 在標註工具選單：**Training → Train Defect Classifier** (或其他模型)

2. **確認配對設定**
   - 在 **Advanced Options** 區塊
   - 確認 ☑ **Preserve Wire Pairs** 已勾選（預設啟用）

3. **開始訓練**
   - 點擊 **Start Training**
   - 觀察日誌輸出：
     ```
     Preparing datasets...
       Using wire-aware split to preserve TOP/SIDE pairs

     Stratified split with pairing:
       Total wires: 86 complete, 0 incomplete
       Train wires: 68 (136 annotations)
       Val wires: 18 (36 annotations)
     ```

#### 使用 Python 腳本

```python
from src.training.data_preparation import DataPreparator

# 初始化資料準備器
prep = DataPreparator(
    db_path='annotations.db',
    random_seed=42
)

# 使用配對感知切分準備完整訓練流程
complete_info = prep.prepare_full_pipeline(
    output_base_dir='datasets',
    val_ratio=0.2,
    stratify_by='defect_type',
    preserve_wire_pairs=True  # 啟用配對保護（預設）
)
```

#### 標籤一致性檢查

如果發現 TOP/SIDE 標籤不一致，系統會發出警告：
```
Warning: Wire 042 has mismatched labels - TOP: PASS, SIDE: 沖線
  → Will use TOP label (PASS) for stratification
```

**建議**: 檢查並修正標註錯誤，確保同一顆 wire 的 TOP/SIDE 標籤相同。

### 推論階段 - 配對推論

#### 使用 GUI 推論（推薦）

1. **啟動推論工具**
   ```bash
   run_inference.bat
   ```

2. **載入模型**
   - 點擊 **Load Models** 載入訓練好的模型

3. **選擇推論模式**
   - 在 **Inference** 區塊中
   - **Inference Mode**: 選擇 `Paired TOP/SIDE (Recommended)`

4. **選擇結合策略**
   - **Combination Strategy**:
     - `Worst Case (Fail if either defect)` - 預設，推薦用於品質控制
     - `Highest Confidence` - 選擇信心度較高的預測

5. **執行批次推論**
   - 點擊 **Open Folder** 選擇包含 TOP/SIDE 圖片的資料夾
   - 點擊 **Infer All Images**
   - 查看配對結果：
     ```
     === PAIRED INFERENCE RESULT ===

     Wire ID: 042
     Processing Time: 1.234s

     >>> FINAL DECISION <<<
     Defect: 沖線
     Confidence: 0.8500
     Decisive View: SIDE

     --- TOP View ---
     View: TOP
       Confidence: 0.9800
     Defect: PASS
       Confidence: 0.9500

     --- SIDE View ---
     View: SIDE
       Confidence: 0.9750
     Defect: 沖線
       Confidence: 0.8500
     ```

#### 使用 Python 腳本

**單組配對推論**:
```python
from src.inference.inference_pipeline import InferencePipeline
from src.inference.model_loader import ModelLoader

# 初始化模型
loader = ModelLoader(
    yolo_model_path='models/yolo_detection.pt',
    view_classifier_path='models/view_classifier.pth',
    defect_classifier_path='models/defect_classifier.pth'
)
pipeline = InferencePipeline(loader)

# 推論單組配對
result = pipeline.infer_wire_pair(
    top_image_path='photo/Top/沖線/042_TOP.jpg',
    side_image_path='photo/Side/沖線/042_SIDE.jpg',
    combination_strategy='worst_case'  # 或 'confidence'
)

# 查看結果
if result['success']:
    combined = result['combined_result']
    print(f"Wire {result['wire_id']}: {combined['defect_type']}")
    print(f"Confidence: {combined['defect_confidence']:.3f}")
    print(f"Decisive View: {combined['decisive_view']}")
```

**批次配對推論**:
```python
# 推論整個資料夾中的所有配對
results = pipeline.infer_batch_with_pairing(
    image_dir='photo/production_batch',
    combination_strategy='worst_case'
)

# 處理結果
for result in results:
    if result['success']:
        combined = result['combined_result']
        print(f"Wire {result['wire_id']}: "
              f"{combined['defect_type']} "
              f"(confidence: {combined['defect_confidence']:.3f}, "
              f"decisive_view: {combined['decisive_view']})")
```

### 結合策略說明

#### Worst Case 策略（預設，推薦）

**原則**: 任一視角有缺陷即判定為缺陷

**缺陷嚴重度排序**:
```
PASS (0) < 沖線 (1) < 晃動 (2) < 碰觸 (3)
```

**範例**:
```
TOP: PASS + SIDE: PASS → 最終: PASS
TOP: PASS + SIDE: 沖線 → 最終: 沖線 (SIDE 決定)
TOP: 晃動 + SIDE: 沖線 → 最終: 晃動 (更嚴重)
TOP: 碰觸 + SIDE: 晃動 → 最終: 碰觸 (最嚴重)
```

**適用場景**:
- ✅ 品質控制
- ✅ 降低漏檢率
- ✅ 保守的判斷策略

#### Confidence 策略

**原則**: 選擇信心度較高的預測

**範例**:
```
TOP: 沖線 (0.95) + SIDE: 晃動 (0.80) → 最終: 沖線 (TOP 決定)
TOP: PASS (0.70) + SIDE: 沖線 (0.90) → 最終: 沖線 (SIDE 決定)
```

**適用場景**:
- ⚠️ 需要更高準確度
- ⚠️ 可接受較高的漏檢率
- ⚠️ 模型信心度已經過良好校準

---

## 常見問題

### Q1: 標註工具無法啟動
**A**: 檢查以下項目：
```bash
# 檢查環境
conda activate wire_sag
python --version  # 應顯示 Python 3.8+

# 檢查依賴
python -c "import PyQt6; import torch; import torchvision; print('All imports OK')"

# 檢查數據庫
ls annotations.db  # 確認數據庫文件存在
```

### Q2: 訓練時 CUDA out of memory
**A**: 減少 batch size 或影像大小：
- 將 batch_size 從 32 降到 16 或 8
- YOLO: 將 imgsz 從 640 降到 512 或 416
- 關閉其他 GPU 占用程序

### Q3: 訓練準確率不提升
**A**: 檢查以下方面：
1. **數據品質**: 檢查標註是否正確
2. **類別平衡**: 使用 `balanced_sampling` 和 `class_weights`
3. **學習率**: 嘗試降低學習率 (例如 0.0001)
4. **數據增強**: 增加訓練數據或調整增強參數
5. **模型容量**: 嘗試更大的 backbone (resnet50/efficientnet_b4)

### Q4: PASS 類別召回率低
**A**: 缺陷分類器專用優化：
- 增加 `pass_augmentation` 倍數 (2.0 → 3.0)
- 使用 `focal` 損失函數
- 調整 `class_weights` 為 "balanced"
- 增加更多 PASS 樣本標註
- 在 TensorBoard 中監控 PASS class metrics

### Q5: 推理速度慢
**A**: 優化方法：
1. **使用 GPU**: 確保 CUDA 可用
2. **批量處理**: 使用 BatchProcessor 而非單張處理
3. **調整 batch_size**: 推理時可用更大的 batch
4. **模型優化**: 考慮模型量化或蒸餾

### Q6: TensorBoard 無法啟動
**A**: 檢查以下項目：
```bash
# 檢查 TensorBoard 安裝
python -m tensorboard --version

# 手動啟動
python -m tensorboard --logdir=runs --port=6006

# 檢查端口占用
netstat -ano | findstr :6006
```

### Q7: 訓練中斷後如何恢復
**A**:
1. 重新開啟 Training Dialog
2. 系統會自動檢測檢查點
3. 選擇 "Resume" 繼續訓練
4. 或選擇 "Discard" 重新開始

### Q8: TOP/SIDE 標籤不一致怎麼辦
**A**: 系統會發出警告並使用 TOP 視角的標註進行分層：
```
Warning: Wire 042 has mismatched labels - TOP: PASS, SIDE: 沖線
  → Will use TOP label (PASS) for stratification
```

**建議**:
- 檢查標註是否有誤
- 確保同一顆 wire 的 TOP/SIDE 標註相同
- 如果確實應該不同，請檢查 wire_id 是否正確

### Q9: 缺少配對圖片會怎樣
**A**:

**訓練階段**:
- 不完整的配對會被排除在配對組之外
- 這些圖片會作為個別樣本處理
- 日誌會顯示不完整配對的數量

**推論階段**:
- 批次推論會跳過沒有配對的圖片
- 日誌會顯示警告訊息
- 可以使用 Single Image 模式推論單張圖片

### Q10: 舊資料不符合檔名格式怎麼辦
**A**:

**解決方案**:
1. **重新命名檔案**（推薦）：使用腳本將檔名改為 `{wire_id}_TOP.jpg` 格式
2. **使用 Single Image 模式**：在推論時選擇「Single Image」模式，每張圖片獨立處理
3. **混合使用**：新資料使用配對格式，舊資料繼續使用個別處理，系統會自動區分

---

## 最佳實踐建議

### 數據標註建議

1. **標註品質優於數量**
   - 寧可少量高質量標註，也不要大量錯誤標註
   - 定期審查標註結果

2. **保持類別平衡**
   - 每個類別盡量平衡
   - 特別注意 PASS 類別數量充足

3. **邊界框標註準則**
   - 緊密貼合線弧
   - 包含完整的線弧結構
   - 避免過多背景

### 訓練流程建議

1. **循序漸進訓練**
   ```
   Step 1: 訓練視角分類器 (快速，20-30 epochs)
   Step 2: 訓練缺陷分類器 (重點，50+ epochs)
   Step 3: 訓練 YOLO 檢測器 (耗時，100+ epochs)
   ```

2. **使用 TensorBoard 監控**
   - 每次訓練都啟用 TensorBoard
   - 觀察損失曲線是否平穩下降
   - 檢查混淆矩陣找出問題類別

3. **定期保存與比較**
   - 保存不同配置的訓練結果
   - 使用模型比較工具選擇最佳模型
   - 記錄訓練參數和結果

4. **關注關鍵指標**
   - 視角分類: Accuracy > 95%
   - 缺陷分類: Balanced Accuracy > 85%, PASS Recall > 90%
   - YOLO 檢測: mAP@0.5 > 0.8

### 推理部署建議

1. **模型版本管理**
   - 使用有意義的模型命名 (例如: `yolo_v2_mAP82.pt`)
   - 保留歷史版本以便回滾
   - 記錄每個版本的訓練配置

2. **閾值調整**
   - 根據實際應用場景調整信心閾值
   - 優先保證 PASS 的召回率（避免漏檢）
   - 平衡精確率和召回率

3. **批量處理優化**
   - 大批量處理時使用命令行工具
   - 合理設置 batch_size 以平衡速度和記憶體
   - 啟用結果導出 (JSON/CSV)

### 維護與更新

1. **定期重新訓練**
   - 累積新標註數據後重新訓練
   - 定期評估模型性能
   - 更新模型以適應新的缺陷類型

2. **數據備份**
   - 定期備份 `annotations.db`
   - 備份訓練好的模型文件
   - 保存重要的訓練日誌

3. **性能監控**
   - 記錄推理速度和準確率
   - 收集誤判案例進行分析
   - 持續優化模型和閾值

---

## 技術支援

如遇到問題，請參考：
1. 項目 README.md
2. 代碼內的文檔字符串
3. TensorBoard 訓練日誌
4. 數據庫記錄 (`training_history` 表)

祝使用順利！🚀
