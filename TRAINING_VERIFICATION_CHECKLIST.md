# 訓練流程驗證檢查清單

## 📋 修復摘要

此文檔詳細說明了對訓練管道的重要修復，以確保所有訓練的模型都能正確保存到資料庫並可通過 GUI 管理。

---

## ✅ 已完成的修復

### 1. **創建資料庫同步工具**
- **文件**: `scripts/sync_model_database.py`
- **功能**: 掃描 `models/` 目錄並將所有模型同步到資料庫
- **使用方法**:
  ```bash
  python scripts/sync_model_database.py          # 執行同步
  python scripts/sync_model_database.py --dry-run  # 預覽變更
  ```

### 2. **修復 YOLO Trainer 資料庫保存**
- **文件**: `src/training/yolo_trainer.py` (第 222-235 行)
- **修復**: 添加 `db.save_model_version()` 調用
- **效果**: 訓練完成後自動將模型版本保存到資料庫並設為活動模型

### 3. **修復 View Classifier 資料庫保存**
- **文件**: `src/training/view_classifier.py` (第 619-640 行)
- **修復**: 添加 `db.save_model_version()` 調用
- **效果**: 訓練完成後自動將最佳模型保存到資料庫並設為活動模型

### 4. **修復 Defect Classifier 資料庫保存**
- **文件**: `src/training/defect_classifier.py` (第 813-835 行)
- **修復**: 添加 `db.save_model_version()` 調用
- **效果**: 訓練完成後自動將最佳模型保存到資料庫並設為活動模型

### 5. **擴展模型比較和選擇功能**
- **文件**:
  - `src/gui/model_selector_dialog.py` - 支持 7 個模型類型
  - `src/training/model_comparator.py` - 支持 VIEW-aware 指標
  - `src/gui/model_comparison_dialog.py` - 更新顯示名稱
  - `src/gui/annotation_window.py` - 更新選擇下拉選單
- **效果**:
  - ModelSelectorDialog 現在顯示所有 7 個模型類型（Grid 佈局）
  - ModelComparisonDialog 支持所有 VIEW-aware 模型比較

### 6. **清理環境**
- 刪除所有舊模型文件（12 個文件）
- 清空 `model_versions` 資料庫表
- 準備好進行全新訓練測試

---

## 🧪 驗證流程

### **階段 1: 訓練 View Classifier**

#### 步驟：
1. 打開 GUI：`python main.py`
2. 選擇 "Training" → "Start Training..."
3. 選擇模型類型：**"View Classifier"**
4. 配置訓練參數（建議少量 epochs 以快速測試）
5. 開始訓練

#### 預期結果：
- ✅ 訓練完成時顯示：`Model version saved to database (ID: X)`
- ✅ 模型文件存在：`models/view/v1_YYYYMMDD_HHMMSS_acc*.pt`
- ✅ JSON 元數據存在：`models/view/v1_YYYYMMDD_HHMMSS_acc*.json`

#### 驗證數據庫：
```bash
python -c "from src.core.database import Database; db = Database('annotations.db'); models = db.get_model_versions('view'); print('View models:', len(models)); [print(f'  {m[\"version\"]} - Active: {m[\"is_active\"]}') for m in models]"
```

#### 驗證 GUI：
1. 按 `Ctrl+Shift+M` 打開 "Model Version Manager"
2. 檢查 **View Classifier (TOP/SIDE)** 欄位
3. 應該看到新訓練的模型，標記為 `[ACTIVE]`

---

### **階段 2: 訓練 Detection Model (TOP View)**

#### 步驟：
1. 選擇模型類型：**"Detection Model (YOLO) - TOP"**
2. 配置 YOLO 參數
3. 開始訓練

#### 預期結果：
- ✅ 訓練完成時顯示：`Model version saved to database (ID: X)`
- ✅ 模型文件存在：`models/detection_top/v1_YYYYMMDD_HHMMSS_map*.pt`
- ✅ JSON 元數據存在：`models/detection_top/v1_YYYYMMDD_HHMMSS_map*.json`

#### 驗證數據庫：
```bash
python -c "from src.core.database import Database; db = Database('annotations.db'); models = db.get_model_versions('detection_top'); print('Detection TOP models:', len(models)); [print(f'  {m[\"version\"]} - Active: {m[\"is_active\"]}') for m in models]"
```

#### 驗證 GUI：
1. 按 `Ctrl+Shift+M` 打開 "Model Version Manager"
2. 檢查 **Detection Model (YOLO) - TOP** 欄位
3. 應該看到新訓練的模型，標記為 `[ACTIVE]`

---

### **階段 3: 訓練 Detection Model (SIDE View)**

#### 步驟：
1. 選擇模型類型：**"Detection Model (YOLO) - SIDE"**
2. 配置 YOLO 參數
3. 開始訓練

#### 預期結果：
- ✅ 訓練完成時顯示：`Model version saved to database (ID: X)`
- ✅ 模型文件存在：`models/detection_side/v1_YYYYMMDD_HHMMSS_map*.pt`
- ✅ JSON 元數據存在：`models/detection_side/v1_YYYYMMDD_HHMMSS_map*.json`

---

### **階段 4: 訓練 Defect Classifier (TOP View)**

#### 步驟：
1. 選擇模型類型：**"Defect Classifier - TOP"**
2. 配置訓練參數
3. 開始訓練

#### 預期結果：
- ✅ 訓練完成時顯示：`Model version saved to database (ID: X)`
- ✅ 模型文件存在：`models/defect_top/v1_YYYYMMDD_HHMMSS_*.pt`
- ✅ JSON 元數據存在：`models/defect_top/v1_YYYYMMDD_HHMMSS_*.json`

---

### **階段 5: 訓練 Defect Classifier (SIDE View)**

#### 步驟：
1. 選擇模型類型：**"Defect Classifier - SIDE"**
2. 配置訓練參數
3. 開始訓練

#### 預期結果：
- ✅ 訓練完成時顯示：`Model version saved to database (ID: X)`
- ✅ 模型文件存在：`models/defect_side/v1_YYYYMMDD_HHMMSS_*.pt`
- ✅ JSON 元數據存在：`models/defect_side/v1_YYYYMMDD_HHMMSS_*.json`

---

### **階段 6: 驗證 Model Selector Dialog**

#### 步驟：
1. 按 `Ctrl+Shift+M` 打開 "Model Version Manager - VIEW-aware Architecture"
2. 檢查所有 7 個模型類型欄位

#### 預期結果：
應該看到以下佈局（Grid 2x4）：

```
┌─────────────┬─────────────┬──────────────┬──────────────┐
│ View        │ Detection   │ Detection    │ Detection    │
│ Classifier  │ (Legacy)    │ TOP          │ SIDE         │
│ ✓ 1 model   │ 0 models    │ ✓ 1 model    │ ✓ 1 model    │
├─────────────┼─────────────┼──────────────┼──────────────┤
│ Defect      │ Defect      │ Defect       │              │
│ (Legacy)    │ TOP         │ SIDE         │              │
│ 0 models    │ ✓ 1 model   │ ✓ 1 model    │              │
└─────────────┴─────────────┴──────────────┴──────────────┘
```

#### 驗證功能：
1. 點擊任一模型查看詳細信息
2. 嘗試 "Set as Active" 功能
3. 嘗試 "Delete Model" 功能（建議用測試模型）
4. 使用 "Refresh List" 按鈕

---

### **階段 7: 驗證 Model Comparison Dialog**

#### 步驟：
1. 按 `Ctrl+M` 打開模型比較對話框
2. 從下拉選單選擇模型類型（應該有 7 個選項）
3. 選擇 2-4 個模型進行比較

#### 預期結果：
- ✅ 下拉選單顯示所有 7 個選項（包含 VIEW-aware 和 Legacy）
- ✅ 可以選擇同類型的多個模型
- ✅ 比較結果顯示指標差異（綠色=改進，紅色=退步）
- ✅ 顯示推薦模型及理由

---

### **階段 8: 驗證推理管道整合**

#### 步驟：
1. 確認所有 5 個必要模型都已訓練並設為活動：
   - View Classifier
   - Detection TOP
   - Detection SIDE
   - Defect TOP
   - Defect SIDE

2. 運行推理測試：
```python
from src.core.database import Database
db = Database('annotations.db')
active_models = db.get_active_models()

print("Active Models:")
for model_type, path in active_models.items():
    if path:
        print(f"  {model_type}: {path}")
```

#### 預期結果：
- ✅ `get_active_models()` 返回所有 5 個模型路徑
- ✅ 所有路徑指向實際存在的 `.pt` 文件
- ✅ 推理管道可以成功加載所有模型

---

## 📊 最終檢查清單

### 文件系統檢查
- [ ] `models/view/` 目錄包含至少 1 個 `.pt` 文件
- [ ] `models/detection_top/` 目錄包含至少 1 個 `.pt` 文件
- [ ] `models/detection_side/` 目錄包含至少 1 個 `.pt` 文件
- [ ] `models/defect_top/` 目錄包含至少 1 個 `.pt` 文件
- [ ] `models/defect_side/` 目錄包含至少 1 個 `.pt` 文件
- [ ] 每個 `.pt` 文件都有對應的 `.json` 元數據文件

### 資料庫檢查
```bash
# 驗證所有模型類型都有記錄
python -c "
from src.core.database import Database
db = Database('annotations.db')

model_types = ['view', 'detection_top', 'detection_side', 'defect_top', 'defect_side']

for mt in model_types:
    models = db.get_model_versions(mt)
    print(f'{mt}: {len(models)} models')
    active = [m for m in models if m.get('is_active')]
    if active:
        print(f'  Active: {active[0][\"version\"]}')
"
```

- [ ] `view` 有至少 1 個模型記錄
- [ ] `detection_top` 有至少 1 個模型記錄
- [ ] `detection_side` 有至少 1 個模型記錄
- [ ] `defect_top` 有至少 1 個模型記錄
- [ ] `defect_side` 有至少 1 個模型記錄
- [ ] 每個類型有且僅有 1 個 `is_active=1` 的模型

### GUI 功能檢查
- [ ] ModelSelectorDialog 顯示所有 7 個模型類型
- [ ] 可以查看每個模型的詳細信息（指標、路徑、日期）
- [ ] 可以設置活動模型
- [ ] 可以刪除模型
- [ ] ModelComparisonDialog 支持所有 7 個模型類型選擇
- [ ] 可以比較 2-4 個同類型模型
- [ ] 比較結果顯示正確的指標和差異

### 推理管道檢查
- [ ] `db.get_active_models()` 返回所有必要的模型路徑
- [ ] 所有返回的路徑指向存在的文件
- [ ] 推理管道可以成功加載並使用模型

---

## 🐛 常見問題排查

### 問題 1: 訓練完成但資料庫沒有記錄

**原因**: 訓練器代碼中的資料庫保存邏輯失敗

**檢查**:
1. 查看訓練輸出是否有 "Model version saved to database" 訊息
2. 如果看到 "Warning: Failed to save model version to database"，檢查錯誤詳情
3. 確認 `annotations.db` 文件存在且可寫入

**解決**:
- 使用同步腳本手動同步：`python scripts/sync_model_database.py`

### 問題 2: 模型文件存在但 GUI 中看不到

**原因**: 資料庫記錄缺失

**解決**:
```bash
cd d:\WORK\user_scrip\TOOL\Wire_Xray_Classfication
python scripts/sync_model_database.py
```

### 問題 3: 推理時找不到模型

**原因**:
1. 模型沒有設為活動
2. 資料庫路徑錯誤

**檢查**:
```python
from src.core.database import Database
db = Database('annotations.db')
active = db.get_active_models()
print(active)
```

**解決**:
- 使用 ModelSelectorDialog 將模型設為活動
- 確認路徑格式正確（應該是相對路徑，如 `models/view/v1_xxx.pt`）

---

## 📝 訓練建議

### 訓練順序
建議按以下順序訓練模型，因為後續模型依賴前面的結果：

1. **View Classifier** （必須先訓練）
   - 影響：Detection 和 Defect 模型的視角分類
   - 目標：Accuracy > 95%

2. **Detection TOP**（視角分類後）
   - 使用：僅 TOP 視角的圖像
   - 目標：mAP@0.5 > 80%

3. **Detection SIDE**（視角分類後）
   - 使用：僅 SIDE 視角的圖像
   - 目標：mAP@0.5 > 80%

4. **Defect TOP**（檢測模型訓練後）
   - 使用：TOP 視角的檢測框
   - 目標：Balanced Accuracy > 80%, PASS Recall > 70%

5. **Defect SIDE**（檢測模型訓練後）
   - 使用：SIDE 視角的檢測框
   - 目標：Balanced Accuracy > 80%, PASS Recall > 70%

### 訓練參數建議

#### View Classifier
```
Epochs: 20-30
Batch Size: 32
Learning Rate: 0.001
Early Stopping: patience=5
```

#### YOLO Detection
```
Epochs: 50-100
Image Size: 640
Batch Size: 16
Model: yolo11n.pt (Nano for fast testing)
```

#### Defect Classifier
```
Epochs: 30-50
Batch Size: 32
Learning Rate: 0.0001
Early Stopping: patience=10
Class Weights: Auto-balanced
```

---

## ✅ 驗證完成標準

當以下所有條件都滿足時，訓練管道修復驗證完成：

1. ✅ 訓練所有 5 個必要模型（View, Detection TOP/SIDE, Defect TOP/SIDE）
2. ✅ 每個訓練完成都顯示 "Model version saved to database"
3. ✅ ModelSelectorDialog 正確顯示所有 7 個模型類型
4. ✅ 每個模型類型都有至少 1 個模型記錄
5. ✅ 每個模型類型都有 1 個活動模型
6. ✅ ModelComparisonDialog 可以比較同類型模型
7. ✅ `db.get_active_models()` 返回所有必要模型路徑
8. ✅ 推理管道可以成功加載並運行

---

**日期**: 2025-11-13
**修復版本**: v1.0
**文檔作者**: Claude AI
