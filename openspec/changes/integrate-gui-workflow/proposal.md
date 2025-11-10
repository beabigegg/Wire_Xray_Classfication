# Change: Integrate GUI Workflow

## Why
當前系統已具備完整的標註、訓練、推理三大功能，但使用者需要在三個獨立的工具間切換：
- `run_annotation.bat` - 標註工具
- `train.bat` / 訓練腳本 - 命令行訓練
- `run_inference.bat` - 推理工具
- TensorBoard 需要手動在命令行啟動

這導致工作流程不連貫，特別是在**人機協作迭代訓練流程**（標註→訓練→推理→修正→重訓練）中，使用者體驗不佳。

**剩餘的 41 個未完成任務**主要集中在三個領域：
1. **TensorBoard 整合** (8 tasks) - 訓練可視化
2. **GUI 訓練對話框** (12 tasks) - 圖形化訓練介面
3. **標註系統整合** (6 tasks) - 一站式工作流程

這三個領域的功能高度相關，應該一起實現以提供完整的 GUI 一站式工作流程。

## What Changes

### 核心變更
- **ADD** 訓練對話框 (`src/gui/training_dialog.py`) 到標註視窗
- **ADD** TensorBoard 啟動按鈕和監控介面
- **MODIFY** 標註視窗主選單，新增「訓練」和「推理」選項
- **ADD** 模型選擇對話框，支援切換不同模型版本
- **ADD** 訓練進度監控和即時指標顯示
- **ADD** 非阻塞訓練 (使用 QThread)

### 功能模組
1. **訓練對話框模組**:
   - 模型類型選擇 (Detection/View/Defect)
   - 訓練參數配置 (epochs, batch_size, learning_rate)
   - 進度條和即時指標顯示
   - 暫停/取消訓練功能
   - 訓練日誌查看器
   - 「Open TensorBoard」按鈕

2. **TensorBoard 整合模組**:
   - 從 GUI 啟動 TensorBoard 服務
   - 在系統瀏覽器中自動開啟
   - 顯示 TensorBoard URL
   - 停止 TensorBoard 服務功能

3. **標註視窗整合**:
   - 主選單新增「Training」選單
   - 主選單新增「Inference」選單
   - 模型版本管理對話框
   - 快捷鍵支援 (Ctrl+T 訓練, Ctrl+I 推理)

## Impact

### 使用者體驗
- ✅ **一站式工作流程**: 標註→訓練→推理全部在同一個 GUI 完成
- ✅ **即時反饋**: 訓練進度和指標即時顯示
- ✅ **易於使用**: 圖形化介面，無需記憶命令行參數
- ✅ **提升效率**: 減少工具切換時間

### 受影響的檔案
- **NEW**: `src/gui/training_dialog.py` - 訓練對話框
- **NEW**: `src/gui/model_selector_dialog.py` - 模型選擇對話框
- **NEW**: `src/gui/training_worker.py` - 訓練 QThread 工作器
- **NEW**: `src/gui/tensorboard_manager.py` - TensorBoard 管理器
- **MODIFY**: `src/gui/annotation_window.py` - 新增訓練/推理選單
- **NEW**: `tests/gui/test_training_dialog.py` - GUI 測試

### 外部依賴
無新增依賴，所有必要的套件已安裝：
- PyQt6 (已有)
- tensorboard (已安裝)
- 訓練器 (已實現)

## Breaking Changes
無 - 這是純新增功能，現有的命令行工具仍然保留並可用

## Risks and Mitigations

### Risk 1: GUI 訓練期間凍結
**風險**: 訓練是 CPU/GPU 密集型任務，可能導致 GUI 無響應
**緩解**:
- 使用 QThread 在背景執行訓練
- 使用 Qt 信號/槽機制更新進度
- 測試長時間訓練期間的 GUI 響應性

### Risk 2: TensorBoard 端口衝突
**風險**: TensorBoard 預設端口 (6006) 可能已被佔用
**緩解**:
- 自動檢測可用端口 (6006-6010)
- 顯示實際使用的端口號
- 提供手動指定端口選項

### Risk 3: 訓練錯誤處理
**風險**: 訓練過程中可能發生各種錯誤（資料不足、GPU OOM 等）
**緩解**:
- 完整的錯誤捕獲和使用者友善錯誤訊息
- 訓練前驗證資料集完整性
- 提供詳細的錯誤日誌

### Risk 4: 模型版本混亂
**風險**: 多個模型版本可能導致使用者困惑
**緩解**:
- 清晰的模型版本顯示 (日期、指標、是否活動)
- 模型選擇對話框顯示詳細資訊
- 活動模型高亮顯示

## Dependencies
本變更依賴於 `add-training-pipeline` 的核心功能已完成（95/136 tasks）：
- ✅ 訓練器已實現
- ✅ 資料準備已實現
- ✅ 模型管理已實現
- ✅ 資料庫支援已實現

## Success Criteria
1. ✅ 使用者可從標註視窗直接啟動訓練
2. ✅ 訓練期間 GUI 保持響應
3. ✅ 即時顯示訓練進度和指標
4. ✅ 一鍵啟動 TensorBoard 並在瀏覽器開啟
5. ✅ 可在 GUI 中選擇和切換模型版本
6. ✅ 完整的工作流程：標註→訓練→檢視 TensorBoard→推理→修正→重訓練
7. ✅ 所有 GUI 操作有適當的錯誤處理和使用者反饋
