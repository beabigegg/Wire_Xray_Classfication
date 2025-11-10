# Add Inference System

## Summary
Implement a complete inference system that enables trained models to classify Wire Loop X-ray images in production. The system provides three-stage inference (YOLO detection → View classification → Defect classification), batch processing capabilities, and result export functionality.

## Motivation
**Current State:**
- Annotation system完成 (標註172張影像)
- Training pipeline完成 (三個模型訓練完成)
- 訓練好的模型存在但無法實際使用
- 無法對新影像進行自動分類

**Problem:**
用戶已完成標註和訓練，但沒有推理功能來使用訓練好的模型進行實際生產檢測。

**Goal:**
提供完整的推理管線，讓用戶能夠：
1. 載入訓練好的三個模型
2. 對單張或批次影像進行自動分類
3. 查看推理結果和信心分數
4. 導出結果供後續分析

## Scope
**In Scope:**
1. **Model Loading** - 載入三個訓練好的模型（YOLO, View Classifier, Defect Classifier）
2. **Single Image Inference** - 對單張影像進行完整三階段推理
3. **Batch Inference** - 批次處理多張影像
4. **Result Visualization** - 顯示檢測框、分類結果和信心分數
5. **Result Export** - 導出CSV/JSON格式結果
6. **CLI Interface** - 命令行介面供批次處理
7. **Error Handling** - 處理模型載入失敗、影像格式錯誤等問題

**Out of Scope:**
- GUI推理工具（第二階段）
- Web API服務（第二階段）
- 模型性能優化（ONNX等）
- 推理結果反饋到訓練系統

## Dependencies
**Required Before This Change:**
- ✅ add-training-pipeline (已完成95/136 tasks)
- ✅ annotation-system spec (已有12 requirements)

**Blocking:**
無

**Related:**
- 未來可能的GUI inference tool
- 未來可能的Web API deployment

## Affected Systems
**New Components:**
- `src/inference/` - 推理模組
  - `model_loader.py` - 模型載入管理
  - `inference_pipeline.py` - 三階段推理管線
  - `batch_processor.py` - 批次處理
  - `result_exporter.py` - 結果導出
- `infer.py` - CLI推理腳本

**Modified Components:**
- `src/training/model_manager.py` - 可能需要共用模型載入邏輯
- `src/core/image_utils.py` - 可能需要添加推理用的圖像預處理

**Configuration:**
- 新增 `inference_config.yaml` - 推理配置（模型路徑、閾值等）

## Success Criteria
1. ✅ 能成功載入三個訓練好的模型
2. ✅ 對單張1004x1004影像完成三階段推理
3. ✅ 批次處理100+影像並導出結果
4. ✅ 推理速度 < 2秒/張（CPU）或 < 500ms/張（GPU）
5. ✅ 導出CSV包含：影像名稱、檢測框座標、視角、缺陷類型、信心分數
6. ✅ 所有推理模組有單元測試覆蓋率 > 80%

## Open Questions
1. **模型路徑配置**：是否使用固定路徑（如`models/best.pt`）還是配置文件指定？
   - 建議：配置文件指定，但提供默認路徑

2. **批次處理記憶體管理**：大批次時如何避免OOM？
   - 建議：分批次載入（chunk size可配置）

3. **錯誤處理策略**：單張影像推理失敗是否繼續處理其他影像？
   - 建議：繼續處理，記錄錯誤到日誌和結果文件

4. **信心分數閾值**：是否需要可配置的閾值來過濾低信心結果？
   - 建議：提供閾值配置，但默認不過濾（記錄所有結果）

## Risks
1. **模型載入失敗** - 模型文件不存在或格式不兼容
   - 緩解：提供清晰的錯誤訊息和模型路徑檢查

2. **記憶體不足** - 批次處理大量影像
   - 緩解：實施分批載入和處理

3. **推理速度慢** - CPU環境下推理時間過長
   - 緩解：提供進度條和ETA估算

4. **模型版本不匹配** - 訓練時的模型與推理時的代碼不兼容
   - 緩解：記錄模型版本信息並驗證兼容性
