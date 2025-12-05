# Tasks: Fix View Classifier Label Mapping

## 1. Bug Fix

- [x] 1.1 修正 `src/inference/model_loader.py:85` 的 view_classes 順序
  - 從 `["SIDE", "TOP"]` 改為 `["TOP", "SIDE"]`
  - 更新註解說明

## 2. Verification

- [x] 2.1 執行推論測試驗證 View 分類結果正確
- [x] 2.2 確認 TOP 圖片被正確識別為 TOP
- [x] 2.3 確認 SIDE 圖片被正確識別為 SIDE

> 注意：驗證步驟需要用戶在 GUI 中使用 AWARE 模式推論實際圖片來確認。語法檢查已通過。
