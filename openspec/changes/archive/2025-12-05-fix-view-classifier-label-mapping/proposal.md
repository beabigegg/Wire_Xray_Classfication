# Change: Fix View Classifier Label Mapping Inversion Bug

## Why

View Side 推論結果完全顛倒（正視被識別為俯視，俯視被識別為正視）。這是因為訓練與推論時的類別索引映射不一致：
- 訓練：`['TOP', 'SIDE']` → TOP=0, SIDE=1
- 推論：`['SIDE', 'TOP']` → SIDE=0, TOP=1

## What Changes

- **BUG FIX**: 修正 `model_loader.py` 中的 `view_classes` 順序，使其與訓練時一致
- 從 `["SIDE", "TOP"]` 改為 `["TOP", "SIDE"]`
- 更新註解以正確說明類別索引

## Impact

- Affected specs: inference-pipeline
- Affected code: `src/inference/model_loader.py:85`
- 修復後所有 AWARE 模式的 View 分類將正確運作
