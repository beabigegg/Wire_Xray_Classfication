# Change: Add Core Annotation System

## Why
Wire Loop X-ray 分類系統需要一個完整的標註系統，作為整個 AI 訓練流程的基礎。此系統需要支援人工標註 Wire Loop 的位置、視角 (TOP/SIDE) 和缺陷類型 (PASS/沖線/晃動/碰觸)，並將標註資料儲存至資料庫以供後續模型訓練使用。

這是專案的第一個核心功能，沒有此系統就無法進行資料收集和模型訓練。

## What Changes
- **影像載入模組**: 支援載入 1004x1004, 24-bit RGB X-ray 影像，自動轉換為灰階
- **資料庫設計**: 使用 SQLite 儲存影像中繼資料、標註資料、模型版本等資訊
- **YOLO 格式支援**: 讀取和寫入 YOLO format (.txt) 標註檔案
- **標註 GUI 介面**: PyQt6 桌面應用程式，支援:
  - 影像瀏覽和縮放
  - Bounding Box 繪製和編輯
  - 視角標籤選擇 (TOP/SIDE)
  - 缺陷標籤選擇 (PASS/沖線/晃動/碰觸)
  - 快捷鍵操作
  - 標註儲存和載入
- **資料驗證**: 驗證影像格式、尺寸、標註座標合法性
- **單元測試**: 涵蓋影像載入、資料庫操作、YOLO 格式轉換等核心功能

## Impact
- **Affected specs**: `annotation-system` (新增功能模組)
- **Affected code**:
  - 新增 `src/core/image_loader.py` - 影像載入
  - 新增 `src/core/database.py` - 資料庫操作
  - 新增 `src/core/yolo_format.py` - YOLO 格式處理
  - 新增 `src/gui/annotation_window.py` - 主視窗
  - 新增 `src/gui/canvas.py` - 繪圖畫布
  - 新增 `tests/` - 測試檔案
- **Dependencies**:
  - PyQt6 (GUI)
  - OpenCV (影像處理)
  - Pillow (影像 I/O)
  - SQLite (內建，無需額外安裝)
- **Testing**: 單元測試 + 手動 GUI 測試
- **Documentation**: README.md 包含安裝和使用說明
