# Implementation Tasks

## 1. 專案結構建立
- [x] 1.1 建立專案目錄結構 (src/, tests/, data/, models/, runs/)
- [x] 1.2 建立 requirements.txt 列出所有相依套件
- [x] 1.3 建立 README.md 專案說明文件
- [x] 1.4 建立 .gitignore 排除不必要的檔案

## 2. 影像載入模組 (src/core/image_loader.py)
- [x] 2.1 實作 ImageLoader 類別
- [x] 2.2 實作影像載入功能 (支援 PNG/JPEG/BMP)
- [x] 2.3 實作 RGB 轉灰階功能
- [x] 2.4 實作影像尺寸驗證 (1004x1004)
- [x] 2.5 實作錯誤處理 (檔案不存在、格式錯誤等)
- [x] 2.6 撰寫影像載入單元測試

## 3. 資料庫模組 (src/core/database.py)
- [x] 3.1 設計 SQLite 資料庫 schema (images, annotations, models 表)
- [x] 3.2 實作 Database 類別
- [x] 3.3 實作影像記錄 CRUD 操作
- [x] 3.4 實作標註記錄 CRUD 操作
- [x] 3.5 實作查詢功能 (按日期、視角、缺陷類型)
- [x] 3.6 撰寫資料庫操作單元測試

## 4. YOLO 格式處理 (src/core/yolo_format.py)
- [x] 4.1 實作 YOLO 格式解析器
- [x] 4.2 實作 YOLO 格式寫入器
- [x] 4.3 實作座標轉換 (像素座標 <-> 歸一化座標)
- [x] 4.4 實作多物件支援 (為未來多線產品做準備)
- [x] 4.5 撰寫 YOLO 格式轉換單元測試

## 5. 配置管理 (src/core/config.py)
- [x] 5.1 建立 config.yaml 配置檔案
- [x] 5.2 實作配置讀取類別
- [x] 5.3 定義預設配置值
- [x] 5.4 實作配置驗證

## 6. GUI 繪圖畫布 (src/gui/canvas.py)
- [x] 6.1 實作 AnnotationCanvas 繼承 QWidget
- [x] 6.2 實作影像顯示功能
- [x] 6.3 實作滑鼠事件處理 (繪製 Bounding Box)
- [x] 6.4 實作 Bounding Box 視覺化 (矩形框 + 標籤)
- [x] 6.5 實作 Bounding Box 編輯功能 (拖曳、調整大小)
- [x] 6.6 實作縮放和平移功能

## 7. GUI 主視窗 (src/gui/annotation_window.py)
- [x] 7.1 實作 AnnotationWindow 主視窗類別
- [x] 7.2 設計 UI 佈局 (左側影像列表、中央畫布、右側屬性面板)
- [x] 7.3 實作影像列表 (QListWidget)
- [x] 7.4 實作屬性面板 (視角/缺陷下拉選單)
- [x] 7.5 實作工具列 (開啟資料夾、儲存、刪除等按鈕)
- [x] 7.6 實作快捷鍵 (W: 繪製, D: 下一張, A: 上一張, S: 儲存)
- [x] 7.7 整合 ImageLoader 和 Database
- [x] 7.8 實作標註儲存邏輯 (同時寫入資料庫和 YOLO 檔案)

## 8. 應用程式進入點 (src/main.py)
- [x] 8.1 建立 main.py 應用程式啟動檔案
- [x] 8.2 實作命令列參數解析
- [x] 8.3 實作資料庫初始化
- [x] 8.4 啟動 PyQt6 應用程式

## 9. 單元測試
- [x] 9.1 建立測試資料夾和測試影像 fixtures
- [x] 9.2 撰寫 test_image_loader.py
- [x] 9.3 撰寫 test_database.py
- [x] 9.4 撰寫 test_yolo_format.py
- [x] 9.5 執行所有測試並確保通過
- [x] 9.6 確保測試覆蓋率 > 80%

## 10. 文件撰寫
- [x] 10.1 撰寫 README.md (專案說明、安裝步驟、使用方法)
- [x] 10.2 撰寫 INSTALL.md (詳細安裝指引) - 合併至 README.md
- [x] 10.3 撰寫 USER_GUIDE.md (使用者操作手冊) - 合併至 README.md
- [x] 10.4 在程式碼中加入 docstrings

## 11. 整合測試與驗收
- [ ] 11.1 手動測試完整標註流程
- [ ] 11.2 測試錯誤處理 (無效影像、錯誤座標等)
- [ ] 11.3 測試資料庫正確性 (標註可正確儲存和讀取)
- [ ] 11.4 測試 YOLO 檔案格式正確性
- [ ] 11.5 驗收測試通過
