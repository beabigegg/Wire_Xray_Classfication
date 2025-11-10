# Design Document: Core Annotation System

## Context
Wire Loop X-ray 分類系統的核心標註模組設計。此模組是整個專案的基礎，需要提供穩定、易用的標註介面，並支援未來擴充至多線產品。

## Goals / Non-Goals

### Goals
- ✅ 提供直覺的 GUI 標註介面供檢驗人員使用
- ✅ 支援 YOLO 格式標註（業界標準，與 YOLOv8 相容）
- ✅ 架構支援多物件偵測（為未來多線產品做準備）
- ✅ 資料可追溯（誰、何時、標註什麼）
- ✅ 離線運作（無需網路連線）

### Non-Goals
- ❌ 不實作協作標註功能（多人同時標註）
- ❌ 不實作雲端同步
- ❌ 不實作進階標註工具（多邊形、關鍵點等）
- ❌ 不實作自動標註建議（此階段為純人工標註）

## Architecture

### 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                    Annotation GUI (PyQt6)               │
│  ┌─────────────┬──────────────────┬──────────────────┐ │
│  │ Image List  │  Canvas          │  Properties      │ │
│  │             │  (Drawing Area)  │  Panel           │ │
│  └─────────────┴──────────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ ImageLoader  │  │   Database   │  │ YOLO Format  │
│              │  │   (SQLite)   │  │   Handler    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  File System     │
                  │  - Images        │
                  │  - Labels (.txt) │
                  │  - Database (.db)│
                  └──────────────────┘
```

### 資料流程

```
1. 載入影像
   User → GUI → ImageLoader → File System
                         ↓
                    Validation
                         ↓
                  Gray Conversion
                         ↓
                  Display on Canvas

2. 標註 Wire Loop
   User draws bbox → Canvas → Coordinate Capture
                                      ↓
                  User selects labels (View + Defect)
                                      ↓
              ┌────────────────────────┴────────────┐
              ▼                                     ▼
         Database.save()                    YOLO.write()
              │                                     │
              ▼                                     ▼
     annotations.db                        labels/xxx.txt
```

## Decisions

### Decision 1: PyQt6 vs Tkinter
**選擇**: PyQt6

**理由**:
- ✅ 更現代化的 UI 外觀
- ✅ 豐富的繪圖功能（QGraphicsView）
- ✅ 更好的事件處理機制
- ✅ 跨平台支援佳
- ✅ 社群資源豐富

**捨棄**: Tkinter（過於陽春，繪圖功能較弱）

### Decision 2: SQLite vs File-based Storage
**選擇**: SQLite

**理由**:
- ✅ 結構化查詢能力（按日期、類別篩選）
- ✅ 事務支援（資料一致性）
- ✅ 無需額外安裝（Python 內建）
- ✅ 支援複雜查詢（統計、報表生成）
- ✅ 可擴充性佳（未來可加入訓練記錄、模型版本等）

**同時保留**: YOLO .txt 檔案（與訓練工具相容）

### Decision 3: YOLO Format vs COCO Format
**選擇**: YOLO format

**理由**:
- ✅ 格式簡單（純文字，易讀易寫）
- ✅ 與 YOLOv8/v10 無縫整合
- ✅ 檔案小，易於版本控制
- ✅ 支援多物件標註（一行一個物件）
- ✅ 廣泛的工具支援（LabelImg 等）

**格式範例**:
```
# labels/image001.txt
# <class_id> <x_center> <y_center> <width> <height>
0 0.5 0.45 0.6 0.15
```

### Decision 4: 影像格式處理策略
**選擇**: 載入時自動轉灰階

**理由**:
- ✅ X-ray 影像雖為 24-bit RGB，但實際為灰階（R=G=B）
- ✅ 轉換為單通道可節省記憶體（3x 減少）
- ✅ 訓練時只需灰階資訊
- ✅ 顯示時仍可轉回 RGB 以相容 GUI

**實作**:
```python
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

### Decision 5: 座標儲存格式
**選擇**: 資料庫使用像素座標，YOLO 檔案使用歸一化座標

**理由**:
- ✅ 資料庫存像素座標便於查詢和顯示
- ✅ YOLO 格式需要歸一化座標（0-1）
- ✅ 轉換邏輯封裝在 YOLO handler 中

**轉換公式**:
```python
# 像素 → 歸一化
x_norm = x_pixel / image_width
y_norm = y_pixel / image_height

# 歸一化 → 像素
x_pixel = x_norm * image_width
y_pixel = y_norm * image_height
```

## Database Schema

```sql
-- 影像表
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    filepath TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 標註表
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    bbox_x REAL NOT NULL,      -- 像素座標
    bbox_y REAL NOT NULL,
    bbox_width REAL NOT NULL,
    bbox_height REAL NOT NULL,
    view_type TEXT NOT NULL,   -- TOP / SIDE
    defect_type TEXT NOT NULL, -- PASS / 沖線 / 晃動 / 碰觸
    confidence REAL DEFAULT 1.0,
    annotator TEXT,            -- 標註者 ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

-- 模型版本表（預留，未來使用）
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,  -- detection / view_classification / defect_classification
    version TEXT NOT NULL,
    filepath TEXT NOT NULL,
    metrics_json TEXT,         -- 儲存評估指標 (JSON)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引優化
CREATE INDEX idx_annotations_image_id ON annotations(image_id);
CREATE INDEX idx_annotations_view_type ON annotations(view_type);
CREATE INDEX idx_annotations_defect_type ON annotations(defect_type);
```

## Class Design

### ImageLoader
```python
class ImageLoader:
    """影像載入器"""

    @staticmethod
    def load(filepath: str) -> np.ndarray:
        """載入影像並轉換為灰階"""

    @staticmethod
    def validate(image: np.ndarray) -> bool:
        """驗證影像尺寸 (1004x1004)"""

    @staticmethod
    def get_metadata(filepath: str) -> dict:
        """取得影像中繼資料"""
```

### Database
```python
class Database:
    """資料庫操作類別"""

    def __init__(self, db_path: str):
        """初始化資料庫連線"""

    def add_image(self, filename: str, filepath: str, width: int, height: int) -> int:
        """新增影像記錄，回傳 image_id"""

    def add_annotation(self, image_id: int, bbox: dict, view: str, defect: str, annotator: str) -> int:
        """新增標註記錄"""

    def get_annotations(self, image_id: int) -> List[dict]:
        """取得指定影像的所有標註"""

    def update_annotation(self, annotation_id: int, **kwargs) -> bool:
        """更新標註"""

    def delete_annotation(self, annotation_id: int) -> bool:
        """刪除標註"""
```

### YOLOFormatHandler
```python
class YOLOFormatHandler:
    """YOLO 格式處理"""

    @staticmethod
    def read(label_file: str, image_width: int, image_height: int) -> List[dict]:
        """讀取 YOLO 標註檔案，轉換為像素座標"""

    @staticmethod
    def write(label_file: str, annotations: List[dict], image_width: int, image_height: int):
        """寫入 YOLO 標註檔案，轉換為歸一化座標"""

    @staticmethod
    def normalize(bbox: dict, image_width: int, image_height: int) -> dict:
        """座標歸一化"""

    @staticmethod
    def denormalize(bbox: dict, image_width: int, image_height: int) -> dict:
        """座標反歸一化"""
```

## GUI Design

### 視窗佈局
```
┌────────────────────────────────────────────────────────────────┐
│  Wire Loop Annotation Tool                          [_][□][X]  │
├────────────────────────────────────────────────────────────────┤
│  File  Edit  View  Help                                        │
├────────────────────────────────────────────────────────────────┤
│ 🖼️ Open  💾 Save  ❌ Delete  ⬅️ Prev  ➡️ Next                  │
├──────────────┬─────────────────────────────────┬───────────────┤
│              │                                 │               │
│  Image List  │        Canvas                   │  Properties   │
│              │                                 │               │
│ ☑️ img_001.png│       [Wire Loop Image]         │ View Type:    │
│ ☐ img_002.png│                                 │ ○ TOP         │
│ ☐ img_003.png│     [Bounding Box Overlay]      │ ○ SIDE        │
│ ☐ img_004.png│                                 │               │
│ ☐ img_005.png│                                 │ Defect Type:  │
│              │                                 │ ○ PASS        │
│ 0 / 100      │                                 │ ○ 沖線         │
│ annotated    │                                 │ ○ 晃動         │
│              │                                 │ ○ 碰觸         │
│              │                                 │               │
│              │                                 │ [Apply]       │
│              │                                 │               │
├──────────────┴─────────────────────────────────┴───────────────┤
│ Status: Ready  |  Image: 1004x1004  |  Annotations: 1         │
└────────────────────────────────────────────────────────────────┘
```

### 快捷鍵設計
- `W`: 開始繪製 Bounding Box
- `D`: 下一張影像
- `A`: 上一張影像
- `S`: 儲存當前標註
- `Delete`: 刪除選中的 Bounding Box
- `Ctrl+Z`: 復原
- `Ctrl+O`: 開啟資料夾
- `Ctrl+S`: 儲存專案
- `Escape`: 取消當前繪製

## Risks / Trade-offs

### Risk 1: GUI 效能問題
**風險**: 大量影像載入時可能造成記憶體不足

**緩解措施**:
- 使用 lazy loading（僅載入當前顯示的影像）
- 實作影像快取機制（LRU cache）
- 顯示縮圖而非完整解析度

### Risk 2: 資料不一致
**風險**: 資料庫和 YOLO 檔案可能不同步

**緩解措施**:
- 使用資料庫事務確保原子性
- 儲存時同時更新資料庫和檔案
- 提供資料驗證工具檢查一致性

### Risk 3: 標註錯誤無法復原
**風險**: 使用者誤刪標註無法復原

**緩解措施**:
- 實作 Undo/Redo 功能
- 資料庫使用軟刪除（標記為刪除而非真正刪除）
- 定期自動備份

### Trade-off: 簡單性 vs 功能豐富度
**決策**: 優先簡單易用，避免過度設計

**理由**:
- 目前僅需 Bounding Box 標註
- 檢驗人員非技術背景，介面需直覺
- 可在未來版本逐步增強功能

## Migration Plan
N/A（這是第一個版本，無需遷移）

## Open Questions
1. **Q**: 是否需要支援標註歷史記錄功能？
   **A**: 暫不實作，可在未來版本加入

2. **Q**: 是否需要多使用者權限管理？
   **A**: 暫不需要，僅記錄標註者 ID

3. **Q**: 是否需要即時統計儀表板？
   **A**: 基礎統計（已標註數量）即可，詳細分析留待未來

4. **Q**: 是否需要匯出功能（CSV/Excel）？
   **A**: 暫不實作，專注於標註功能
