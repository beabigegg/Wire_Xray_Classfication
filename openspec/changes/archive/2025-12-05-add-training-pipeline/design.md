# Training Pipeline Design

## Context
We have 172 annotated Wire Loop samples ready for training. The goal is to build a 3-stage inference pipeline (detection → view classification → defect classification) through separate model training. The key challenge is severe class imbalance, particularly the PASS class with only 6 samples (3.5% of total).

**Data Distribution Analysis:**
- Total annotations: 172
- View types: TOP (86), SIDE (86) - perfectly balanced
- Defect types: 晃動 (58), 碰觸 (68), 沖線 (40), PASS (6) - imbalanced
- BBox dimensions: avg 206x78px, min 88x30px, max 406x158px
- Image size: Fixed 1004x1004, 24-bit RGB

**Constraints:**
- Offline operation required (no external data sources)
- GPU/CPU fallback support
- Model versioning for iterative improvement
- Integration with existing PyQt6 GUI
- Conda environment: wire_sag

## Goals / Non-Goals

### Goals
1. Train YOLOv8 for Wire Loop detection with mAP@0.5 > 0.85
2. Train view classifier with accuracy > 0.95 (balanced classes)
3. Train defect classifier with balanced accuracy considering class imbalance
4. Implement robust augmentation strategy for PASS class (6 samples)
5. Provide TensorBoard monitoring for all training runs
6. Store model versions with training metadata
7. Enable retraining with incremental data from annotation GUI

### Non-Goals
- Automated hyperparameter tuning (manual tuning acceptable for first version)
- Model ensemble methods (single models sufficient)
- Transfer learning from external datasets (use pretrained backbones only)
- Real-time inference during training
- Distributed training (single GPU/CPU sufficient)

## Decisions

### Decision 1: Three-Stage Model Architecture
**Choice**: Train three separate models instead of a unified multi-task model
- **Detection Model**: YOLOv8n (nano) - fast, accurate for single object per image
- **View Classifier**: ResNet18 - lightweight, proven for binary classification
- **Defect Classifier**: EfficientNet-B0 - better for imbalanced multi-class tasks

**Rationale**:
- Modularity: Each model can be trained/updated independently
- Flexibility: Different augmentation strategies per task
- Debugging: Easier to identify which stage fails
- Performance: Task-specific optimization possible

**Alternatives considered**:
- Multi-task YOLO with classification heads - rejected due to complexity and less flexibility for class imbalance handling

### Decision 2: Class Imbalance Strategy
**Choice**: Multi-pronged approach for PASS class (6 samples)
1. **Heavy augmentation**: 20x augmentation factor for PASS class only
   - Random rotation (±15°)
   - Random brightness/contrast (±20%)
   - Random Gaussian noise
   - Random horizontal flip
   - Elastic transforms
2. **Weighted Cross-Entropy Loss**: Class weights inversely proportional to frequency
   - PASS weight: 11.33 (68/6)
   - 沖線 weight: 1.70 (68/40)
   - 晃動 weight: 1.17 (68/58)
   - 碰觸 weight: 1.00 (baseline)
3. **Focal Loss Option**: For severe cases, use focal loss (γ=2) to focus on hard examples
4. **Stratified Sampling**: Ensure all classes represented in each training batch

**Rationale**:
- Single technique insufficient for 11:1 class imbalance
- Augmentation increases effective PASS samples from 6 to ~120
- Weighted loss ensures model pays attention to rare class
- Focal loss prevents easy majority class examples from dominating

**Alternatives considered**:
- SMOTE (synthetic minority oversampling) - rejected because augmentation is more natural for images
- Undersampling majority class - rejected due to already small dataset

### Decision 3: Train/Val Split Strategy
**Choice**: Stratified split maintaining class distribution
- Split ratio: 80% train, 20% val
- Stratification: Per defect type (maintains view balance automatically)
- Expected split:
  - PASS: 5 train, 1 val (after augmentation: ~100 train, ~20 val)
  - 沖線: 32 train, 8 val
  - 晃動: 46 train, 12 val
  - 碰觸: 54 train, 14 val

**Rationale**:
- Standard 80/20 split widely accepted
- Stratification ensures validation set has all classes
- Small validation set acceptable given total dataset size

**Alternatives considered**:
- K-fold cross-validation - rejected due to longer training time
- 70/30 split - rejected to maximize training data

### Decision 4: Data Augmentation Library
**Choice**: Use `albumentations` library
- Provides advanced augmentation techniques
- Optimized for performance
- Compatible with PyTorch
- Supports bounding box transformations for YOLO training

**Alternatives considered**:
- torchvision.transforms - rejected due to limited augmentation options
- imgaug - rejected due to maintenance concerns

### Decision 5: Training Configuration Management
**Choice**: YAML configuration files
```yaml
yolo_detection:
  model: yolov8n.pt
  epochs: 100
  batch_size: 16
  imgsz: 1004
  patience: 20

view_classifier:
  backbone: resnet18
  epochs: 50
  batch_size: 32
  lr: 0.001

defect_classifier:
  backbone: efficientnet_b0
  epochs: 100
  batch_size: 16
  lr: 0.001
  loss: weighted_focal  # or 'weighted_ce'
  class_weights: [11.33, 1.70, 1.17, 1.00]
```

**Rationale**:
- Human-readable and editable
- Separates configuration from code
- Easy to version control
- Standard format in ML projects

### Decision 6: Model Versioning Strategy
**Choice**: Timestamp + metrics-based versioning
- Format: `{model_type}_{YYYYMMDD_HHMMSS}_{metric}.pt`
- Example: `yolo_detect_20251106_143022_mAP0.87.pt`
- Store in `models/` directory with metadata in database

Database schema:
```sql
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY,
    model_type TEXT,  -- 'detection', 'view', 'defect'
    filepath TEXT,
    training_date TEXT,
    metrics JSON,  -- {mAP: 0.87, accuracy: 0.95, ...}
    training_config JSON,
    dataset_size INTEGER,
    is_active BOOLEAN
);
```

**Rationale**:
- Timestamp ensures uniqueness
- Metrics in filename allows quick identification of best model
- Database enables querying and comparison
- `is_active` flag for production model selection

### Decision 7: TensorBoard Integration
**Choice**: Real-time TensorBoard logging
- Log directory: `runs/{model_type}/{timestamp}`
- Metrics tracked:
  - Loss curves (train/val)
  - Accuracy/mAP per epoch
  - Confusion matrices
  - Learning rate schedule
  - Sample predictions (every N epochs)
- Access via: `tensorboard --logdir=runs`

**Rationale**:
- Standard tool in PyTorch ecosystem
- Real-time monitoring prevents wasting compute on failed runs
- Visual debugging of model behavior
- Historical comparison of training runs

### Decision 8: GPU/CPU Handling
**Choice**: Automatic device detection with graceful fallback
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Adjust batch size and workers based on device
if device.type == 'cpu':
    batch_size = batch_size // 2  # Reduce memory usage
    num_workers = 0  # Prevent multiprocessing issues on CPU
```

**Rationale**:
- Ensures functionality on all hardware
- Prevents out-of-memory errors
- Maintains user experience on CPU-only machines

## Risks / Trade-offs

### Risk 1: Overfitting on Small Dataset (172 samples)
**Likelihood**: High
**Impact**: High
**Mitigation**:
- Early stopping with patience=20 epochs
- Strong regularization (dropout=0.5, weight decay=1e-4)
- Data augmentation reduces effective memorization
- Monitor train/val gap closely in TensorBoard
- Use pretrained backbones to reduce parameters to learn

### Risk 2: PASS Class Still Underperforms Despite Augmentation
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Track per-class metrics (not just overall accuracy)
- Consider adjusting decision threshold for PASS class in inference
- Collect more PASS samples in next annotation round (guided by model)
- If critical, train binary "PASS vs not-PASS" classifier separately

### Risk 3: Training Time on CPU-Only Machines
**Likelihood**: Medium
**Impact**: Low (inconvenience, not failure)
**Mitigation**:
- Use lightweight models (YOLOv8n, ResNet18, EfficientNet-B0)
- Provide time estimates in GUI
- Allow pausing/resuming training
- Recommend GPU usage in documentation

### Risk 4: Model Versioning Storage Size
**Likelihood**: Low
**Impact**: Low
**Mitigation**:
- YOLO model: ~6MB
- Classifiers: ~10MB each
- Total per training run: ~25MB
- Implement optional cleanup of old models
- Compress models using torch.save(..., _use_new_zipfile_serialization=True)

## Migration Plan
N/A - This is a new feature, not a migration.

## Integration Points

### 1. Database Integration
Extend `src/core/database.py` with:
- `add_training_run(model_type, config, metrics)` → training_history.id
- `save_model_version(model_type, filepath, metrics)` → model_versions.id
- `get_active_model(model_type)` → filepath
- `list_model_versions(model_type)` → List[ModelVersion]

### 2. GUI Integration
Add to `src/gui/annotation_window.py`:
- Menu: Tools → Train Models
- Dialog: Training Configuration (epochs, batch size, etc.)
- Progress bar: Current epoch, ETA
- Button: Open TensorBoard
- Model selection dropdown for inference

### 3. File Structure
```
src/training/
├── __init__.py
├── data_preparation.py     # Dataset splitting and YOLO format export
├── yolo_trainer.py          # YOLOv8 training wrapper
├── classifier_trainer.py    # View and defect classifier training
├── augmentation.py          # Augmentation strategies
├── model_manager.py         # Model versioning and loading
├── config.py                # Configuration management
└── training_dialog.py       # PyQt6 training GUI

models/                       # Trained model storage
├── detection/
├── view/
└── defect/

runs/                         # TensorBoard logs
├── detection/
├── view/
└── defect/

configs/                      # Training configuration files
├── yolo_detection.yaml
├── view_classifier.yaml
└── defect_classifier.yaml
```

## Open Questions
1. Should we implement hyperparameter tuning (Optuna) in first version?
   - **Decision**: No, manual tuning acceptable initially
2. Do we need model quantization for deployment?
   - **Decision**: Defer to deployment phase (not needed for .exe)
3. Should training be blocking or non-blocking in GUI?
   - **Decision**: Non-blocking with progress updates (use QThread)
4. How to handle interrupted training?
   - **Decision**: Save checkpoints every N epochs, allow resume

## Success Metrics
- YOLOv8 detection: mAP@0.5 > 0.85
- View classifier: Accuracy > 0.95
- Defect classifier: Balanced accuracy > 0.80 (considering imbalance)
- PASS class: Recall > 0.70 (critical for not missing defects)
- Training time: < 30 minutes on GPU, < 2 hours on CPU
- TensorBoard accessible and informative
- Model versioning working correctly
