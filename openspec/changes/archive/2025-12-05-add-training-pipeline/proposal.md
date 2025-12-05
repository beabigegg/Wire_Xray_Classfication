# Change: Add Training Pipeline System

## Why
The annotation system is now complete with 172 annotated Wire Loop samples in the database. To progress from manual annotation to semi-automated detection, we need a training pipeline that can:

1. **Prepare training data** from the annotation database with proper train/val splits
2. **Train three specialized models**:
   - YOLOv8 for Wire Loop detection (bounding box localization)
   - CNN classifier for view angle (TOP/SIDE - 2 classes)
   - CNN classifier for defect type (PASS/沖線/晃動/碰觸 - 4 classes)
3. **Handle severe class imbalance** - PASS class has only 6 samples while 碰觸 has 68 samples
4. **Monitor training progress** with TensorBoard and metrics tracking
5. **Manage model versions** for iterative improvement

Current data distribution shows:
- View balance: TOP (86), SIDE (86) - perfectly balanced ✓
- Defect imbalance: 晃動 (58), 碰觸 (68), 沖線 (40), PASS (6) - severe imbalance ⚠
- BBox size: avg 206x78 pixels, suitable for YOLO detection
- Image specs: 1004x1004 pixels, 24-bit RGB

## What Changes
- **ADD** dataset preparation module with stratified train/val splitting
- **ADD** YOLO training pipeline for Wire Loop detection
- **ADD** view angle classifier training pipeline (2-class CNN)
- **ADD** defect type classifier training pipeline (4-class CNN)
- **ADD** data augmentation strategy to address PASS class (only 6 samples)
- **ADD** training monitoring with TensorBoard integration
- **ADD** model version management and storage
- **ADD** training configuration management (YAML-based)
- **ADD** class imbalance handling (weighted loss, focal loss, augmentation)
- **ADD** training history tracking in database
- **ADD** model evaluation metrics (mAP, accuracy, precision, recall, F1)
- **ADD** integration with annotation GUI for model selection

## Impact
- **Affected specs**: New `training-system` capability
- **Affected code**:
  - New: `src/training/data_preparation.py` - Dataset splitting and preparation
  - New: `src/training/yolo_trainer.py` - YOLO detection model training
  - New: `src/training/classifier_trainer.py` - View and defect classifiers
  - New: `src/training/augmentation.py` - Data augmentation strategies
  - New: `src/training/model_manager.py` - Model versioning and storage
  - New: `src/training/config.py` - Training configuration management
  - Extend: `src/core/database.py` - Add training_history table
  - Extend: `src/gui/annotation_window.py` - Add training dialog
- **Database schema**: Add `training_history` and `model_versions` tables
- **External dependencies**: Add TensorBoard, timm, albumentations
- **Testing**: Add training pipeline integration tests
- **Documentation**: Add training workflow guide

## Breaking Changes
None - This is a pure addition. Existing annotation functionality remains unchanged.

## Risks and Mitigations
1. **Risk**: PASS class (6 samples) insufficient for training
   - **Mitigation**: Heavy augmentation + weighted loss + potential synthetic data generation
2. **Risk**: Training may be slow on CPU-only machines
   - **Mitigation**: Automatic GPU/CPU detection with graceful fallback
3. **Risk**: Large model files may impact application size
   - **Mitigation**: Store models separately, load on-demand
4. **Risk**: Overfitting on small dataset (172 samples total)
   - **Mitigation**: Strong regularization, early stopping, extensive validation monitoring
