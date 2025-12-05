# Change Proposal: Enhance Training Controls

## Why

The current training system (add-training-pipeline, 114/136 tasks complete) provides basic GUI training functionality through TrainingDialog, but users cannot fully control the training process:

**Current limitations**:
1. **Limited hyperparameter control**: Only epochs, batch size, learning rate, and device are configurable
2. **No advanced model options**: Cannot select model backbones, optimizers, or learning rate schedulers
3. **No training flow control**: Cannot pause, resume, or gracefully cancel training
4. **No TensorBoard auto-logging**: Training metrics are not automatically logged to TensorBoard
5. **No model comparison**: Cannot compare multiple model versions side-by-side

**User needs** (from NEXT_PHASE_PLAN.md):
- Fine-tune YOLO detection parameters (image size, patience, confidence threshold)
- Select classifier backbones (ResNet variants, EfficientNet variants)
- Configure class imbalance handling (loss functions, class weights, augmentation factors)
- Monitor training progress in TensorBoard with auto-logging
- Pause/resume long training runs
- Compare model versions to select the best performing model

**Impact on workflow**:
- Users currently must edit YAML config files to change advanced parameters (non-intuitive)
- Training cannot be interrupted without losing all progress
- No visual comparison of model performance (must manually check metrics in database)
- TensorBoard must be manually configured and launched separately

## What Changes

This change enhances the training system with three major capabilities:

### 1. Advanced Training Controls (training-controls capability)
- **MODIFY** TrainingDialog to expose all hyperparameters in GUI
- **ADD** Model-specific parameter sections (YOLO, View Classifier, Defect Classifier)
- **ADD** Pause/Resume/Cancel training functionality
- **ADD** Checkpoint save/load for interrupted training
- **MODIFY** TrainingWorker to support pause/resume/cancel signals

### 2. TensorBoard Deep Integration (tensorboard-integration capability)
- **ADD** Automatic TensorBoard logging in all trainers (YOLO, View, Defect)
- **ADD** Scalar metrics logging (loss, accuracy, mAP, per-class metrics)
- **ADD** Visual logging (confusion matrices, sample predictions, histograms)
- **ADD** Special monitoring for PASS class (minority class in defect detection)
- **ADD** Auto-launch TensorBoard when training starts
- **MODIFY** Trainers to use SummaryWriter throughout training loops

### 3. Model Comparison Dashboard (model-comparison capability)
- **ADD** ModelComparator backend for metrics comparison
- **ADD** ModelComparisonDialog GUI for side-by-side comparison
- **ADD** Support for comparing 2-4 models simultaneously
- **ADD** Delta calculation and visual indicators
- **ADD** Automatic recommendation based on metrics
- **ADD** Integration with annotation window menu

## Impact

**Affected specs**:
- **MODIFIED**: `training-system` capability (extends existing training functionality)
- **ADDED**: `training-controls` capability (new GUI controls)
- **ADDED**: `tensorboard-integration` capability (automatic logging)
- **ADDED**: `model-comparison` capability (model evaluation)

**Affected code**:
- Modify: `src/gui/training_dialog.py` - Expand parameter UI (+300 lines)
- Modify: `src/gui/training_worker.py` - Add pause/resume/cancel (+150 lines)
- Modify: `src/training/yolo_trainer.py` - Add TensorBoard logging (+100 lines)
- Modify: `src/training/view_classifier.py` - Add TensorBoard logging (+120 lines)
- Modify: `src/training/defect_classifier.py` - Add TensorBoard logging (+150 lines)
- New: `src/training/model_comparator.py` - Model comparison backend
- New: `src/gui/model_comparison_dialog.py` - Comparison UI
- Modify: `src/gui/annotation_window.py` - Add comparison menu item

**Database schema**: No changes (uses existing model_versions and training_history tables)

**External dependencies**:
- matplotlib (for confusion matrix plotting) - already installed
- seaborn (for better heatmaps) - needs pip install seaborn

**Testing**:
- Unit tests for ModelComparator
- Integration tests for pause/resume functionality
- End-to-end test for TensorBoard logging

**Documentation**:
- Update GUI_WORKFLOW_GUIDE.md with advanced parameter descriptions
- Add TRAINING_CONTROLS_GUIDE.md for pause/resume usage
- Add MODEL_COMPARISON_GUIDE.md for comparison feature

## Breaking Changes

None - All changes are additive or extend existing functionality without breaking current API.

**Backward compatibility**:
- Existing training configs will work (new parameters have defaults)
- Existing trained models remain compatible
- Existing database schema unchanged

## Risks and Mitigations

### Risk 1: UI Complexity
**Risk**: Too many parameters may overwhelm users
**Mitigation**:
- Use collapsible "Advanced Options" section (default collapsed)
- Provide sensible defaults for all parameters
- Add comprehensive tooltips explaining each parameter
- Progressive disclosure: show only relevant params based on model type

### Risk 2: Pause/Resume State Management
**Risk**: Checkpoint corruption or incomplete state restoration
**Mitigation**:
- Atomic checkpoint saving (write to temp file, then rename)
- Validate checkpoint integrity before loading
- Store complete state (model weights, optimizer, epoch, scheduler)
- Add checkpoint versioning for compatibility

### Risk 3: TensorBoard Performance Impact
**Risk**: Logging overhead may slow down training
**Mitigation**:
- Log scalars every epoch (lightweight)
- Log images/matrices every N epochs (configurable, default 5-10)
- Use async writes where possible
- Allow disabling TensorBoard via checkbox

### Risk 4: Model Comparison Memory Usage
**Risk**: Loading multiple large models may consume too much memory
**Mitigation**:
- Compare metrics only (don't load model weights)
- Limit comparison to 4 models maximum
- Use database queries (models already saved)

## Dependencies

**Depends on**:
- `add-training-pipeline` (114/136 tasks) - must be at current state
- `integrate-gui-workflow` (complete) - uses TrainingDialog and TrainingWorker

**Blocks**:
- None - This is an enhancement, not a blocker

**Related**:
- Future: Active Learning integration (will use model comparison)
- Future: Hyperparameter tuning (will use enhanced controls)

## Success Criteria

1. **Advanced Controls**: User can configure all major hyperparameters from GUI
2. **Pause/Resume**: Training can be paused and resumed without data loss
3. **TensorBoard Auto-logging**: All metrics automatically appear in TensorBoard during training
4. **Model Comparison**: User can compare 2-4 models and see delta metrics
5. **Performance**: Training speed not degraded by more than 5% with TensorBoard enabled
6. **Stability**: Pause/resume works reliably across 100+ test runs
7. **Usability**: Advanced parameters collapsed by default, 90%+ users find it intuitive

## Estimated Effort

**Total**: 20-26 hours (3-4 days for 1 developer)

**Breakdown**:
- Training Controls: 8-11 hours
  - Parameter UI expansion: 3-4 hours
  - TrainingWorker pause/resume: 3-4 hours
  - Checkpoint management: 2-3 hours

- TensorBoard Integration: 6-7 hours
  - YOLO logging: 2 hours
  - View Classifier logging: 2 hours
  - Defect Classifier logging (with PASS monitoring): 2-3 hours

- Model Comparison: 6-8 hours
  - ModelComparator backend: 2-3 hours
  - ComparisonDialog UI: 3-4 hours
  - Integration: 1 hour

**Testing**: 4-5 hours (included in above estimates)
**Documentation**: 2-3 hours (included in above estimates)
