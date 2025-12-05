# tensorboard-integration Specification

## Purpose
Automatically log all training metrics and visualizations to TensorBoard for real-time monitoring and post-training analysis.

## ADDED Requirements

### Requirement: Automatic TensorBoard Logging
All trainers (YOLO, View Classifier, Defect Classifier) SHALL automatically log metrics to TensorBoard without requiring manual configuration.

#### Scenario: Auto-launch TensorBoard on training start
- **WHEN** user clicks "Start Training" button in Training Dialog
- **AND** "Auto-launch TensorBoard" checkbox is checked
- **THEN** system starts TensorBoard server on available port (6006 or next available)
- **AND** opens TensorBoard in default browser
- **AND** TensorBoard URL is displayed in training dialog: "http://localhost:6006"
- **AND** status label shows "TensorBoard running on port 6006"
- **WHEN** training completes or is cancelled
- **THEN** TensorBoard server remains running (user can close manually)

#### Scenario: YOLO detection metrics logging
- **WHEN** YOLO trainer runs training epoch
- **THEN** system logs to TensorBoard every epoch:
  - Scalar: train/box_loss (bounding box regression loss)
  - Scalar: train/cls_loss (classification loss)
  - Scalar: train/dfl_loss (distribution focal loss)
  - Scalar: val/mAP@0.5 (mean average precision at IoU 0.5)
  - Scalar: val/mAP@0.5:0.95 (mean average precision at IoU 0.5-0.95)
  - Scalar: val/precision (detection precision)
  - Scalar: val/recall (detection recall)
  - Scalar: learning_rate (current learning rate)
- **AND** every 10 epochs, system logs:
  - Images: Sample predictions with bounding boxes (6 images grid)
  - Images: Ground truth vs predictions comparison
- **AND** all metrics appear in TensorBoard within 5 seconds of epoch completion

#### Scenario: View classifier metrics logging
- **WHEN** View Classifier trainer runs training epoch
- **THEN** system logs to TensorBoard every epoch:
  - Scalar: train/loss (cross-entropy loss)
  - Scalar: val/loss (validation loss)
  - Scalar: train/accuracy (training accuracy)
  - Scalar: val/accuracy (validation accuracy)
  - Scalar: val/precision_正視 (precision for 正視 class)
  - Scalar: val/precision_俯視 (precision for 俯視 class)
  - Scalar: val/recall_正視 (recall for 正視 class)
  - Scalar: val/recall_俯視 (recall for 俯視 class)
  - Scalar: val/f1_正視 (F1-score for 正視 class)
  - Scalar: val/f1_俯視 (F1-score for 俯視 class)
  - Scalar: learning_rate (current learning rate)
- **AND** every 5 epochs, system logs:
  - Images: Confusion matrix heatmap (2x2)
- **AND** every 10 epochs, system logs:
  - Images: Sample predictions grid (8 images with predicted and true labels)
  - Histogram: model/conv1.weight (first layer weights)
  - Histogram: model/fc.weight (final layer weights)

#### Scenario: Defect classifier metrics logging with PASS class monitoring
- **WHEN** Defect Classifier trainer runs training epoch
- **THEN** system logs to TensorBoard every epoch:
  - Scalar: train/loss (weighted cross-entropy or focal loss)
  - Scalar: val/loss (validation loss)
  - Scalar: train/accuracy (standard accuracy)
  - Scalar: val/accuracy (validation accuracy)
  - Scalar: val/balanced_accuracy (balanced accuracy for imbalanced data)
  - Scalar: val/precision_PASS (PASS class precision - CRITICAL)
  - Scalar: val/precision_沖線 (沖線 class precision)
  - Scalar: val/precision_晃動 (晃動 class precision)
  - Scalar: val/precision_碰觸 (碰觸 class precision)
  - Scalar: val/recall_PASS (PASS class recall - CRITICAL)
  - Scalar: val/recall_沖線 (沖線 class recall)
  - Scalar: val/recall_晃動 (晃動 class recall)
  - Scalar: val/recall_碰觸 (碰觸 class recall)
  - Scalar: val/f1_PASS (PASS class F1-score - CRITICAL)
  - Scalar: val/f1_沖線 (沖線 class F1-score)
  - Scalar: val/f1_晃動 (晃動 class F1-score)
  - Scalar: val/f1_碰觸 (碰觸 class F1-score)
  - Scalar: PASS/false_negatives (count of PASS samples misclassified as defects)
  - Scalar: PASS/false_positives (count of defect samples misclassified as PASS)
  - Scalar: learning_rate (current learning rate)
- **AND** every 5 epochs, system logs:
  - Images: 4x4 confusion matrix heatmap
- **AND** every 10 epochs, system logs:
  - Images: PASS class correct predictions (grid of 8 images)
  - Images: PASS class incorrect predictions (grid of 8 images with predicted labels)
  - Images: Full sample predictions grid (16 images from all classes)
- **AND** every epoch, system logs batch class distribution:
  - Histogram: batch/class_distribution (verify balanced sampling)
- **AND** system highlights PASS class metrics in training dialog status

#### Scenario: TensorBoard performance overhead validation
- **WHEN** training runs with TensorBoard logging enabled
- **THEN** training time per epoch should NOT increase by more than 5%
- **AND** IF overhead > 5%, system automatically reduces logging frequency:
  - Images logged every 20 epochs instead of 10
  - Confusion matrix logged every 10 epochs instead of 5
- **AND** user is notified: "TensorBoard logging frequency reduced to maintain performance"

### Requirement: TensorBoard Logger Utility
The system SHALL provide a reusable TensorBoardLogger utility class for consistent logging across all trainers.

#### Scenario: Initialize TensorBoard logger in trainer
- **WHEN** trainer class is instantiated (YOLOTrainer, ViewClassifier, DefectClassifier)
- **AND** TensorBoard logging is enabled
- **THEN** trainer creates SummaryWriter with log directory:
  - YOLO: `runs/detection/train_YYYYMMDD_HHMMSS`
  - View: `runs/view_classifier/train_YYYYMMDD_HHMMSS`
  - Defect: `runs/defect_classifier/train_YYYYMMDD_HHMMSS`
- **AND** logger is ready to accept metrics

#### Scenario: Log scalar metrics
- **WHEN** trainer calls `logger.log_scalar(tag, value, step)`
- **THEN** system writes to TensorBoard with given tag, value, and global step
- **AND** data is flushed to disk immediately

#### Scenario: Log confusion matrix
- **WHEN** trainer calls `logger.log_confusion_matrix(cm, class_names, step)`
- **THEN** system creates confusion matrix heatmap using matplotlib
- **AND** annotates each cell with count and percentage
- **AND** logs figure to TensorBoard as image
- **AND** closes matplotlib figure to free memory

#### Scenario: Log prediction samples
- **WHEN** trainer calls `logger.log_prediction_grid(images, true_labels, pred_labels, step)`
- **THEN** system creates grid of images using torchvision.utils.make_grid
- **AND** adds text annotations for true and predicted labels
- **AND** logs grid to TensorBoard as single image
- **AND** frees image tensor memory after logging

### Requirement: TensorBoard Manager Integration
The existing TensorBoardManager SHALL be enhanced to support auto-launch and status tracking.

#### Scenario: Check TensorBoard availability
- **WHEN** Training Dialog opens
- **THEN** system checks if TensorBoard is installed
- **AND** IF not installed, disables "Auto-launch TensorBoard" checkbox
- **AND** shows tooltip: "TensorBoard not installed. Run: pip install tensorboard"

#### Scenario: Manage multiple TensorBoard instances
- **WHEN** user trains different model types simultaneously
- **THEN** each model type has separate TensorBoard log directory
- **AND** single TensorBoard instance can view all runs
- **AND** runs are organized by model type in TensorBoard UI

#### Scenario: Clean up old TensorBoard logs
- **WHEN** user opens annotation window
- **AND** TensorBoard logs directory has > 50 run folders
- **THEN** system shows notification:
  - "Found 65 TensorBoard run folders. Clean up old runs?"
  - [Keep Last 30 Runs] [Keep Last 50 Runs] [Keep All]
- **WHEN** user selects cleanup option
- **THEN** system deletes oldest run folders beyond selected count
- **AND** confirms: "Cleaned up 35 old TensorBoard runs"

## ADDED Requirements

### Requirement: Smart Logging Configuration
The system SHALL intelligently adjust logging frequency based on training characteristics.

#### Scenario: Adjust logging for short training runs
- **WHEN** training runs for < 20 epochs
- **THEN** system logs images every 5 epochs (instead of 10)
- **AND** logs confusion matrix every 2 epochs (instead of 5)
- **AND** ensures at least 3 image logging events during training

#### Scenario: Adjust logging for long training runs
- **WHEN** training runs for > 100 epochs
- **THEN** system logs images every 20 epochs (instead of 10)
- **AND** logs confusion matrix every 10 epochs (instead of 5)
- **AND** reduces overhead for long training sessions

#### Scenario: Async figure logging (optional performance optimization)
- **WHEN** trainer logs matplotlib figures (confusion matrix, prediction grid)
- **THEN** system MAY use background thread for figure creation
- **AND** training loop continues without blocking
- **AND** figure is logged asynchronously when ready
- **AND** memory is freed after logging completes

### Requirement: PASS Class Special Monitoring
For Defect Classifier, the system SHALL provide dedicated monitoring for the PASS class (minority class).

#### Scenario: PASS class metrics dashboard
- **WHEN** Defect Classifier training is running
- **THEN** TensorBoard shows dedicated PASS class metrics:
  - PASS/recall (most critical - should be ≥ 95%)
  - PASS/precision (should be ≥ 90%)
  - PASS/f1_score (harmonic mean)
  - PASS/false_negatives (count of missed PASS samples)
  - PASS/false_positives (count of wrongly flagged defects)
- **AND** training dialog shows PASS recall in status bar
- **AND** IF PASS recall drops below 90%, system shows warning

#### Scenario: PASS class sample visualization
- **WHEN** system logs PASS class samples every 10 epochs
- **THEN** creates two separate grids:
  - Correct PASS predictions (8 samples)
  - Incorrect PASS predictions (8 samples with predicted defect labels)
- **AND** highlights false negatives (PASS predicted as defect) in red border
- **AND** shows confidence scores for each prediction

## Success Criteria

1. **Automatic Logging**: All trainers log metrics without manual SummaryWriter configuration
2. **Real-time Monitoring**: Metrics appear in TensorBoard within 5 seconds of epoch completion
3. **Performance**: Training overhead < 5% with TensorBoard enabled
4. **PASS Monitoring**: Defect classifier provides dedicated PASS class metrics
5. **Visual Quality**: Confusion matrices and prediction grids are clearly readable
6. **Auto-launch**: TensorBoard opens automatically when training starts (if enabled)
7. **Cleanup**: Old runs can be cleaned up to save disk space
