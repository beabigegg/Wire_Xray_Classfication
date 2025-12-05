# training-controls Specification

## Purpose
Provide comprehensive GUI controls for all training hyperparameters and enable pause/resume/cancel functionality for long-running training sessions.

## ADDED Requirements

### Requirement: Advanced Hyperparameter Configuration
The Training Dialog SHALL expose all major hyperparameters for user configuration based on selected model type.

#### Scenario: Configure YOLO detection parameters
- **WHEN** user selects "Detection Model (YOLO)" in Training Dialog
- **AND** expands "Advanced Options" section
- **THEN** system displays YOLO-specific parameters:
  - Image Size dropdown [640, 1004, 1280] (default: 1004)
  - Optimizer dropdown [Adam, SGD, AdamW] (default: Adam)
  - Weight Decay [0.0 - 0.01] (default: 0.0005)
  - Warmup Epochs [0 - 10] (default: 3)
  - Patience (early stopping) [5 - 50] (default: 20)
  - Confidence Threshold [0.1 - 0.9] (default: 0.25)
  - IoU Threshold [0.3 - 0.9] (default: 0.5)
- **AND** all parameters have tooltips explaining their purpose
- **AND** changes are reflected in training configuration

#### Scenario: Configure View Classifier parameters
- **WHEN** user selects "View Classifier" in Training Dialog
- **AND** expands "Advanced Options" section
- **THEN** system displays View Classifier-specific parameters:
  - Model Backbone dropdown [ResNet18, ResNet34, ResNet50, ResNet101] (default: ResNet18)
  - Pretrained Weights checkbox (default: checked)
  - Optimizer dropdown [Adam, SGD, AdamW] (default: Adam)
  - Learning Rate Scheduler [None, StepLR, CosineAnnealing, ReduceLROnPlateau] (default: StepLR)
  - Step Size (if StepLR) [5 - 50] (default: 10)
  - Gamma (if StepLR) [0.1 - 0.9] (default: 0.1)
  - Dropout Rate [0.0 - 0.8] (default: 0.3)
  - Weight Decay [0.0 - 0.01] (default: 0.0001)
  - Data Augmentation Intensity [Low, Medium, High] (default: Medium)
- **AND** parameters dynamically update based on scheduler selection
- **AND** all parameters have comprehensive tooltips

#### Scenario: Configure Defect Classifier parameters (class imbalance handling)
- **WHEN** user selects "Defect Classifier" in Training Dialog
- **AND** expands "Advanced Options" section
- **THEN** system displays Defect Classifier-specific parameters:
  - Model Backbone dropdown [EfficientNet-B0, B1, B2, B3] (default: EfficientNet-B0)
  - Loss Function dropdown [CrossEntropy, WeightedCE, FocalLoss] (default: WeightedCE)
  - Focal Loss Gamma [0.5 - 5.0] (default: 2.0, only if FocalLoss selected)
  - Class Weights [Auto, Manual, Balanced] (default: Auto)
  - PASS Class Augmentation Factor [5x, 10x, 20x, 30x] (default: 20x)
  - Balanced Batch Sampling checkbox (default: checked)
  - Optimizer dropdown [Adam, SGD, AdamW] (default: Adam)
  - Learning Rate Scheduler [None, StepLR, CosineAnnealing] (default: StepLR)
  - Dropout Rate [0.0 - 0.8] (default: 0.4)
  - Weight Decay [0.0 - 0.01] (default: 0.0001)
- **AND** system shows warning if PASS augmentation < 10x (insufficient for 6 samples)
- **AND** Focal Loss Gamma is only enabled when FocalLoss is selected
- **AND** all parameters have detailed tooltips with class imbalance context

#### Scenario: Save and load parameter presets
- **WHEN** user configures custom parameters
- **AND** clicks "Save Preset" button
- **THEN** system prompts for preset name
- **AND** saves all current parameters as named preset
- **AND** preset appears in "Load Preset" dropdown
- **WHEN** user selects a preset from dropdown
- **THEN** all parameters are restored to preset values

### Requirement: Training Flow Control
The system SHALL support pausing, resuming, and cancelling training with proper state management.

#### Scenario: Pause training during execution
- **WHEN** training is in progress (state = RUNNING)
- **AND** user clicks "Pause Training" button
- **THEN** system sets pause flag in TrainingWorker
- **AND** training completes current epoch
- **AND** system saves checkpoint with complete state:
  - Model weights
  - Optimizer state
  - Learning rate scheduler state
  - Current epoch number
  - Training/validation loss history
  - Best metric value
  - Full training configuration
- **AND** training state changes to PAUSED
- **AND** "Pause Training" button becomes disabled
- **AND** "Resume Training" button becomes enabled
- **AND** status label shows "Training paused at epoch X/Y"
- **AND** checkpoint file is saved atomically (no corruption risk)

#### Scenario: Resume paused training
- **WHEN** training is paused (state = PAUSED)
- **AND** user clicks "Resume Training" button
- **THEN** system loads checkpoint file
- **AND** verifies checkpoint integrity
- **AND** restores complete training state:
  - Model weights
  - Optimizer state
  - Learning rate scheduler state
  - Epoch counter
  - Loss histories
- **AND** training state changes to RUNNING
- **AND** "Resume Training" button becomes disabled
- **AND** "Pause Training" button becomes enabled
- **AND** training continues from next epoch
- **AND** TensorBoard graphs show continuous curve (no discontinuity)

#### Scenario: Cancel training with cleanup
- **WHEN** training is in progress or paused
- **AND** user clicks "Cancel Training" button
- **THEN** system shows confirmation dialog:
  - "Are you sure you want to cancel training?"
  - "Current progress will be saved as a checkpoint for future reference."
  - [Yes] [No] buttons
- **WHEN** user confirms cancellation
- **THEN** system sets cancel flag in TrainingWorker
- **AND** training stops gracefully at end of current epoch
- **AND** system saves final checkpoint (optional, user-configurable)
- **AND** updates database training_history with status='cancelled'
- **AND** records final metrics achieved before cancellation
- **AND** closes TrainingDialog
- **AND** displays message: "Training cancelled. Checkpoint saved."

#### Scenario: Handle checkpoint save failure
- **WHEN** system attempts to save checkpoint
- **AND** disk space is insufficient OR path is invalid OR permissions denied
- **THEN** system shows error message:
  - "Failed to save checkpoint: [reason]"
  - "Training will continue but cannot be resumed if stopped."
- **AND** training continues (does not abort)
- **AND** "Pause" button becomes disabled (cannot pause without checkpoint)
- **AND** logs error to application log

#### Scenario: Resume training from previous session
- **WHEN** user opens Training Dialog
- **AND** checkpoint file exists from previous session
- **THEN** system shows notification:
  - "Found checkpoint from previous training session"
  - Model Type: Defect Classifier
  - Last Epoch: 45/100
  - Date: 2025-11-07 14:30
  - [Resume Training] [Start New Training] [Discard Checkpoint]
- **WHEN** user clicks "Resume Training"
- **THEN** system loads checkpoint and continues from epoch 46
- **WHEN** user clicks "Start New Training"
- **THEN** system archives old checkpoint and starts fresh training
- **WHEN** user clicks "Discard Checkpoint"
- **THEN** system deletes checkpoint file

### Requirement: Parameter Validation and Defaults
The system SHALL validate all parameter inputs and provide sensible defaults based on model type and dataset characteristics.

#### Scenario: Validate batch size vs available memory
- **WHEN** user sets batch size to large value (e.g., 128)
- **AND** device is CUDA
- **THEN** system estimates GPU memory requirement
- **AND** IF estimated memory > available GPU memory
  - Show warning: "Batch size 128 may cause out-of-memory errors. Recommended: 16-32"
  - Allow user to proceed or change value
- **AND** IF device is CPU
  - Show info: "Large batch sizes on CPU may be very slow. Recommended: 4-8"

#### Scenario: Auto-adjust parameters for small dataset
- **WHEN** user starts training
- **AND** total training samples < 100
- **THEN** system shows warning:
  - "Small dataset detected (N samples). Adjusting parameters to prevent overfitting:"
  - Reducing epochs to 30 (was 100)
  - Increasing dropout to 0.5 (was 0.3)
  - Enabling stronger augmentation
  - [Accept Changes] [Use Original Settings] [Customize]

#### Scenario: Validate learning rate range
- **WHEN** user enters learning rate outside reasonable range
- **AND** learning rate > 0.1 OR learning rate < 0.00001
- **THEN** system shows warning:
  - "Learning rate X.XXX is unusually [high|low]"
  - "Typical range: 0.0001 - 0.01"
  - "Are you sure you want to use this value?"
  - [Yes] [No, use recommended: 0.001]

## ADDED Requirements

### Requirement: Advanced Options UI Pattern
The Training Dialog SHALL use progressive disclosure to manage UI complexity.

#### Scenario: Expand advanced options
- **WHEN** user opens Training Dialog
- **THEN** "Advanced Options" section is collapsed by default
- **AND** shows "▶ Advanced Options" with expansion indicator
- **AND** basic parameters (epochs, batch size, learning rate, device) are always visible
- **WHEN** user clicks "Advanced Options" header
- **THEN** section expands to show model-specific parameters
- **AND** indicator changes to "▼ Advanced Options"
- **AND** section height auto-adjusts to fit content

#### Scenario: Context-sensitive parameter visibility
- **WHEN** user changes Model Type from "View Classifier" to "Detection Model"
- **THEN** system hides View Classifier parameters
- **AND** shows YOLO Detection parameters
- **AND** preserves basic parameter values (epochs, batch size, learning rate)
- **AND** advanced parameters reset to model-specific defaults

### Requirement: Training Control Button States
The system SHALL manage button states to prevent invalid operations.

#### Scenario: Button states during different training phases
- **WHEN** training state is IDLE
- **THEN** "Start Training" button is enabled
- **AND** "Pause Training" button is disabled and grayed out
- **AND** "Resume Training" button is disabled and grayed out
- **AND** "Cancel Training" button is disabled and grayed out
- **AND** parameter input fields are enabled

- **WHEN** training state is RUNNING
- **THEN** "Start Training" button is disabled
- **AND** "Pause Training" button is enabled
- **AND** "Resume Training" button is disabled
- **AND** "Cancel Training" button is enabled
- **AND** parameter input fields are disabled (locked)

- **WHEN** training state is PAUSED
- **THEN** "Start Training" button is disabled
- **AND** "Pause Training" button is disabled
- **AND** "Resume Training" button is enabled
- **AND** "Cancel Training" button is enabled
- **AND** parameter input fields remain disabled

### Requirement: Checkpoint Management
The system SHALL manage checkpoints with integrity verification and cleanup.

#### Scenario: Automatic checkpoint cleanup
- **WHEN** training completes successfully
- **THEN** system deletes checkpoint file (no longer needed)
- **AND** keeps only final trained model

- **WHEN** training is cancelled
- **THEN** system keeps checkpoint file for X days (default: 7 days)
- **AND** checkpoint filename includes timestamp for identification
- **AND** old checkpoints are auto-deleted after retention period

#### Scenario: Checkpoint integrity verification
- **WHEN** system saves checkpoint
- **THEN** writes to temporary file first (.ckpt.tmp)
- **AND** attempts to load temporary file to verify integrity
- **AND** IF verification succeeds: rename to actual checkpoint file (.ckpt)
- **AND** IF verification fails: delete temporary file and raise error
- **AND** never overwrites valid checkpoint with invalid one

#### Scenario: Multiple checkpoints for different models
- **WHEN** multiple models are being trained
- **THEN** each model type has separate checkpoint file:
  - checkpoints/yolo_detection_latest.ckpt
  - checkpoints/view_classifier_latest.ckpt
  - checkpoints/defect_classifier_latest.ckpt
- **AND** models do not interfere with each other's checkpoints
