# training-controls Specification

## Purpose
TBD - created by archiving change enhance-training-controls. Update Purpose after archive.
## Requirements
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

