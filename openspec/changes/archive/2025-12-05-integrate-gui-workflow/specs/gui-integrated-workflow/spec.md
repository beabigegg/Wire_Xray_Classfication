# gui-integrated-workflow Specification

## Purpose
Provide a unified GUI experience for the complete Wire Loop classification workflow (annotation → training → inference), eliminating the need to switch between separate command-line tools and multiple applications.

## ADDED Requirements

### Requirement: GUI Training Configuration
The system SHALL provide a graphical interface for configuring and launching model training from the annotation window.

#### Scenario: Launch training dialog from annotation window
- **WHEN** user selects "Training" → "Train Detection Model" from menu
- **THEN** the system opens the Training Dialog
- **AND** pre-populates fields with default values from config
- **AND** displays current data statistics (total samples, train/val split)
- **AND** shows device availability (GPU/CPU)

#### Scenario: Configure training parameters
- **WHEN** user modifies training parameters (epochs, batch_size, learning_rate)
- **THEN** the system validates parameter ranges
- **AND** shows tooltips explaining each parameter
- **AND** enables/disables "Start Training" button based on validation

#### Scenario: Start training from GUI
- **WHEN** user clicks "Start Training" button
- **THEN** the system validates dataset preparation
- **AND** launches training in a background thread (QThread)
- **AND** keeps GUI responsive during training
- **AND** disables training parameters while training is active

#### Scenario: Handle insufficient training data
- **WHEN** user attempts to train with insufficient data (e.g., < 10 samples per class)
- **THEN** the system shows a warning dialog
- **AND** displays data statistics and recommendations
- **AND** prevents training from starting

### Requirement: Real-time Training Progress Monitoring
The system SHALL display real-time training progress, metrics, and logs in the GUI.

#### Scenario: Display training progress
- **WHEN** training is in progress
- **THEN** the system updates progress bar showing current epoch / total epochs
- **AND** displays current metrics (loss, accuracy/mAP)
- **AND** shows ETA (estimated time remaining)
- **AND** updates every epoch completion

#### Scenario: Display training logs
- **WHEN** training generates log messages
- **THEN** the system appends logs to the log viewer
- **AND** auto-scrolls to show latest messages
- **AND** color-codes log levels (INFO/WARNING/ERROR)

#### Scenario: Handle training completion
- **WHEN** training completes successfully
- **THEN** the system shows completion dialog with final metrics
- **AND** displays model save path and version
- **AND** offers options to "View in TensorBoard" or "Close"
- **AND** re-enables training parameter fields

#### Scenario: Handle training error
- **WHEN** training fails due to an error (GPU OOM, data issue, etc.)
- **THEN** the system displays user-friendly error message
- **AND** shows detailed error in log viewer
- **AND** suggests remediation steps (e.g., reduce batch size for OOM)
- **AND** stops training gracefully

### Requirement: Training Pause and Cancellation
The system SHALL support pausing, resuming, and canceling training operations.

#### Scenario: Pause training
- **WHEN** user clicks "Pause Training" button during active training
- **THEN** the system pauses training at the end of current epoch
- **AND** saves a checkpoint
- **AND** enables "Resume Training" button
- **AND** keeps progress and metrics displayed

#### Scenario: Resume paused training
- **WHEN** user clicks "Resume Training" button
- **THEN** the system resumes from last checkpoint
- **AND** continues with same parameters
- **AND** updates progress bar to show resumption

#### Scenario: Cancel training
- **WHEN** user clicks "Cancel Training" button
- **THEN** the system shows confirmation dialog
- **AND** IF confirmed, stops training at end of current epoch
- **AND** saves a checkpoint (if enabled in config)
- **AND** closes training dialog or resets to initial state

### Requirement: TensorBoard Integration
The system SHALL launch and manage TensorBoard from the GUI for training visualization.

#### Scenario: Launch TensorBoard from GUI
- **WHEN** user clicks "Open TensorBoard" button
- **THEN** the system detects available port (6006-6010)
- **AND** starts TensorBoard subprocess with runs/ directory
- **AND** opens system browser to TensorBoard URL (http://localhost:PORT)
- **AND** displays "TensorBoard: Running on port PORT" in status bar

#### Scenario: Handle TensorBoard port conflict
- **WHEN** TensorBoard default port 6006 is already in use
- **THEN** the system automatically tries ports 6007-6010
- **AND** uses the first available port
- **AND** displays the actual port used in status message

#### Scenario: Stop TensorBoard
- **WHEN** user closes annotation window or clicks "Stop TensorBoard"
- **THEN** the system terminates TensorBoard subprocess
- **AND** removes status bar indicator
- **AND** cleans up resources

#### Scenario: TensorBoard auto-open during training
- **WHEN** training starts and TensorBoard is not running
- **THEN** the system offers to auto-launch TensorBoard
- **AND** IF user accepts, launches TensorBoard and opens browser
- **AND** continues training without blocking

### Requirement: Model Version Management
The system SHALL provide GUI for viewing, selecting, and managing trained model versions.

#### Scenario: View available model versions
- **WHEN** user selects "Training" → "Manage Models" from menu
- **THEN** the system opens Model Selector Dialog
- **AND** displays three sections (Detection / View / Defect)
- **AND** lists all model versions with: name, date, metrics, [ACTIVE] badge
- **AND** sorts by date (newest first)

#### Scenario: Select active model
- **WHEN** user selects a model and clicks "Set as Active"
- **THEN** the system updates database to mark model as active
- **AND** updates [ACTIVE] badge in list
- **AND** removes [ACTIVE] from previous active model
- **AND** shows confirmation message

#### Scenario: View model details
- **WHEN** user clicks on a model in the list
- **THEN** the system displays model details panel showing:
  - Model file path
  - Training date and duration
  - Metrics (mAP for detection, accuracy for classifiers)
  - Training configuration used (epochs, batch_size, lr)
  - Dataset version used

#### Scenario: Delete model version
- **WHEN** user selects a model and clicks "Delete Model"
- **THEN** the system shows confirmation dialog with model details
- **AND** IF confirmed, deletes model file from disk
- **AND** CASCADE deletes database record
- **AND** IF was active model, marks no model as active (prompts user to select new active)
- **AND** refreshes model list

### Requirement: Integrated Inference Workflow
The system SHALL integrate inference functionality into the annotation window for seamless workflow.

#### Scenario: Run inference on current image
- **WHEN** user selects "Inference" → "Run on Current Image" (Ctrl+I)
- **THEN** the system loads active models (Detection/View/Defect)
- **AND** runs three-stage inference pipeline
- **AND** displays predicted bounding box on image
- **AND** shows predicted view and defect labels with confidence scores
- **AND** allows user to accept or modify prediction

#### Scenario: Accept predicted annotation
- **WHEN** user reviews inference result and clicks "Accept"
- **THEN** the system adds predicted annotation to database
- **AND** assigns current annotator username
- **AND** marks annotation as "model_predicted = True" with model version
- **AND** moves to next image

#### Scenario: Modify predicted annotation
- **WHEN** user corrects prediction (bbox, view, or defect)
- **THEN** the system saves corrected annotation
- **AND** marks as "model_predicted = False" (human-corrected)
- **AND** records both original prediction and correction for training feedback

#### Scenario: Run batch inference
- **WHEN** user selects "Inference" → "Run Batch Inference"
- **THEN** the system opens folder selection dialog
- **AND** processes all images in folder with progress bar
- **AND** saves results to CSV/JSON
- **AND** displays summary statistics (total, detected, failed)

### Requirement: Workflow State Management
The system SHALL maintain consistent state across annotation, training, and inference operations.

#### Scenario: Training with pending annotations
- **WHEN** user starts training while annotation window has unsaved changes
- **THEN** the system prompts to save or discard changes
- **AND** IF saved, includes in training dataset
- **AND** IF discarded, continues without new data

#### Scenario: Model update notification
- **WHEN** new model version is trained and saved
- **THEN** the system updates model list in background
- **AND** shows notification "New model available: [version]"
- **AND** offers to set as active model

#### Scenario: Concurrent operations handling
- **WHEN** user attempts to start training while another training is in progress
- **THEN** the system shows warning "Training already in progress"
- **AND** offers to view current training progress
- **AND** prevents multiple concurrent trainings

## ADDED Requirements (Menu and Status Bar)

### Requirement: Annotation Window Menu Structure
The annotation window menu bar SHALL be extended to include Training and Inference menus.

#### Scenario: Access training functions from menu
- **WHEN** user opens "Training" menu
- **THEN** the system displays menu items:
  - Train Detection Model (Ctrl+Shift+D)
  - Train View Classifier (Ctrl+Shift+V)
  - Train Defect Classifier (Ctrl+Shift+F)
  - Train All Models (Ctrl+Shift+A)
  - [Separator]
  - Open TensorBoard (Ctrl+Shift+T)
  - Manage Models (Ctrl+Shift+M)

#### Scenario: Access inference functions from menu
- **WHEN** user opens "Inference" menu
- **THEN** the system displays menu items:
  - Run on Current Image (Ctrl+I)
  - Run Batch Inference (Ctrl+Shift+I)
  - [Separator]
  - Select Active Models
  - View Results History

### Requirement: Status Bar Information
The annotation window status bar SHALL display training and TensorBoard status.

#### Scenario: Display active models in status bar
- **WHEN** annotation window is open
- **THEN** the status bar shows "Models: Detection v1, View v2, Defect v1"
- **AND** clicking on models indicator opens Model Selector Dialog

#### Scenario: Display training status
- **WHEN** training is in progress
- **THEN** the status bar shows "Training: [Model Type] - Epoch X/Y"
- **AND** clicking opens Training Dialog to view progress

#### Scenario: Display TensorBoard status
- **WHEN** TensorBoard is running
- **THEN** the status bar shows "TensorBoard: localhost:6006"
- **AND** clicking opens TensorBoard in browser
