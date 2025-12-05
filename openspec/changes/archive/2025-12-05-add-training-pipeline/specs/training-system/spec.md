## ADDED Requirements

### Requirement: Dataset Preparation and Splitting
The system SHALL prepare training datasets from annotation database with stratified train/validation splits maintaining class distribution.

#### Scenario: Stratified train/val split
- **WHEN** preparing datasets for training
- **THEN** the system splits data 80% train, 20% validation
- **AND** stratifies split by defect type to maintain class distribution
- **AND** ensures all defect classes are represented in both train and val sets
- **AND** stores split metadata (which images in train vs val) in database

#### Scenario: Export YOLO format for detection
- **WHEN** preparing data for YOLO detection training
- **THEN** the system exports annotations to YOLO format (.txt files)
- **AND** converts pixel coordinates to normalized center-based format
- **AND** creates train.txt and val.txt listing image paths
- **AND** creates data.yaml with class names and dataset paths
- **AND** validates all bounding boxes are within image boundaries

#### Scenario: Generate classifier datasets
- **WHEN** preparing data for view or defect classifier training
- **THEN** the system crops bounding box regions from full images
- **AND** resizes crops to 224x224 for classifier input
- **AND** organizes crops into class-based directory structure
- **AND** maintains original aspect ratio with padding if needed
- **AND** stores mapping between crops and original annotations

#### Scenario: Validate minimum samples per class
- **WHEN** preparing datasets with class imbalance
- **THEN** the system checks if any class has fewer than 5 samples
- **AND** warns user about severely underrepresented classes
- **AND** recommends collecting more samples before training
- **AND** allows proceeding with warning acknowledgment

#### Scenario: Handle insufficient data
- **WHEN** total dataset has fewer than 50 annotations
- **THEN** the system prevents training initiation
- **AND** displays error message with minimum data requirement
- **AND** shows current annotation count and gap to minimum
- **AND** suggests continuing annotation work

### Requirement: Data Augmentation Strategy
The system SHALL apply appropriate data augmentation to training data with class-specific augmentation factors for imbalanced classes.

#### Scenario: Standard augmentation for balanced classes
- **WHEN** augmenting training data for majority classes
- **THEN** the system applies standard augmentation pipeline:
  - Random rotation (±10°)
  - Random brightness/contrast adjustment (±15%)
  - Random horizontal flip (50% probability)
  - Random Gaussian blur (σ=0.5-1.5)
  - Normalize to ImageNet statistics
- **AND** preserves original image size (1004x1004 for full images, 224x224 for crops)
- **AND** maintains label integrity after transformations

#### Scenario: Heavy augmentation for PASS class
- **WHEN** augmenting training data for PASS class (minority class)
- **THEN** the system applies 20x augmentation factor
- **AND** uses more aggressive augmentation parameters:
  - Random rotation (±15°)
  - Random brightness/contrast (±20%)
  - Random Gaussian noise (σ=0.01-0.03)
  - Elastic transform (α=1, σ=50)
  - Random horizontal flip
  - Random crop and resize
- **AND** generates augmented samples on-the-fly during training
- **AND** ensures augmented samples remain realistic

#### Scenario: Bounding box augmentation for YOLO
- **WHEN** augmenting images for YOLO detection training
- **THEN** the system transforms bounding boxes along with images
- **AND** validates transformed boxes remain within image boundaries
- **AND** filters out boxes that become too small (<10 pixels)
- **AND** maintains YOLO format coordinate normalization

#### Scenario: Augmentation configuration
- **WHEN** configuring augmentation parameters
- **THEN** the system loads parameters from YAML configuration
- **AND** allows per-class augmentation factor specification
- **AND** supports enabling/disabling specific augmentation techniques
- **AND** validates augmentation parameters are within valid ranges

### Requirement: YOLO Detection Model Training
The system SHALL train YOLOv8 models for Wire Loop detection with comprehensive monitoring and checkpointing.

#### Scenario: Initialize YOLO training
- **WHEN** starting YOLO detection training
- **THEN** the system loads YOLOv8n pretrained weights
- **AND** configures training for single-class detection (Wire Loop)
- **AND** sets image size to 1004x1004 pixels
- **AND** configures batch size based on available GPU memory (default: 16)
- **AND** sets number of epochs (default: 100)
- **AND** creates TensorBoard logging directory

#### Scenario: YOLO training execution
- **WHEN** YOLO training is running
- **THEN** the system trains on prepared YOLO format dataset
- **AND** logs loss, mAP@0.5, mAP@0.95 per epoch to TensorBoard
- **AND** saves best model based on mAP@0.5
- **AND** saves checkpoint every 10 epochs
- **AND** updates training progress in GUI
- **AND** displays current epoch, loss, and mAP in real-time

#### Scenario: YOLO early stopping
- **WHEN** validation mAP does not improve for 20 consecutive epochs
- **THEN** the system stops training early
- **AND** loads best model checkpoint
- **AND** logs early stopping trigger in training history
- **AND** displays early stopping message in GUI

#### Scenario: YOLO model evaluation
- **WHEN** YOLO training completes
- **THEN** the system calculates final validation metrics:
  - mAP@0.5
  - mAP@0.5:0.95
  - Precision
  - Recall
- **AND** generates confusion matrix (detections vs ground truth)
- **AND** saves evaluation report as JSON
- **AND** stores metrics in model_versions table

#### Scenario: YOLO training failure recovery
- **WHEN** YOLO training is interrupted (crash, user cancellation)
- **THEN** the system saves current model checkpoint
- **AND** logs interruption reason in training history
- **AND** allows resuming training from last checkpoint
- **AND** preserves training configuration and state

### Requirement: View Angle Classifier Training
The system SHALL train binary CNN classifier for Wire Loop view angle classification (TOP/SIDE).

#### Scenario: Initialize view classifier training
- **WHEN** starting view classifier training
- **THEN** the system loads ResNet18 with pretrained ImageNet weights
- **AND** modifies final layer for 2-class output (TOP/SIDE)
- **AND** configures training hyperparameters:
  - Epochs: 50
  - Batch size: 32
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
- **AND** creates TensorBoard logging directory

#### Scenario: View classifier training execution
- **WHEN** view classifier training is running
- **THEN** the system trains on cropped Wire Loop images (224x224)
- **AND** applies standard data augmentation
- **AND** logs train/val loss and accuracy per epoch to TensorBoard
- **AND** saves best model based on validation accuracy
- **AND** updates training progress in GUI
- **AND** displays current epoch, loss, and accuracy in real-time

#### Scenario: View classifier balanced sampling
- **WHEN** creating training batches for view classifier
- **THEN** the system uses balanced sampling (equal TOP and SIDE per batch)
- **AND** prevents class imbalance affecting training
- **AND** ensures both classes equally represented across epochs

#### Scenario: View classifier early stopping
- **WHEN** validation accuracy does not improve for 15 consecutive epochs
- **THEN** the system stops training early
- **AND** loads best model checkpoint
- **AND** logs early stopping event
- **AND** displays completion message

#### Scenario: View classifier evaluation
- **WHEN** view classifier training completes
- **THEN** the system calculates per-class metrics:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix
- **AND** generates evaluation report
- **AND** stores metrics in model_versions table
- **AND** validates accuracy > 0.95 (target threshold)

### Requirement: Defect Type Classifier Training
The system SHALL train multi-class CNN classifier for defect type classification with class imbalance handling.

#### Scenario: Initialize defect classifier training
- **WHEN** starting defect classifier training
- **THEN** the system loads EfficientNet-B0 with pretrained ImageNet weights
- **AND** modifies final layer for 4-class output (PASS/沖線/晃動/碰觸)
- **AND** calculates class weights based on inverse frequency
- **AND** configures training hyperparameters:
  - Epochs: 100
  - Batch size: 16
  - Learning rate: 0.001
  - Optimizer: AdamW with weight decay
  - Loss: Weighted CrossEntropyLoss or Focal Loss
- **AND** creates TensorBoard logging directory

#### Scenario: Defect classifier weighted loss
- **WHEN** training defect classifier with class imbalance
- **THEN** the system applies weighted cross-entropy loss
- **AND** sets class weights inversely proportional to frequency:
  - PASS: 11.33 (68/6)
  - 沖線: 1.70 (68/40)
  - 晃動: 1.17 (68/58)
  - 碰觸: 1.00 (baseline)
- **AND** configures loss to penalize minority class errors more
- **AND** monitors per-class loss separately in TensorBoard

#### Scenario: Defect classifier heavy augmentation
- **WHEN** generating training batches for defect classifier
- **THEN** the system applies 20x augmentation to PASS class samples
- **AND** applies standard augmentation to other classes
- **AND** ensures PASS class effectively represented in training
- **AND** validates augmented samples maintain label correctness

#### Scenario: Defect classifier stratified sampling
- **WHEN** creating training batches for defect classifier
- **THEN** the system uses stratified batch sampling
- **AND** ensures all 4 defect classes present in each batch
- **AND** oversamples PASS class to match other classes
- **AND** shuffles samples within each batch

#### Scenario: Defect classifier training execution
- **WHEN** defect classifier training is running
- **THEN** the system trains on cropped Wire Loop images (224x224)
- **AND** logs train/val loss and accuracy per epoch to TensorBoard
- **AND** logs per-class precision/recall/F1 every 5 epochs
- **AND** saves best model based on balanced accuracy
- **AND** updates training progress in GUI
- **AND** displays current metrics emphasizing PASS class performance

#### Scenario: Defect classifier early stopping
- **WHEN** validation balanced accuracy does not improve for 20 consecutive epochs
- **THEN** the system stops training early
- **AND** loads best model checkpoint
- **AND** logs early stopping event
- **AND** verifies PASS class recall meets minimum threshold (0.70)

#### Scenario: Defect classifier evaluation
- **WHEN** defect classifier training completes
- **THEN** the system calculates comprehensive metrics:
  - Overall accuracy
  - Balanced accuracy (average of per-class recalls)
  - Per-class precision, recall, F1-score
  - Confusion matrix (4x4)
  - PASS class specific metrics
- **AND** generates evaluation report highlighting PASS class performance
- **AND** stores metrics in model_versions table
- **AND** warns if PASS class recall < 0.70

### Requirement: Training Configuration Management
The system SHALL manage training configurations using YAML files with validation and versioning.

#### Scenario: Load default training configuration
- **WHEN** initializing training without custom configuration
- **THEN** the system loads default YAML configuration for model type
- **AND** uses sensible defaults:
  - YOLO: epochs=100, batch=16, imgsz=1004
  - View: epochs=50, batch=32, lr=0.001
  - Defect: epochs=100, batch=16, lr=0.001, loss=weighted_focal
- **AND** validates all required parameters are present
- **AND** applies GPU/CPU specific adjustments

#### Scenario: Load custom training configuration
- **WHEN** user provides custom configuration file
- **THEN** the system loads YAML configuration
- **AND** validates parameter types and ranges
- **AND** overrides defaults with custom values
- **AND** warns about potentially problematic parameter values
- **AND** logs configuration to TensorBoard

#### Scenario: Validate training configuration
- **WHEN** loading training configuration
- **THEN** the system validates:
  - Epochs > 0 and < 500
  - Batch size > 0 and power of 2
  - Learning rate > 0 and < 1
  - Image size matches dataset (1004 or 224)
  - Loss type is valid (ce, weighted_ce, focal)
- **AND** raises ValidationError for invalid parameters
- **AND** provides clear error messages with valid ranges

#### Scenario: Save configuration with training run
- **WHEN** training starts
- **THEN** the system saves complete configuration to database
- **AND** stores configuration as JSON in training_history table
- **AND** includes derived parameters (class weights, augmentation factors)
- **AND** enables exact reproduction of training run

### Requirement: Model Version Management
The system SHALL manage trained model versions with metadata tracking and activation control.

#### Scenario: Save trained model
- **WHEN** training completes or checkpoint is saved
- **THEN** the system saves model with naming format: `{type}_{timestamp}_{metric}.pt`
- **AND** stores in appropriate directory (models/detection/, models/view/, models/defect/)
- **AND** creates entry in model_versions table with:
  - Model type
  - Filepath
  - Training date/time
  - Final metrics (JSON)
  - Training configuration (JSON)
  - Dataset size used
  - is_active flag (False by default)
- **AND** logs model save event

#### Scenario: List model versions
- **WHEN** querying available models for a model type
- **THEN** the system retrieves all model_versions records
- **AND** orders by training_date descending (newest first)
- **AND** displays model name, metrics, training date
- **AND** highlights currently active model
- **AND** allows filtering by minimum metric threshold

#### Scenario: Set active model
- **WHEN** user selects a model version as active
- **THEN** the system sets is_active=True for selected model
- **AND** sets is_active=False for all other models of same type
- **AND** updates GUI to show new active model
- **AND** uses this model for inference in annotation system
- **AND** logs model activation event

#### Scenario: Compare model versions
- **WHEN** comparing multiple model versions
- **THEN** the system displays side-by-side metrics:
  - Training date
  - Dataset size
  - Final loss
  - Accuracy/mAP
  - Per-class metrics
- **AND** highlights best performing model per metric
- **AND** shows improvement percentage between versions
- **AND** enables visual comparison via charts

#### Scenario: Delete model version
- **WHEN** user deletes a model version
- **THEN** the system checks if model is currently active
- **AND** prevents deletion of active model
- **AND** removes model file from disk
- **AND** deletes entry from model_versions table
- **AND** cascades delete to associated training_history record
- **AND** logs deletion event with reason

### Requirement: TensorBoard Integration
The system SHALL integrate TensorBoard for real-time training visualization and monitoring.

#### Scenario: Initialize TensorBoard logging
- **WHEN** training starts
- **THEN** the system creates TensorBoard SummaryWriter
- **AND** creates log directory: `runs/{model_type}/{timestamp}`
- **AND** logs hyperparameters as text
- **AND** logs model architecture summary
- **AND** logs dataset statistics

#### Scenario: Log scalar metrics during training
- **WHEN** each training epoch completes
- **THEN** the system logs to TensorBoard:
  - Train loss
  - Validation loss
  - Accuracy (or mAP for detection)
  - Learning rate (current)
- **AND** creates separate scalar graphs for each metric
- **AND** enables comparison across training runs

#### Scenario: Log confusion matrix
- **WHEN** validation epoch completes (every 5 epochs for classifiers)
- **THEN** the system calculates confusion matrix
- **AND** logs as image to TensorBoard
- **AND** includes class names on axes
- **AND** highlights diagonal (correct predictions)
- **AND** shows absolute counts and percentages

#### Scenario: Log sample predictions
- **WHEN** validation epoch completes (every 10 epochs)
- **THEN** the system selects random validation samples
- **AND** runs inference and logs predictions with ground truth
- **AND** for detection: draws predicted and true bounding boxes
- **AND** for classification: shows image with predicted and true labels
- **AND** includes confidence scores
- **AND** logs as image grid to TensorBoard

#### Scenario: Launch TensorBoard from GUI
- **WHEN** user clicks "Open TensorBoard" button
- **THEN** the system launches TensorBoard server on port 6006
- **AND** sets logdir to runs/ directory
- **AND** opens TensorBoard URL in default browser
- **AND** displays server status in GUI
- **AND** allows stopping TensorBoard server

### Requirement: Training Progress Monitoring
The system SHALL provide real-time training progress updates with accurate time estimates.

#### Scenario: Display current training progress
- **WHEN** training is in progress
- **THEN** the system updates GUI with:
  - Current epoch / Total epochs
  - Progress percentage
  - Current train loss
  - Current validation metric
  - Estimated time remaining (ETA)
- **AND** updates every epoch completion
- **AND** uses smooth ETA calculation (moving average)

#### Scenario: Calculate training ETA
- **WHEN** training progresses through epochs
- **THEN** the system measures time per epoch
- **AND** calculates moving average of last 5 epochs
- **AND** estimates remaining time: (epochs_left × avg_time_per_epoch)
- **AND** displays in human-readable format (e.g., "12m 34s")
- **AND** updates ETA every epoch

#### Scenario: Pause training
- **WHEN** user clicks "Pause" button during training
- **THEN** the system completes current batch
- **AND** saves checkpoint with training state
- **AND** pauses training loop
- **AND** displays "Paused" status with resume option
- **AND** stops ETA calculation

#### Scenario: Resume training
- **WHEN** user clicks "Resume" button on paused training
- **THEN** the system loads training state from checkpoint
- **AND** resumes from paused epoch
- **AND** continues TensorBoard logging seamlessly
- **AND** recalculates ETA based on new timing
- **AND** logs pause/resume events

#### Scenario: Cancel training
- **WHEN** user clicks "Cancel" button during training
- **THEN** the system prompts for confirmation
- **AND** completes current batch before stopping
- **AND** saves final checkpoint
- **AND** marks training as cancelled in training_history
- **AND** cleans up resources (closes files, frees GPU memory)
- **AND** displays cancellation summary with partial results

### Requirement: GPU and CPU Compatibility
The system SHALL support training on both GPU and CPU with automatic optimization for each.

#### Scenario: Detect available compute device
- **WHEN** training initialization starts
- **THEN** the system checks for CUDA availability
- **AND** selects torch.device('cuda') if available
- **AND** fallbacks to torch.device('cpu') otherwise
- **AND** displays detected device in GUI (e.g., "CUDA: NVIDIA GeForce RTX 3060")
- **AND** logs device information to training history

#### Scenario: Optimize for GPU training
- **WHEN** training on GPU
- **THEN** the system uses configured batch size (e.g., 16 or 32)
- **AND** sets num_workers=4 for data loading
- **AND** enables pin_memory=True for faster GPU transfer
- **AND** uses mixed precision training (fp16) if supported
- **AND** monitors GPU memory usage to prevent OOM errors

#### Scenario: Optimize for CPU training
- **WHEN** training on CPU
- **THEN** the system reduces batch size by 50% (e.g., 16 → 8)
- **AND** sets num_workers=0 to prevent multiprocessing issues
- **AND** disables pin_memory
- **AND** uses float32 precision (no mixed precision)
- **AND** warns user about longer training time (provide estimate)

#### Scenario: Handle GPU out-of-memory error
- **WHEN** GPU out-of-memory error occurs during training
- **THEN** the system catches CUDA OOM exception
- **AND** reduces batch size by 50%
- **AND** clears GPU cache (torch.cuda.empty_cache())
- **AND** retries training with smaller batch size
- **AND** logs batch size reduction event
- **AND** displays warning to user about performance impact

#### Scenario: Fallback to CPU on GPU error
- **WHEN** unrecoverable GPU error occurs
- **THEN** the system switches to CPU device
- **AND** adjusts batch size and num_workers for CPU
- **AND** logs device fallback event
- **AND** displays warning: "GPU error, continuing on CPU"
- **AND** allows training to continue without interruption

### Requirement: Class Imbalance Handling
The system SHALL implement multi-strategy approach to handle severe class imbalance in defect classification.

#### Scenario: Calculate class weights
- **WHEN** preparing defect classifier training
- **THEN** the system calculates class frequencies from training set
- **AND** computes inverse frequency weights
- **AND** normalizes weights so majority class = 1.0
- **AND** applies weights: [11.33, 1.70, 1.17, 1.00] for [PASS, 沖線, 晃動, 碰觸]
- **AND** logs class weights to TensorBoard

#### Scenario: Apply weighted cross-entropy loss
- **WHEN** training defect classifier with weighted loss
- **THEN** the system initializes CrossEntropyLoss with class weights
- **AND** applies weights to loss calculation per batch
- **AND** ensures minority class errors have higher penalty
- **AND** logs weighted loss per class to TensorBoard

#### Scenario: Apply focal loss
- **WHEN** training defect classifier with focal loss option
- **THEN** the system implements focal loss with γ=2.0, α=class_weights
- **AND** focuses learning on hard-to-classify examples
- **AND** down-weights easy examples (especially from majority class)
- **AND** logs focal loss components to TensorBoard

#### Scenario: Oversample minority class
- **WHEN** creating training batches for defect classifier
- **THEN** the system oversamples PASS class by 20x during augmentation
- **AND** ensures PASS class equally represented in batches
- **AND** generates augmented PASS samples on-the-fly
- **AND** tracks effective class distribution after oversampling

#### Scenario: Monitor per-class performance
- **WHEN** evaluating defect classifier during training
- **THEN** the system logs per-class metrics every epoch:
  - Per-class loss
  - Per-class accuracy
  - Per-class precision, recall, F1
- **AND** emphasizes PASS class metrics in logs
- **AND** alerts if PASS class recall drops below 0.70
- **AND** provides class-wise confusion matrix

### Requirement: Training History Tracking
The system SHALL record complete training history in database for reproducibility and analysis.

#### Scenario: Create training run record
- **WHEN** training starts
- **THEN** the system creates entry in training_history table with:
  - Model type
  - Training start timestamp
  - Configuration (JSON)
  - Dataset size (train/val split)
  - Device type (CUDA/CPU)
  - Initial learning rate
  - Status: 'running'
- **AND** returns training_run_id for subsequent updates

#### Scenario: Update training run progress
- **WHEN** each epoch completes
- **THEN** the system updates training_history record with:
  - Current epoch
  - Train loss
  - Validation loss
  - Validation metrics (accuracy/mAP)
  - Updated timestamp
- **AND** stores metrics as JSON array for historical tracking

#### Scenario: Complete training run record
- **WHEN** training completes or is cancelled
- **THEN** the system updates training_history with:
  - Training end timestamp
  - Final metrics
  - Best epoch number
  - Status: 'completed' or 'cancelled'
  - Total training time
  - Final model filepath
- **AND** links to model_versions table entry

#### Scenario: Query training history
- **WHEN** viewing training history
- **THEN** the system retrieves all training runs
- **AND** orders by training_date descending
- **AND** displays summary: model type, date, final metrics, status
- **AND** allows filtering by model type, date range, status
- **AND** enables viewing detailed metrics per epoch

#### Scenario: Training run comparison
- **WHEN** comparing multiple training runs
- **THEN** the system loads metrics from training_history
- **AND** displays loss curves overlaid for comparison
- **AND** shows final metrics side-by-side
- **AND** highlights best performing run per metric
- **AND** enables exporting comparison report

### Requirement: Model Evaluation Metrics
The system SHALL calculate comprehensive evaluation metrics appropriate for each model type.

#### Scenario: Calculate YOLO detection metrics
- **WHEN** evaluating YOLO detection model
- **THEN** the system calculates:
  - mAP@0.5 (primary metric)
  - mAP@0.5:0.95 (COCO metric)
  - Precision at optimal threshold
  - Recall at optimal threshold
  - F1-score
- **AND** generates precision-recall curve
- **AND** logs metrics to TensorBoard
- **AND** stores in model_versions

#### Scenario: Calculate classification accuracy
- **WHEN** evaluating view or defect classifier
- **THEN** the system calculates:
  - Overall accuracy
  - Per-class accuracy
  - Macro-average precision, recall, F1
  - Weighted-average precision, recall, F1
- **AND** generates confusion matrix
- **AND** logs metrics to TensorBoard
- **AND** stores in model_versions

#### Scenario: Calculate balanced accuracy for imbalanced data
- **WHEN** evaluating defect classifier with class imbalance
- **THEN** the system calculates balanced accuracy:
  - Balanced_Acc = mean(per_class_recall)
- **AND** gives equal weight to all classes regardless of frequency
- **AND** emphasizes minority class performance (PASS)
- **AND** reports as primary metric for model selection

#### Scenario: Generate confusion matrix
- **WHEN** calculating classification metrics
- **THEN** the system generates confusion matrix
- **AND** displays absolute counts and percentages
- **AND** highlights diagonal (correct predictions)
- **AND** identifies most common misclassifications
- **AND** visualizes as heatmap in TensorBoard

#### Scenario: Generate evaluation report
- **WHEN** training completes
- **THEN** the system generates JSON evaluation report containing:
  - Model metadata (type, training date, dataset size)
  - Training configuration
  - Final metrics (all calculated metrics)
  - Per-class metrics (for classifiers)
  - Confusion matrix (for classifiers)
  - Training time and device used
- **AND** saves report to models/{type}/{model_name}_eval.json
- **AND** generates human-readable summary text

### Requirement: Integration with Annotation GUI
The system SHALL integrate training functionality seamlessly into existing annotation GUI workflow.

#### Scenario: Access training from menu
- **WHEN** user opens annotation window
- **THEN** the menu bar includes "Tools" → "Train Models" option
- **AND** clicking opens training configuration dialog
- **AND** dialog is non-blocking (can minimize while training)

#### Scenario: Select model type to train
- **WHEN** user opens training dialog
- **THEN** the dialog displays three model type options:
  - "YOLO Detection" - Train Wire Loop detector
  - "View Classifier" - Train TOP/SIDE classifier
  - "Defect Classifier" - Train defect type classifier
- **AND** shows current active model version for each type
- **AND** displays last training date and metrics
- **AND** enables selecting which model(s) to train

#### Scenario: Configure training parameters
- **WHEN** user configures training in dialog
- **THEN** the dialog provides input fields for:
  - Number of epochs
  - Batch size
  - Learning rate
  - Loss type (for defect classifier)
  - Validation split ratio
- **AND** displays current/default values
- **AND** validates input ranges on change
- **AND** shows tooltip help for each parameter

#### Scenario: Start training from GUI
- **WHEN** user clicks "Start Training" button
- **THEN** the system validates sufficient annotated data exists
- **AND** prepares dataset in background thread
- **AND** launches training in separate QThread
- **AND** displays progress bar and status updates
- **AND** keeps GUI responsive during training

#### Scenario: Monitor training progress in GUI
- **WHEN** training is running
- **THEN** the dialog displays:
  - Current epoch / Total epochs progress bar
  - Real-time train loss
  - Real-time validation metrics
  - Estimated time remaining
  - "Open TensorBoard" button
  - "Pause" and "Cancel" buttons
- **AND** updates every epoch completion
- **AND** allows minimizing dialog while training continues

#### Scenario: Use trained model for inference
- **WHEN** training completes successfully
- **THEN** the system prompts "Set as active model for inference?"
- **AND** if accepted, sets new model as active
- **AND** reloads model in annotation system
- **AND** enables semi-automatic annotation with new model
- **AND** logs model activation event

#### Scenario: Select model version for inference
- **WHEN** user opens model selection dropdown in annotation GUI
- **THEN** the system displays list of available model versions:
  - Model name with metrics
  - Training date
  - Active indicator
- **AND** allows selecting different model version
- **AND** reloads selected model for inference
- **AND** updates GUI to show active model

#### Scenario: Retrain with incremental data
- **WHEN** user continues annotating after initial training
- **THEN** the annotation system tracks new annotations
- **AND** displays "X new annotations since last training" message
- **AND** suggests retraining when sufficient new data (e.g., 50+ new annotations)
- **AND** clicking "Retrain" opens training dialog pre-configured
- **AND** enables iterative improvement workflow
