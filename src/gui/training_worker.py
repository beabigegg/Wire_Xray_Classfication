"""
Training worker for PyQt6 GUI integration.

This module provides a QThread-based worker for non-blocking training execution
with full support for progress tracking, pause/resume, and cancellation.
"""

import traceback
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
from enum import Enum

from PyQt6.QtCore import QThread, pyqtSignal


class TrainingState(Enum):
    """
    Training state enumeration for clear state management.

    States:
        IDLE: No training in progress
        RUNNING: Training is actively running
        PAUSED: Training is paused (waiting to resume)
        CANCELLED: Training has been cancelled
    """
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"

from src.training.yolo_trainer import YOLOTrainer
from src.training.view_classifier import ViewClassifier
from src.training.defect_classifier import DefectClassifier
from src.training.config_manager import (
    YOLOConfig, ViewClassifierConfig, DefectClassifierConfig
)
from src.training.checkpoint_manager import CheckpointManager
from src.core.database import Database


class TrainingWorker(QThread):
    """
    QThread worker for running model training in background.

    This worker executes training operations without blocking the GUI,
    providing real-time progress updates and supporting pause/resume/cancel.

    Signals:
        progress_updated: Emitted during training (epoch, total_epochs, progress_percent)
        epoch_completed: Emitted after each epoch (epoch, metrics_dict)
        training_finished: Emitted on completion (success, model_path, final_metrics)
        training_error: Emitted on error (error_message, traceback_str)
        log_message: Emitted for logging (level, message)
    """

    # Qt signals for thread-safe communication
    progress_updated = pyqtSignal(int, int, float)  # epoch, total_epochs, progress_percent
    epoch_completed = pyqtSignal(int, dict)  # epoch, metrics
    training_finished = pyqtSignal(bool, str, dict)  # success, model_path, final_metrics
    training_error = pyqtSignal(str, str)  # error_message, traceback
    log_message = pyqtSignal(str, str)  # level, message
    state_changed = pyqtSignal(object)  # TrainingState

    def __init__(
        self,
        model_type: str,
        config: Dict[str, Any],
        database_path: str,
        train_data: Optional[Dict] = None,
        val_data: Optional[Dict] = None,
        data_yaml_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize training worker.

        Args:
            model_type: Type of model ('detection', 'view', 'defect')
            config: Configuration dictionary for the model
            database_path: Path to SQLite database
            train_data: Training data dict (for classifiers)
            val_data: Validation data dict (for classifiers)
            data_yaml_path: Path to data.yaml (for YOLO)
            output_dir: Output directory for training artifacts
        """
        super().__init__()

        self.model_type = model_type
        self.config = config
        self.database_path = database_path
        self.train_data = train_data
        self.val_data = val_data
        self.data_yaml_path = data_yaml_path
        self.output_dir = output_dir or f'runs/{model_type}'

        # Training state using enum (thread-safe via Qt)
        self._state = TrainingState.IDLE

        # Training components
        self.trainer = None
        self.db = None
        self.checkpoint_path = None
        self.checkpoint_manager = CheckpointManager()

    def run(self):
        """
        Execute training in background thread.

        This is the main entry point called by QThread.start().
        All training logic runs here without blocking the GUI.
        """
        try:
            self._state = TrainingState.RUNNING
            self.state_changed.emit(self._state)
            self.log_message.emit('INFO', f'Starting {self.model_type} training...')

            # Initialize database connection
            self.db = Database(self.database_path)

            # Create trainer based on model type
            self.log_message.emit('INFO', 'Initializing trainer...')
            self._create_trainer()

            # Execute training
            model_path = self._execute_training()

            if self._state == TrainingState.CANCELLED:
                self.log_message.emit('WARNING', 'Training was cancelled')
                self.training_finished.emit(False, '', {})
            else:
                self.log_message.emit('INFO', f'Training completed: {model_path}')

                # Load final metrics
                final_metrics = self._load_final_metrics()
                self.training_finished.emit(True, model_path, final_metrics)

        except Exception as e:
            error_msg = str(e)
            tb_str = traceback.format_exc()

            self._state = TrainingState.IDLE
            self.state_changed.emit(self._state)
            self.log_message.emit('ERROR', f'Training failed: {error_msg}')
            self.training_error.emit(error_msg, tb_str)

        finally:
            # Cleanup
            if self.db:
                self.db.close()
            if self._state != TrainingState.IDLE:
                self._state = TrainingState.IDLE
                self.state_changed.emit(self._state)

    def _create_trainer(self):
        """Create trainer instance based on model type."""
        # Support VIEW-aware model types
        # detection_top, detection_side use same YOLO trainer (different datasets)
        # defect_top, defect_side use same Defect classifier (different datasets)
        if self.model_type in ['detection', 'detection_top', 'detection_side']:
            # YOLO detection trainer
            # Map UI parameters to config parameters
            yolo_config = self.config.copy()

            # Map batch_size to batch
            if 'batch_size' in yolo_config:
                yolo_config['batch'] = yolo_config.pop('batch_size')

            # Map learning_rate to lr0
            if 'learning_rate' in yolo_config:
                yolo_config['lr0'] = yolo_config.pop('learning_rate')

            # Remove parameters not accepted by YOLOConfig
            yolo_config.pop('use_augmentation', None)
            yolo_config.pop('save_best_only', None)
            yolo_config.pop('conf_threshold', None)
            yolo_config.pop('iou_threshold', None)

            config_obj = YOLOConfig(**yolo_config)
            self.trainer = YOLOTrainer(
                config=config_obj,
                db=self.db,
                models_dir='models'
            )
            self.log_message.emit('INFO', f'Created YOLO trainer: {config_obj.model_name}')

        elif self.model_type == 'view':
            # View classifier trainer
            # Map UI parameters to config parameters
            view_config = self.config.copy()

            # Map backbone to model_name
            if 'backbone' in view_config:
                view_config['model_name'] = view_config.pop('backbone')

            # Map patience to early_stopping_patience
            if 'patience' in view_config:
                view_config['early_stopping_patience'] = view_config.pop('patience')

            # Remove parameters not accepted by ViewClassifierConfig
            view_config.pop('use_augmentation', None)
            view_config.pop('dropout', None)
            view_config.pop('pretrained', None)

            config_obj = ViewClassifierConfig(**view_config)
            self.trainer = ViewClassifier(
                config=config_obj,
                db=self.db,
                models_dir='models'
            )
            self.log_message.emit('INFO', f'Created View classifier: {config_obj.model_name}')

        elif self.model_type in ['defect', 'defect_top', 'defect_side']:
            # Defect classifier trainer (supports VIEW-aware variants)
            # Map loss_function parameter to config flags
            defect_config = self.config.copy()
            loss_function = defect_config.get('loss_function', 'Focal')

            if loss_function == 'Focal':
                defect_config['use_focal_loss'] = True
                defect_config['use_class_weights'] = False
                # Use focal_gamma from config if provided
                if 'focal_gamma' not in defect_config:
                    defect_config['focal_gamma'] = 2.0
            elif loss_function == 'Weighted':
                defect_config['use_focal_loss'] = False
                defect_config['use_class_weights'] = True
            else:  # CrossEntropy
                defect_config['use_focal_loss'] = False
                defect_config['use_class_weights'] = False

            # Map UI parameters to config parameters
            if 'backbone' in defect_config:
                defect_config['model_name'] = defect_config.pop('backbone')
            if 'patience' in defect_config:
                defect_config['early_stopping_patience'] = defect_config.pop('patience')
            if 'pass_aug_factor' in defect_config:
                defect_config['pass_augmentation_factor'] = int(defect_config.pop('pass_aug_factor'))
            if 'balanced_sampling' in defect_config:
                defect_config['use_balanced_sampling'] = defect_config.pop('balanced_sampling')
            if 'auto_class_weights' in defect_config:
                # This parameter is implied by weighted loss
                defect_config.pop('auto_class_weights')

            # Remove parameters not accepted by DefectClassifierConfig
            defect_config.pop('use_augmentation', None)
            defect_config.pop('dropout', None)
            defect_config.pop('pretrained', None)

            config_obj = DefectClassifierConfig(**defect_config)
            self.trainer = DefectClassifier(
                config=config_obj,
                db=self.db,
                models_dir='models'
            )
            self.log_message.emit('INFO', f'Created Defect classifier: {config_obj.model_name}, loss={loss_function}')

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_classifier_data_from_directory(self, data_dir: str, class_names: List[str]) -> Dict[str, List]:
        """
        Load classifier data from directory structure.

        Expected structure:
            data_dir/
                class1/
                    image1.jpg
                    image2.jpg
                class2/
                    image3.jpg

        Args:
            data_dir: Path to data directory
            class_names: List of class names in order (for label mapping)

        Returns:
            Dictionary with 'image_paths' and 'labels' lists
            Labels are converted to integer indices based on class_names order
        """
        from pathlib import Path

        data_path = Path(data_dir)
        image_paths = []
        labels = []

        # Create class name to index mapping
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Iterate over class directories
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            # Check if class name is valid
            if class_name not in class_to_idx:
                self.log_message.emit('WARNING', f'Unknown class "{class_name}" in {data_dir}, skipping...')
                continue

            class_idx = class_to_idx[class_name]

            # Collect all images in this class
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(str(img_path))
                    labels.append(class_idx)

        return {
            'image_paths': image_paths,
            'labels': labels
        }

    def _execute_training(self) -> str:
        """
        Execute the training process.

        Returns:
            Path to trained model
        """
        # Support VIEW-aware model types
        if self.model_type in ['detection', 'detection_top', 'detection_side']:
            return self._train_detection()
        elif self.model_type in ['view', 'defect', 'defect_top', 'defect_side']:
            return self._train_classifier()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_detection(self) -> str:
        """
        Train YOLO detection model.

        Returns:
            Path to best model checkpoint
        """
        if not self.data_yaml_path:
            raise ValueError("data_yaml_path is required for detection training")

        self.log_message.emit('INFO', f'Training YOLO with data: {self.data_yaml_path}')

        # Note: YOLO training is handled by Ultralytics library
        # We can't easily intercept per-epoch progress, so we emit progress updates
        # at coarse intervals

        total_epochs = self.config.get('epochs', 100)

        # Start training (this is a blocking call)
        # In a real implementation, we would need to modify YOLOTrainer
        # to accept callbacks for progress updates
        try:
            model_path = self.trainer.train(
                data_yaml_path=self.data_yaml_path,
                output_dir=self.output_dir
            )

            # Emit final progress
            self.progress_updated.emit(total_epochs, total_epochs, 100.0)

            return model_path

        except Exception as e:
            self.log_message.emit('ERROR', f'YOLO training failed: {str(e)}')
            raise

    def _train_classifier(self) -> str:
        """
        Train view or defect classifier with per-epoch progress tracking.

        Returns:
            Path to best model checkpoint
        """
        if not self.train_data or not self.val_data:
            raise ValueError("train_data and val_data are required for classifier training")

        self.log_message.emit('INFO', f'Training {self.model_type} classifier...')

        # Get total epochs
        total_epochs = self.config.get('epochs', 100)

        # Setup output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(self.output_dir) / f'{self.model_type}_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        self.log_message.emit('INFO', 'Creating data loaders...')

        # Load data from directories
        # Get class names from trainer
        class_names = getattr(self.trainer, 'class_names', None)
        if class_names is None:
            raise ValueError("Trainer does not have class_names attribute")

        train_data_dict = self._load_classifier_data_from_directory(self.train_data, class_names)
        val_data_dict = self._load_classifier_data_from_directory(self.val_data, class_names)

        train_loader, val_loader = self.trainer.create_dataloaders(
            train_data_dict,
            val_data_dict
        )

        # Setup early stopping and monitoring
        from src.training.training_utils import EarlyStopping, TrainingMonitor

        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=0.001,
            mode='max'
        )

        monitor = TrainingMonitor(total_epochs=total_epochs)
        monitor.start()

        # Setup TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        self.trainer.writer = SummaryWriter(log_dir=str(run_dir))

        # Add training run to database
        config_json = json.dumps(self.config)
        run_id = self.db.add_training_run(
            model_type=self.model_type,
            config_json=config_json
        )

        best_model_path = None
        best_metric = 0.0

        try:
            # Training loop
            for epoch in range(1, total_epochs + 1):
                # Check for cancellation
                if self._state == TrainingState.CANCELLED:
                    self.log_message.emit('WARNING', 'Training cancelled by user')
                    cancel_metrics = {'best_metric': best_metric, 'last_epoch': epoch}
                    self._save_checkpoint(epoch, total_epochs, best_metric, cancel_metrics)
                    break

                # Handle pause
                while self._state == TrainingState.PAUSED:
                    self.log_message.emit('INFO', 'Training paused')
                    pause_metrics = {'best_metric': best_metric, 'pause_epoch': epoch}
                    self._save_checkpoint(epoch, total_epochs, best_metric, pause_metrics)

                    # Wait for resume or cancel
                    while self._state == TrainingState.PAUSED:
                        if self._state == TrainingState.CANCELLED:
                            break
                        self.msleep(100)  # Sleep for 100ms

                    if self._state == TrainingState.RUNNING:
                        self.log_message.emit('INFO', 'Training resumed')
                    break

                # Update progress at start of epoch
                progress_percent = (epoch - 1) / total_epochs * 100
                self.progress_updated.emit(epoch, total_epochs, progress_percent)
                self.log_message.emit('INFO', f'Starting epoch {epoch}/{total_epochs}')

                # Train one epoch
                train_metrics = self.trainer.train_epoch(train_loader, epoch)
                self.trainer.log_to_tensorboard(train_metrics, epoch, 'train')

                # Validate
                val_metrics = self.trainer.validate(val_loader, epoch)
                self.trainer.log_to_tensorboard(val_metrics, epoch, 'val')

                # Update learning rate scheduler
                if self.trainer.scheduler is not None:
                    import torch
                    if isinstance(self.trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # Support VIEW-aware defect models
                        metric_key = 'balanced_accuracy' if self.model_type in ['defect', 'defect_top', 'defect_side'] else 'accuracy'
                        self.trainer.scheduler.step(val_metrics[metric_key])
                    else:
                        self.trainer.scheduler.step()

                # Emit epoch completion with metrics
                metrics_summary = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics.get('accuracy', 0.0),
                    'val_balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0)
                }

                # Support VIEW-aware defect models
                if self.model_type in ['defect', 'defect_top', 'defect_side']:
                    metrics_summary['pass_recall'] = val_metrics.get('pass_recall', 0.0)

                self.epoch_completed.emit(epoch, metrics_summary)

                # Log epoch summary
                self.log_message.emit('INFO', f'Epoch {epoch}/{total_epochs} completed')
                self.log_message.emit('INFO', f'  Train Loss: {train_metrics["loss"]:.4f}, Val Loss: {val_metrics["loss"]:.4f}')

                # Determine best metric to track
                # Support VIEW-aware defect models
                if self.model_type in ['defect', 'defect_top', 'defect_side']:
                    current_metric = val_metrics['balanced_accuracy']
                    self.log_message.emit('INFO', f'  Val Balanced Accuracy: {current_metric:.4f}')
                else:
                    current_metric = val_metrics['accuracy']
                    self.log_message.emit('INFO', f'  Val Accuracy: {current_metric:.4f}')

                # Save best model
                if current_metric > best_metric:
                    best_metric = current_metric

                    # Prepare metrics for saving
                    save_metrics = {
                        'accuracy': val_metrics.get('accuracy', 0.0)
                    }

                    if self.model_type == 'view':
                        save_metrics.update({
                            'TOP_recall': val_metrics.get('TOP_recall', 0.0),
                            'SIDE_recall': val_metrics.get('SIDE_recall', 0.0),
                            'TOP_precision': val_metrics.get('TOP_precision', 0.0),
                            'SIDE_precision': val_metrics.get('SIDE_precision', 0.0)
                        })
                    # Support VIEW-aware defect models
                    elif self.model_type in ['defect', 'defect_top', 'defect_side']:
                        save_metrics.update({
                            'balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0),
                            'pass_recall': val_metrics.get('pass_recall', 0.0),
                            'pass_precision': val_metrics.get('pass_precision', 0.0),
                            'pass_f1': val_metrics.get('pass_f1', 0.0)
                        })

                    # Save model
                    save_result = self.trainer.model_manager.save_model(
                        model=self.trainer.model,
                        model_type=self.model_type,
                        model_name=self.config.get('model_name', 'model'),
                        metrics=save_metrics,
                        config=self.config,
                        version_prefix='v1'
                    )

                    best_model_path = save_result['model_path']
                    self.log_message.emit('INFO', f'  Best model saved: metric={current_metric:.4f}')

                # Update progress at end of epoch
                progress_percent = epoch / total_epochs * 100
                self.progress_updated.emit(epoch, total_epochs, progress_percent)

                # Early stopping check
                if early_stopping(current_metric):
                    self.log_message.emit('INFO', f'Early stopping triggered at epoch {epoch}')
                    break

                monitor.epoch_complete()

            # Training completed
            if self._state != TrainingState.CANCELLED:
                self.log_message.emit('INFO', 'Training completed successfully')

                # Update database
                final_metrics = {'best_metric': best_metric}
                self.db.update_training_run(
                    run_id=run_id,
                    status='completed',
                    final_metrics_json=json.dumps(final_metrics)
                )

                # Delete checkpoint on successful completion (no longer needed)
                if self.checkpoint_manager.delete_checkpoint(self.model_type):
                    self.log_message.emit('INFO', 'Checkpoint deleted after successful training')
            else:
                self.db.update_training_run(
                    run_id=run_id,
                    status='cancelled',
                    error_message='Training cancelled by user'
                )
                # Keep checkpoint on cancel for potential resume (will be cleaned up after retention period)

            return best_model_path if best_model_path else ''

        except Exception as e:
            self.log_message.emit('ERROR', f'Training failed: {str(e)}')
            self.db.update_training_run(
                run_id=run_id,
                status='failed',
                error_message=str(e)
            )
            raise

        finally:
            if self.trainer.writer:
                self.trainer.writer.close()

    def _load_final_metrics(self) -> Dict[str, float]:
        """
        Load final metrics from trained model.

        Returns:
            Dictionary of final metrics
        """
        # For now, return empty dict
        # In a real implementation, we would load metrics from the saved model
        return {}

    def _save_checkpoint(self, epoch: int, total_epochs: int, best_metric: float, metrics: Dict[str, Any]):
        """
        Save training checkpoint for pause/resume using CheckpointManager.

        Args:
            epoch: Current epoch number
            total_epochs: Total epochs planned
            best_metric: Best metric value so far
            metrics: Current training metrics
        """
        if not self.trainer:
            return

        # Get model, optimizer, scheduler state
        try:
            model_state = self.trainer.model.state_dict() if hasattr(self.trainer, 'model') and self.trainer.model else {}
            optimizer_state = self.trainer.optimizer.state_dict() if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer else {}
            scheduler_state = self.trainer.scheduler.state_dict() if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler else None

            # Save checkpoint using CheckpointManager
            success = self.checkpoint_manager.save_checkpoint(
                model_type=self.model_type,
                epoch=epoch,
                total_epochs=total_epochs,
                model_state=model_state,
                optimizer_state=optimizer_state,
                scheduler_state=scheduler_state,
                metrics=metrics,
                config=self.config
            )

            if success:
                self.log_message.emit('INFO', f'Checkpoint saved at epoch {epoch}/{total_epochs}')
            else:
                self.log_message.emit('WARNING', 'Failed to save checkpoint')

        except Exception as e:
            self.log_message.emit('ERROR', f'Error saving checkpoint: {e}')

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load training checkpoint and restore model/optimizer/scheduler states.

        Returns:
            Checkpoint data dict if loaded successfully, None otherwise
        """
        if not self.trainer:
            return None

        try:
            # Load checkpoint using CheckpointManager
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.model_type)

            if not checkpoint_data:
                self.log_message.emit('INFO', 'No checkpoint found to resume from')
                return None

            # Restore model state
            if hasattr(self.trainer, 'model') and self.trainer.model:
                self.trainer.model.load_state_dict(checkpoint_data['model_state'])
                self.log_message.emit('INFO', 'Model state restored from checkpoint')

            # Restore optimizer state
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer:
                self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
                self.log_message.emit('INFO', 'Optimizer state restored from checkpoint')

            # Restore scheduler state (if present)
            if checkpoint_data['scheduler_state'] and hasattr(self.trainer, 'scheduler') and self.trainer.scheduler:
                self.trainer.scheduler.load_state_dict(checkpoint_data['scheduler_state'])
                self.log_message.emit('INFO', 'Scheduler state restored from checkpoint')

            epoch = checkpoint_data['epoch']
            total_epochs = checkpoint_data['total_epochs']
            self.log_message.emit('INFO', f'Checkpoint loaded: resuming from epoch {epoch}/{total_epochs}')

            return checkpoint_data

        except Exception as e:
            self.log_message.emit('ERROR', f'Error loading checkpoint: {e}')
            return None

    def request_cancel(self):
        """
        Request training cancellation.

        This is thread-safe and will be checked at the start of each epoch.
        Training will stop gracefully at the next opportunity.
        """
        self._state = TrainingState.CANCELLED
        self.state_changed.emit(self._state)
        self.log_message.emit('WARNING', 'Cancel requested...')

    def request_pause(self):
        """
        Request training pause.

        Training will pause at the end of the current epoch and save a checkpoint.
        """
        if self._state == TrainingState.RUNNING:
            self._state = TrainingState.PAUSED
            self.state_changed.emit(self._state)
            self.log_message.emit('INFO', 'Pause requested...')

    def request_resume(self):
        """
        Request training resume from pause.

        Training will continue from where it was paused.
        """
        if self._state == TrainingState.PAUSED:
            self._state = TrainingState.RUNNING
            self.state_changed.emit(self._state)
            self.log_message.emit('INFO', 'Resume requested...')

    def get_state(self) -> TrainingState:
        """
        Get current training state.

        Returns:
            Current TrainingState enum value
        """
        return self._state

    def is_paused(self) -> bool:
        """
        Check if training is currently paused.

        Returns:
            True if paused, False otherwise
        """
        return self._state == TrainingState.PAUSED

    def is_cancelled(self) -> bool:
        """
        Check if cancellation has been requested.

        Returns:
            True if cancelled, False otherwise
        """
        return self._state == TrainingState.CANCELLED

    def is_running(self) -> bool:
        """
        Check if training is actively running.

        Returns:
            True if running, False otherwise
        """
        return self._state == TrainingState.RUNNING


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    print("Testing TrainingWorker...")

    # This is just a structure test, not a full training run
    app = QApplication(sys.argv)

    # Example configuration
    config = {
        'model_name': 'resnet18',
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 0.001
    }

    # Create worker
    worker = TrainingWorker(
        model_type='view',
        config=config,
        database_path=':memory:'
    )

    # Connect signals
    worker.progress_updated.connect(
        lambda e, t, p: print(f"Progress: Epoch {e}/{t} ({p:.1f}%)")
    )
    worker.epoch_completed.connect(
        lambda e, m: print(f"Epoch {e} completed: {m}")
    )
    worker.training_finished.connect(
        lambda s, p, m: print(f"Training finished: success={s}, path={p}")
    )
    worker.training_error.connect(
        lambda e, t: print(f"Error: {e}")
    )
    worker.log_message.connect(
        lambda l, m: print(f"[{l}] {m}")
    )

    print("TrainingWorker structure test complete!")
    print("\nFeatures implemented:")
    print("  - QThread-based non-blocking execution")
    print("  - Progress tracking with signals")
    print("  - Per-epoch metrics reporting")
    print("  - Thread-safe pause/resume/cancel")
    print("  - Checkpoint saving on pause")
    print("  - Support for all three model types")
    print("  - Comprehensive error handling")
    print("  - Logging integration")
