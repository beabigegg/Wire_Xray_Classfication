"""
View classifier trainer for binary classification (TOP/SIDE).

This module implements training for ResNet18-based view classifier.
This is straightforward training with balanced data (86 TOP / 86 SIDE).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from datetime import datetime
from timm import create_model
import cv2
from tqdm import tqdm

from src.training.config_manager import ViewClassifierConfig
from src.training.model_manager import ModelManager
from src.training.training_utils import (
    DeviceManager, MetricsCalculator, EarlyStopping, TrainingMonitor
)
from src.training.augmentation import AugmentationPipeline
from src.training.tensorboard_logger import (
    TensorBoardLogger, plot_confusion_matrix, create_prediction_grid,
    log_model_parameters, log_learning_rate
)
from src.core.database import Database


class ViewDataset(Dataset):
    """
    Dataset for view classification (TOP/SIDE).
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        augmentation: Optional[AugmentationPipeline] = None,
        training: bool = True
    ):
        """
        Initialize dataset.

        Args:
            image_paths: List of paths to cropped images
            labels: List of class labels (0=TOP, 1=SIDE)
            class_names: List of class names in order
            augmentation: Augmentation pipeline
            training: Training mode (applies augmentation)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.augmentation = augmentation
        self.training = training

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        if self.augmentation:
            pipeline = self.augmentation.get_classifier_augmentation(
                training=self.training,
                heavy=False
            )
            aug_result = self.augmentation.augment_image(image, pipeline)
            image = aug_result['image']

        # Convert to tensor (already normalized in augmentation)
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, label


class ViewClassifier:
    """
    ResNet18-based view classifier for binary TOP/SIDE classification.

    This is straightforward binary classification with balanced data.
    """

    def __init__(
        self,
        config: ViewClassifierConfig,
        db: Database,
        models_dir: str = 'models'
    ):
        """
        Initialize view classifier trainer.

        Args:
            config: Training configuration
            db: Database instance
            models_dir: Directory for model storage
        """
        self.config = config
        self.db = db
        self.model_manager = ModelManager(models_dir=models_dir)
        self.device_manager = DeviceManager()

        # Device setup
        self.device = self.device_manager.get_available_device(config.device)
        print(f"Training on device: {self.device}")

        # Class names (fixed order)
        self.class_names = ['TOP', 'SIDE']
        self.num_classes = len(self.class_names)

        # Model
        self.model = None
        self._create_model()

        # Loss function (simple cross-entropy for balanced data)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = None
        self._create_optimizer()

        # Scheduler
        self.scheduler = None
        self._create_scheduler()

        # Augmentation
        self.augmentation = AugmentationPipeline(image_size=224)

        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0

        # TensorBoard
        self.writer = None
        self.tb_logger = None  # TensorBoardLogger helper

        # Visualization data collection
        self.val_predictions = []  # Store (images, true_labels, pred_labels, confidences)
        self.val_confusion_matrix = None

    def _create_model(self):
        """Create ResNet18 model."""
        self.model = create_model(
            self.config.model_name,
            pretrained=True,
            num_classes=self.num_classes
        )
        self.model.to(self.device)

    def _create_optimizer(self):
        """Create optimizer."""
        if self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        else:
            self.scheduler = None

    def create_dataloaders(
        self,
        train_data: Dict[str, List],
        val_data: Dict[str, List]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders.

        Args:
            train_data: Dict with 'image_paths' and 'labels' keys
            val_data: Dict with 'image_paths' and 'labels' keys

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = ViewDataset(
            image_paths=train_data['image_paths'],
            labels=train_data['labels'],
            class_names=self.class_names,
            augmentation=self.augmentation,
            training=True
        )

        val_dataset = ViewDataset(
            image_paths=val_data['image_paths'],
            labels=val_data['labels'],
            class_names=self.class_names,
            augmentation=self.augmentation,
            training=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=(self.device == 'cuda')
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(self.device == 'cuda')
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = running_loss / len(train_loader)
        metrics = MetricsCalculator.calculate_classification_metrics(
            np.array(all_labels),
            np.array(all_preds),
            self.class_names
        )
        metrics['loss'] = avg_loss

        return metrics

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []  # For confidence scores
        sample_images = []  # Store sample images for visualization

        # Clear previous predictions
        self.val_predictions = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.max(dim=1)[0].cpu().numpy())

                # Collect sample images for visualization (first 16 images)
                if len(sample_images) < 16:
                    for i in range(min(len(images), 16 - len(sample_images))):
                        sample_images.append(images[i].cpu())

        # Store predictions for confusion matrix and visualization
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        # Store confusion matrix data
        from sklearn.metrics import confusion_matrix
        self.val_confusion_matrix = confusion_matrix(all_labels_np, all_preds_np)

        # Store sample predictions for visualization
        if sample_images:
            num_samples = min(16, len(sample_images))
            self.val_predictions = [
                (sample_images[i], all_labels_np[i], all_preds_np[i], all_probs_np[i])
                for i in range(num_samples)
            ]

        # Calculate validation metrics
        avg_loss = running_loss / len(val_loader)
        metrics = MetricsCalculator.calculate_classification_metrics(
            all_labels_np,
            all_preds_np,
            self.class_names
        )
        metrics['loss'] = avg_loss

        return metrics

    def log_to_tensorboard(self, metrics: Dict, epoch: int, phase: str):
        """
        Log metrics to TensorBoard with enhanced visualizations.

        Args:
            metrics: Metrics dictionary
            epoch: Current epoch
            phase: 'train' or 'val'
        """
        if self.writer is None or self.tb_logger is None:
            return

        # Scalar metrics
        self.tb_logger.log_scalar(f'{phase}/loss', metrics['loss'], epoch)
        self.tb_logger.log_scalar(f'{phase}/accuracy', metrics['accuracy'], epoch)

        # Per-class metrics (precision, recall, F1)
        for class_name in self.class_names:
            # Precision and Recall
            self.tb_logger.log_scalar(
                f'{phase}/{class_name}_recall',
                metrics[f'{class_name}_recall'],
                epoch
            )
            self.tb_logger.log_scalar(
                f'{phase}/{class_name}_precision',
                metrics[f'{class_name}_precision'],
                epoch
            )

            # F1 score (calculate from precision and recall)
            precision = metrics[f'{class_name}_precision']
            recall = metrics[f'{class_name}_recall']
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            self.tb_logger.log_scalar(f'{phase}/{class_name}_f1', f1, epoch)

        # Learning rate
        if phase == 'train':
            log_learning_rate(self.tb_logger, self.optimizer, epoch)

        # Enhanced visualizations for validation phase
        if phase == 'val':
            # Log confusion matrix every 5 epochs
            if epoch % 5 == 0 and self.val_confusion_matrix is not None:
                cm_figure = plot_confusion_matrix(
                    self.val_confusion_matrix,
                    self.class_names,
                    title=f'Confusion Matrix (Epoch {epoch})',
                    normalize=True
                )
                self.tb_logger.log_figure('val/confusion_matrix', cm_figure, epoch)
                import matplotlib.pyplot as plt
                plt.close(cm_figure)

            # Log sample predictions every 10 epochs
            if self.tb_logger.should_log_images(epoch) and self.val_predictions:
                # Prepare data for prediction grid
                images_list = []
                true_labels_list = []
                pred_labels_list = []
                confidences_list = []

                for img_tensor, true_label, pred_label, confidence in self.val_predictions:
                    images_list.append(img_tensor.numpy())
                    true_labels_list.append(self.class_names[true_label])
                    pred_labels_list.append(self.class_names[pred_label])
                    confidences_list.append(confidence)

                pred_grid = create_prediction_grid(
                    images_list,
                    true_labels_list,
                    pred_labels_list,
                    confidences=confidences_list,
                    max_images=16,
                    grid_cols=4
                )
                self.tb_logger.log_figure('val/predictions', pred_grid, epoch)
                import matplotlib.pyplot as plt
                plt.close(pred_grid)

            # Log model parameter histograms every 10 epochs
            if self.tb_logger.should_log_histograms(epoch):
                log_model_parameters(self.tb_logger, self.model, epoch)

    def train(
        self,
        train_data: Dict[str, List],
        val_data: Dict[str, List],
        output_dir: str = 'runs/view_classifier'
    ) -> str:
        """
        Train view classifier.

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            output_dir: Output directory for logs and checkpoints

        Returns:
            Path to best model checkpoint
        """
        # Setup output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(output_dir) / f'view_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=str(run_dir))
        self.tb_logger = TensorBoardLogger(
            log_dir=str(run_dir),
            image_log_frequency=10,  # Log images every 10 epochs
            histogram_log_frequency=10  # Log histograms every 10 epochs
        )

        # Add training run to database
        config_json = json.dumps(self.config.__dict__)
        run_id = self.db.add_training_run(
            model_type='view',
            config_json=config_json
        )

        # Create data loaders
        print("\nCreating data loaders...")
        train_loader, val_loader = self.create_dataloaders(train_data, val_data)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=0.001,
            mode='max'
        )

        # Training monitor
        monitor = TrainingMonitor(total_epochs=self.config.epochs)
        monitor.start()

        # Training loop
        print(f"\n{'='*70}")
        print(f"Starting View Classifier Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*70}\n")

        best_model_path = None

        try:
            for epoch in range(1, self.config.epochs + 1):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                self.log_to_tensorboard(train_metrics, epoch, 'train')

                # Validate
                val_metrics = self.validate(val_loader, epoch)
                self.log_to_tensorboard(val_metrics, epoch, 'val')

                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['accuracy'])
                    else:
                        self.scheduler.step()

                # Print epoch summary
                print(f"\nEpoch {epoch}/{self.config.epochs} Summary:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  TOP Recall: {val_metrics['TOP_recall']:.4f}, SIDE Recall: {val_metrics['SIDE_recall']:.4f}")
                print(f"  ETA: {monitor.get_eta(epoch)}")

                # Save best model
                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']

                    # Save model
                    save_result = self.model_manager.save_model(
                        model=self.model,
                        model_type='view',
                        model_name=self.config.model_name,
                        metrics={
                            'accuracy': val_metrics['accuracy'],
                            'TOP_recall': val_metrics['TOP_recall'],
                            'SIDE_recall': val_metrics['SIDE_recall'],
                            'TOP_precision': val_metrics['TOP_precision'],
                            'SIDE_precision': val_metrics['SIDE_precision']
                        },
                        config=self.config.__dict__,
                        version_prefix='v1'
                    )

                    best_model_path = save_result['model_path']

                    print(f"  ✓ Best model saved: accuracy={val_metrics['accuracy']:.4f}")

                # Early stopping check
                if early_stopping(val_metrics['accuracy']):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

                monitor.epoch_complete()

            # Training completed successfully
            print(f"\n{'='*70}")
            print(f"Training Completed Successfully!")
            print(f"{'='*70}")
            print(f"Best Accuracy: {self.best_accuracy:.4f}")

            if self.best_accuracy > 0.95:
                print("\n✓ SUCCESS: Model meets target accuracy > 0.95!")
            else:
                print(f"\n⚠ WARNING: Model accuracy {self.best_accuracy:.4f} below target 0.95")

            print(f"{'='*70}\n")

            # Save final best model version to database
            if best_model_path:
                try:
                    # Extract version from path
                    from pathlib import Path
                    model_file = Path(best_model_path)
                    version = model_file.stem

                    model_id = self.db.save_model_version(
                        model_name=self.config.model_name,
                        model_type='view',
                        version=version,
                        filepath=best_model_path,
                        metrics={
                            'accuracy': self.best_accuracy
                        },
                        set_active=True  # Set newly trained model as active
                    )
                    print(f"Model version saved to database (ID: {model_id})")
                except Exception as e:
                    print(f"Warning: Failed to save model version to database: {e}")
                    # Continue even if database save fails

            # Update database
            final_metrics = {
                'best_accuracy': self.best_accuracy
            }
            self.db.update_training_run(
                run_id=run_id,
                status='completed',
                final_metrics_json=json.dumps(final_metrics)
            )

        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self.db.update_training_run(
                run_id=run_id,
                status='failed',
                error_message=str(e)
            )
            raise

        finally:
            if self.writer:
                self.writer.close()

        return best_model_path


# Test if run directly
if __name__ == "__main__":
    print("Testing view classifier...")

    # Create dummy configuration
    config = ViewClassifierConfig(
        epochs=2,
        batch_size=4
    )

    # Create dummy database
    db = Database(':memory:')

    # Create trainer
    trainer = ViewClassifier(config, db)

    print(f"Model: {config.model_name}")
    print(f"Device: {trainer.device}")
    print(f"Class names: {trainer.class_names}")
    print(f"Loss function: {type(trainer.criterion).__name__}")

    print("\nView classifier test complete!")
