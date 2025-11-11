"""
Training Dialog UI for model training configuration and monitoring.

This module provides a comprehensive training dialog that:
- Configures training parameters for different model types
- Monitors training progress in real-time
- Integrates with TrainingWorker QThread
- Manages TensorBoard visualization
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QCheckBox, QTableWidget,
    QTableWidgetItem, QMessageBox, QWidget, QFormLayout,
    QSizePolicy, QFileDialog, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

try:
    import yaml
except ImportError:
    yaml = None

from src.gui.tensorboard_manager import TensorBoardManager
from src.gui.training_worker import TrainingWorker
from src.training.dataset_analyzer import DatasetAnalyzer
from src.training.data_preparation import DataPreparator
from src.training.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class TrainingDialog(QDialog):
    """
    Comprehensive training dialog for model training.

    Features:
    - Model type selection
    - Training parameter configuration
    - Real-time progress monitoring
    - Metrics display
    - Training log viewer
    - TensorBoard integration
    """

    def __init__(self, db_path: str, config_dir: str = "configs", data_dir: str = "data", parent=None):
        """
        Initialize training dialog.

        Args:
            db_path: Path to annotations database
            config_dir: Directory containing training config YAML files
            data_dir: Directory containing training data
            parent: Parent widget (optional)
        """
        super().__init__(parent)

        self.db_path = db_path
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)

        # State
        self.worker: Optional[TrainingWorker] = None
        self.tensorboard_manager = TensorBoardManager()
        self.checkpoint_manager = CheckpointManager()
        self.is_training = False
        self.current_config: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.resume_checkpoint = None  # Checkpoint data if resuming

        # Batch training state
        self.is_batch_training = False
        self.training_queue: List[str] = []  # Queue of model types to train
        self.current_queue_index = 0
        self.batch_training_configs: Dict[str, Dict] = {}  # Configs for each model type

        # Setup UI
        self.setWindowTitle("Model Training")
        self.setMinimumSize(1000, 750)
        self.resize(1000, 800)  # Default size
        self.setModal(True)
        # Allow window resizing
        self.setSizeGripEnabled(True)

        self._create_ui()
        self._load_dataset_statistics()
        self._check_for_checkpoint()  # Check for existing checkpoints
        self._update_button_states()

    def _create_ui(self):
        """Create main UI layout."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Content widget inside scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(10)

        # Configuration section
        config_group = self._create_configuration_section()
        content_layout.addWidget(config_group)

        # Dataset statistics
        stats_group = self._create_statistics_section()
        content_layout.addWidget(stats_group)

        # Progress section
        progress_group = self._create_progress_section()
        content_layout.addWidget(progress_group, stretch=1)

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area, stretch=1)

        # Control buttons (always visible at bottom)
        buttons_layout = self._create_control_buttons()
        main_layout.addLayout(buttons_layout)

    def _create_configuration_section(self) -> QGroupBox:
        """Create training configuration section."""
        group = QGroupBox("Training Configuration")
        main_layout = QVBoxLayout()

        # Config management buttons at the top
        config_buttons_layout = QHBoxLayout()
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.setToolTip("Save current training configuration to file")
        self.save_config_button.clicked.connect(self._on_save_config)
        config_buttons_layout.addWidget(self.save_config_button)

        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.setToolTip("Load training configuration from file")
        self.load_config_button.clicked.connect(self._on_load_config)
        config_buttons_layout.addWidget(self.load_config_button)

        config_buttons_layout.addStretch()
        main_layout.addLayout(config_buttons_layout)

        # Main configuration form
        layout = QFormLayout()
        layout.setSpacing(8)

        # Model type selector
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "View Classifier",
            "Detection Model (YOLO) - TOP",
            "Detection Model (YOLO) - SIDE",
            "Defect Classifier - TOP",
            "Defect Classifier - SIDE",
            "--- Legacy (Unified Models) ---",
            "Detection Model (YOLO) - Unified",
            "Defect Classifier - Unified"
        ])
        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        self.model_type_combo.setToolTip(
            "Select model type to train:\n"
            "• View Classifier: Classifies TOP/SIDE views (uses full image)\n"
            "• Detection/Defect (TOP/SIDE): VIEW-aware models (recommended)\n"
            "• Unified: Old architecture (both views in one model)"
        )
        layout.addRow("Model Type:", self.model_type_combo)

        # Training parameters with tooltips
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setToolTip(
            "Number of training epochs.\n"
            "More epochs = better learning, but slower training.\n"
            "Training will stop early if no improvement is detected."
        )
        layout.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setToolTip(
            "Number of images processed together.\n"
            "Larger batch = faster training but more GPU memory.\n"
            "If GPU out of memory, reduce this value (try 8 or 4)."
        )
        layout.addRow("Batch Size:", self.batch_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00001, 0.1)
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setValue(0.01)
        self.learning_rate_spin.setToolTip(
            "Learning rate controls how fast the model learns.\n"
            "Higher = faster learning but may be unstable.\n"
            "Lower = slower but more stable learning.\n"
            "Default values are usually good."
        )
        layout.addRow("Learning Rate:", self.learning_rate_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.device_combo.setToolTip(
            "Select training device:\n"
            "• Auto: Automatically use GPU if available, else CPU\n"
            "• CUDA: Force GPU (fastest, requires NVIDIA GPU)\n"
            "• CPU: Force CPU (slower, works on all systems)"
        )
        layout.addRow("Device:", self.device_combo)

        # Advanced options (collapsible)
        self.advanced_group = QGroupBox("Advanced Options")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QFormLayout()

        # Common parameters
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(20)
        self.patience_spin.setToolTip(
            "Number of epochs to wait for improvement before stopping.\n"
            "If validation loss doesn't improve for this many epochs,\n"
            "training will stop automatically to prevent overfitting."
        )
        advanced_layout.addRow("Early Stopping Patience:", self.patience_spin)

        self.save_best_checkbox = QCheckBox()
        self.save_best_checkbox.setChecked(True)
        self.save_best_checkbox.setToolTip(
            "Only save the model when it achieves the best validation performance.\n"
            "Recommended: Keep this enabled to save disk space."
        )
        advanced_layout.addRow("Save Best Only:", self.save_best_checkbox)

        self.augmentation_checkbox = QCheckBox()
        self.augmentation_checkbox.setChecked(True)
        self.augmentation_checkbox.setToolTip(
            "Apply random transformations to training images:\n"
            "• Random rotation, flip, brightness/contrast changes\n"
            "• Helps model generalize better\n"
            "Recommended: Keep this enabled for better results."
        )
        advanced_layout.addRow("Use Data Augmentation:", self.augmentation_checkbox)

        self.preserve_wire_pairs_checkbox = QCheckBox()
        self.preserve_wire_pairs_checkbox.setChecked(True)
        self.preserve_wire_pairs_checkbox.setToolTip(
            "Preserve TOP/SIDE wire pairs in train/val split:\n"
            "• Ensures same wire's TOP and SIDE stay in same set\n"
            "• Prevents data leakage during training\n"
            "• Required for proper paired inference\n"
            "Recommended: Keep this enabled for wire datasets."
        )
        advanced_layout.addRow("Preserve Wire Pairs:", self.preserve_wire_pairs_checkbox)

        self.advanced_group.setLayout(advanced_layout)
        layout.addRow(self.advanced_group)

        # Model-specific advanced parameters
        self.model_specific_group = QGroupBox("Model-Specific Parameters")
        self.model_specific_group.setCheckable(True)
        self.model_specific_group.setChecked(False)
        self.model_specific_layout = QFormLayout()

        # YOLO-specific parameters
        self._create_yolo_params()

        # View Classifier parameters
        self._create_view_classifier_params()

        # Defect Classifier parameters
        self._create_defect_classifier_params()

        self.model_specific_group.setLayout(self.model_specific_layout)
        layout.addRow(self.model_specific_group)

        # Initially show appropriate parameters
        self._update_model_specific_visibility()

        # Add the form layout to main layout
        main_layout.addLayout(layout)
        group.setLayout(main_layout)

        # Load default config
        self._load_default_config()

        return group

    def _create_yolo_params(self):
        """Create YOLO-specific parameters."""
        # Model name (base model selection)
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.yolo_model_combo.setCurrentText("yolov8m")
        self.yolo_model_combo.setToolTip(
            "YOLO base model selection:\n"
            "• yolov8n: Nano - fastest, smallest (3M params)\n"
            "• yolov8s: Small - good balance (11M params)\n"
            "• yolov8m: Medium - recommended (26M params)\n"
            "• yolov8l: Large - high accuracy (44M params)\n"
            "• yolov8x: Extra large - best accuracy (68M params)"
        )
        self.yolo_model_label = QLabel("Base Model:")
        self.model_specific_layout.addRow(self.yolo_model_label, self.yolo_model_combo)

        # Image size
        self.yolo_imgsz_spin = QSpinBox()
        self.yolo_imgsz_spin.setRange(320, 1280)
        self.yolo_imgsz_spin.setSingleStep(32)
        self.yolo_imgsz_spin.setValue(640)
        self.yolo_imgsz_spin.setToolTip(
            "Input image size for training (must be multiple of 32).\n"
            "Larger = better accuracy but slower and more memory.\n"
            "640 is a good balance for most cases."
        )
        self.yolo_imgsz_label = QLabel("Image Size:")
        self.model_specific_layout.addRow(self.yolo_imgsz_label, self.yolo_imgsz_spin)

        # Optimizer
        self.yolo_optimizer_combo = QComboBox()
        self.yolo_optimizer_combo.addItems(["SGD", "Adam", "AdamW"])
        self.yolo_optimizer_combo.setCurrentText("SGD")
        self.yolo_optimizer_combo.setToolTip(
            "Optimizer algorithm:\n"
            "• SGD: Classic, stable, good for most cases\n"
            "• Adam: Adaptive learning rate, faster convergence\n"
            "• AdamW: Adam with weight decay, better regularization"
        )
        self.yolo_optimizer_label = QLabel("Optimizer:")
        self.model_specific_layout.addRow(self.yolo_optimizer_label, self.yolo_optimizer_combo)

        # Warmup epochs
        self.yolo_warmup_spin = QSpinBox()
        self.yolo_warmup_spin.setRange(0, 10)
        self.yolo_warmup_spin.setValue(3)
        self.yolo_warmup_spin.setToolTip(
            "Number of warmup epochs.\n"
            "Gradually increases learning rate from 0 to target value.\n"
            "Helps stabilize training in early stages."
        )
        self.yolo_warmup_label = QLabel("Warmup Epochs:")
        self.model_specific_layout.addRow(self.yolo_warmup_label, self.yolo_warmup_spin)

        # Confidence threshold
        self.yolo_conf_spin = QDoubleSpinBox()
        self.yolo_conf_spin.setRange(0.01, 0.99)
        self.yolo_conf_spin.setDecimals(2)
        self.yolo_conf_spin.setSingleStep(0.05)
        self.yolo_conf_spin.setValue(0.25)
        self.yolo_conf_spin.setToolTip(
            "Confidence threshold for predictions during validation.\n"
            "Lower = more detections but more false positives.\n"
            "Higher = fewer detections but more precise."
        )
        self.yolo_conf_label = QLabel("Confidence Threshold:")
        self.model_specific_layout.addRow(self.yolo_conf_label, self.yolo_conf_spin)

        # IoU threshold
        self.yolo_iou_spin = QDoubleSpinBox()
        self.yolo_iou_spin.setRange(0.1, 0.9)
        self.yolo_iou_spin.setDecimals(2)
        self.yolo_iou_spin.setSingleStep(0.05)
        self.yolo_iou_spin.setValue(0.45)
        self.yolo_iou_spin.setToolTip(
            "IoU (Intersection over Union) threshold for NMS.\n"
            "Lower = fewer overlapping detections (more strict).\n"
            "Higher = allow more overlapping detections."
        )
        self.yolo_iou_label = QLabel("IoU Threshold:")
        self.model_specific_layout.addRow(self.yolo_iou_label, self.yolo_iou_spin)

    def _create_view_classifier_params(self):
        """Create View Classifier-specific parameters."""
        # Backbone (model_name)
        self.view_backbone_combo = QComboBox()
        self.view_backbone_combo.addItems([
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"
        ])
        self.view_backbone_combo.setCurrentText("resnet50")  # Updated default to resnet50
        self.view_backbone_combo.setToolTip(
            "Backbone network architecture:\n"
            "• resnet18: Fast, lightweight (11M params)\n"
            "• resnet34: Good balance (22M params)\n"
            "• resnet50: Recommended - high accuracy (25M params)\n"
            "• resnet101: Very high accuracy (45M params)\n"
            "• efficientnet_b0: Efficient baseline (5M params)\n"
            "• efficientnet_b3: Good balance (12M params)"
        )
        self.view_backbone_label = QLabel("Base Model:")
        self.model_specific_layout.addRow(self.view_backbone_label, self.view_backbone_combo)

        # Pretrained
        self.view_pretrained_checkbox = QCheckBox()
        self.view_pretrained_checkbox.setChecked(True)
        self.view_pretrained_checkbox.setToolTip(
            "Use pretrained weights from ImageNet.\n"
            "Highly recommended: Dramatically speeds up training\n"
            "and improves accuracy, especially with small datasets."
        )
        self.view_pretrained_label = QLabel("Use Pretrained:")
        self.model_specific_layout.addRow(self.view_pretrained_label, self.view_pretrained_checkbox)

        # Scheduler
        self.view_scheduler_combo = QComboBox()
        self.view_scheduler_combo.addItems(["StepLR", "CosineAnnealing", "ReduceLROnPlateau"])
        self.view_scheduler_combo.setCurrentText("CosineAnnealing")
        self.view_scheduler_combo.setToolTip(
            "Learning rate scheduler:\n"
            "• StepLR: Reduce LR every N epochs (simple)\n"
            "• CosineAnnealing: Smooth decay (recommended)\n"
            "• ReduceLROnPlateau: Adaptive based on validation"
        )
        self.view_scheduler_label = QLabel("LR Scheduler:")
        self.model_specific_layout.addRow(self.view_scheduler_label, self.view_scheduler_combo)

        # Dropout
        self.view_dropout_spin = QDoubleSpinBox()
        self.view_dropout_spin.setRange(0.0, 0.8)
        self.view_dropout_spin.setDecimals(2)
        self.view_dropout_spin.setSingleStep(0.1)
        self.view_dropout_spin.setValue(0.5)
        self.view_dropout_spin.setToolTip(
            "Dropout rate for regularization.\n"
            "Higher = more regularization, prevents overfitting.\n"
            "0.3-0.5 is typical. Use higher if overfitting occurs."
        )
        self.view_dropout_label = QLabel("Dropout Rate:")
        self.model_specific_layout.addRow(self.view_dropout_label, self.view_dropout_spin)

        # Weight decay
        self.view_weight_decay_spin = QDoubleSpinBox()
        self.view_weight_decay_spin.setRange(0.0, 0.01)
        self.view_weight_decay_spin.setDecimals(5)
        self.view_weight_decay_spin.setSingleStep(0.0001)
        self.view_weight_decay_spin.setValue(0.0001)
        self.view_weight_decay_spin.setToolTip(
            "L2 regularization weight decay.\n"
            "Helps prevent overfitting by penalizing large weights.\n"
            "0.0001 is a good default value."
        )
        self.view_weight_decay_label = QLabel("Weight Decay:")
        self.model_specific_layout.addRow(self.view_weight_decay_label, self.view_weight_decay_spin)

    def _create_defect_classifier_params(self):
        """Create Defect Classifier-specific parameters."""
        # Backbone (model_name)
        self.defect_backbone_combo = QComboBox()
        self.defect_backbone_combo.addItems([
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
            "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
        ])
        self.defect_backbone_combo.setCurrentText("efficientnet_b3")  # Updated default to efficientnet_b3
        self.defect_backbone_combo.setToolTip(
            "Backbone network architecture:\n"
            "• resnet18: Fast, lightweight (11M params)\n"
            "• resnet50: Good for balanced data (25M params)\n"
            "• efficientnet_b0: Efficient baseline (5M params)\n"
            "• efficientnet_b3: Recommended for imbalanced data (12M params)\n"
            "• efficientnet_b5: High accuracy (30M params)\n"
            "• efficientnet_b7: Best accuracy but slower (66M params)"
        )
        self.defect_backbone_label = QLabel("Base Model:")
        self.model_specific_layout.addRow(self.defect_backbone_label, self.defect_backbone_combo)

        # Loss function
        self.defect_loss_combo = QComboBox()
        self.defect_loss_combo.addItems(["CrossEntropy", "Focal", "Weighted"])
        self.defect_loss_combo.setCurrentText("Focal")
        self.defect_loss_combo.setToolTip(
            "Loss function for training:\n"
            "• CrossEntropy: Standard loss (simple)\n"
            "• Focal: Better for imbalanced data (recommended)\n"
            "• Weighted: Class-weighted CrossEntropy"
        )
        self.defect_loss_label = QLabel("Loss Function:")
        self.model_specific_layout.addRow(self.defect_loss_label, self.defect_loss_combo)

        # Focal gamma (only for focal loss)
        self.defect_focal_gamma_spin = QDoubleSpinBox()
        self.defect_focal_gamma_spin.setRange(0.0, 5.0)
        self.defect_focal_gamma_spin.setDecimals(1)
        self.defect_focal_gamma_spin.setSingleStep(0.5)
        self.defect_focal_gamma_spin.setValue(2.0)
        self.defect_focal_gamma_spin.setToolTip(
            "Focal loss gamma parameter.\n"
            "Higher = focus more on hard examples.\n"
            "2.0 is standard, increase if struggling with hard cases."
        )
        self.defect_focal_gamma_label = QLabel("Focal Gamma:")
        self.model_specific_layout.addRow(self.defect_focal_gamma_label, self.defect_focal_gamma_spin)

        # Class weights (for weighted loss)
        self.defect_class_weights_checkbox = QCheckBox()
        self.defect_class_weights_checkbox.setChecked(True)
        self.defect_class_weights_checkbox.setToolTip(
            "Automatically compute class weights based on frequency.\n"
            "Recommended for imbalanced datasets (e.g., PASS vs defects).\n"
            "Gives more weight to underrepresented classes."
        )
        self.defect_class_weights_label = QLabel("Auto Class Weights:")
        self.model_specific_layout.addRow(self.defect_class_weights_label, self.defect_class_weights_checkbox)

        # PASS augmentation factor
        self.defect_pass_aug_spin = QDoubleSpinBox()
        self.defect_pass_aug_spin.setRange(0.0, 5.0)
        self.defect_pass_aug_spin.setDecimals(1)
        self.defect_pass_aug_spin.setSingleStep(0.5)
        self.defect_pass_aug_spin.setValue(1.0)
        self.defect_pass_aug_spin.setToolTip(
            "Extra augmentation multiplier for PASS class.\n"
            "Higher = more aggressive augmentation for PASS samples.\n"
            "Use 1.5-2.0 if PASS class is underrepresented."
        )
        self.defect_pass_aug_label = QLabel("PASS Aug Factor:")
        self.model_specific_layout.addRow(self.defect_pass_aug_label, self.defect_pass_aug_spin)

        # Balanced sampling
        self.defect_balanced_sampling_checkbox = QCheckBox()
        self.defect_balanced_sampling_checkbox.setChecked(True)
        self.defect_balanced_sampling_checkbox.setToolTip(
            "Use balanced sampling during training.\n"
            "Ensures equal probability of sampling each class.\n"
            "Highly recommended for imbalanced datasets."
        )
        self.defect_balanced_sampling_label = QLabel("Balanced Sampling:")
        self.model_specific_layout.addRow(self.defect_balanced_sampling_label, self.defect_balanced_sampling_checkbox)

        # Scheduler
        self.defect_scheduler_combo = QComboBox()
        self.defect_scheduler_combo.addItems(["StepLR", "CosineAnnealing", "ReduceLROnPlateau"])
        self.defect_scheduler_combo.setCurrentText("CosineAnnealing")
        self.defect_scheduler_combo.setToolTip(
            "Learning rate scheduler:\n"
            "• StepLR: Reduce LR every N epochs (simple)\n"
            "• CosineAnnealing: Smooth decay (recommended)\n"
            "• ReduceLROnPlateau: Adaptive based on validation"
        )
        self.defect_scheduler_label = QLabel("LR Scheduler:")
        self.model_specific_layout.addRow(self.defect_scheduler_label, self.defect_scheduler_combo)

        # Dropout
        self.defect_dropout_spin = QDoubleSpinBox()
        self.defect_dropout_spin.setRange(0.0, 0.8)
        self.defect_dropout_spin.setDecimals(2)
        self.defect_dropout_spin.setSingleStep(0.1)
        self.defect_dropout_spin.setValue(0.5)
        self.defect_dropout_spin.setToolTip(
            "Dropout rate for regularization.\n"
            "Higher = more regularization, prevents overfitting.\n"
            "0.3-0.5 is typical. Use higher if overfitting occurs."
        )
        self.defect_dropout_label = QLabel("Dropout Rate:")
        self.model_specific_layout.addRow(self.defect_dropout_label, self.defect_dropout_spin)

        # Weight decay
        self.defect_weight_decay_spin = QDoubleSpinBox()
        self.defect_weight_decay_spin.setRange(0.0, 0.01)
        self.defect_weight_decay_spin.setDecimals(5)
        self.defect_weight_decay_spin.setSingleStep(0.0001)
        self.defect_weight_decay_spin.setValue(0.0001)
        self.defect_weight_decay_spin.setToolTip(
            "L2 regularization weight decay.\n"
            "Helps prevent overfitting by penalizing large weights.\n"
            "0.0001 is a good default value."
        )
        self.defect_weight_decay_label = QLabel("Weight Decay:")
        self.model_specific_layout.addRow(self.defect_weight_decay_label, self.defect_weight_decay_spin)

    def _update_model_specific_visibility(self):
        """Update visibility of model-specific parameters based on selected model type."""
        model_type = self._get_model_type_key()

        # Hide all first
        yolo_widgets = [
            self.yolo_model_label, self.yolo_model_combo,  # Added base model selection
            self.yolo_imgsz_label, self.yolo_imgsz_spin,
            self.yolo_optimizer_label, self.yolo_optimizer_combo,
            self.yolo_warmup_label, self.yolo_warmup_spin,
            self.yolo_conf_label, self.yolo_conf_spin,
            self.yolo_iou_label, self.yolo_iou_spin
        ]
        view_widgets = [
            self.view_backbone_label, self.view_backbone_combo,  # This is the base model selection
            self.view_pretrained_label, self.view_pretrained_checkbox,
            self.view_scheduler_label, self.view_scheduler_combo,
            self.view_dropout_label, self.view_dropout_spin,
            self.view_weight_decay_label, self.view_weight_decay_spin
        ]
        defect_widgets = [
            self.defect_backbone_label, self.defect_backbone_combo,
            self.defect_loss_label, self.defect_loss_combo,
            self.defect_focal_gamma_label, self.defect_focal_gamma_spin,
            self.defect_class_weights_label, self.defect_class_weights_checkbox,
            self.defect_pass_aug_label, self.defect_pass_aug_spin,
            self.defect_balanced_sampling_label, self.defect_balanced_sampling_checkbox,
            self.defect_scheduler_label, self.defect_scheduler_combo,
            self.defect_dropout_label, self.defect_dropout_spin,
            self.defect_weight_decay_label, self.defect_weight_decay_spin
        ]

        for widget in yolo_widgets + view_widgets + defect_widgets:
            widget.hide()

        # Show relevant widgets
        if model_type == "detection":
            for widget in yolo_widgets:
                widget.show()
        elif model_type == "view":
            for widget in view_widgets:
                widget.show()
        elif model_type == "defect":
            for widget in defect_widgets:
                widget.show()

    def _create_statistics_section(self) -> QGroupBox:
        """Create dataset statistics section."""
        group = QGroupBox("Dataset Statistics")
        layout = QHBoxLayout()

        self.stats_label = QLabel("Loading statistics...")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        group.setLayout(layout)
        return group

    def _create_progress_section(self) -> QGroupBox:
        """Create training progress monitoring section."""
        group = QGroupBox("Training Progress")
        layout = QVBoxLayout()

        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, stretch=1)
        layout.addLayout(progress_layout)

        # Metrics display
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setMaximumHeight(150)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metrics_table)

        # Training log
        log_label = QLabel("Training Log:")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text, stretch=1)

        group.setLayout(layout)
        return group

    def _create_control_buttons(self) -> QHBoxLayout:
        """Create control buttons."""
        layout = QHBoxLayout()

        # Start button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self._on_start_training)
        layout.addWidget(self.start_button)

        # Train All button (VIEW-aware)
        self.train_all_button = QPushButton("🚀 Train All VIEW-aware Models")
        self.train_all_button.clicked.connect(self._on_train_all_models)
        self.train_all_button.setToolTip(
            "Automatically train all 5 VIEW-aware models in sequence:\n"
            "1. View Classifier\n"
            "2. Detection Model (YOLO) - TOP\n"
            "3. Detection Model (YOLO) - SIDE\n"
            "4. Defect Classifier - TOP\n"
            "5. Defect Classifier - SIDE\n\n"
            "Loads saved configs for each model type if available."
        )
        self.train_all_button.setStyleSheet("QPushButton { font-weight: bold; color: blue; }")
        layout.addWidget(self.train_all_button)

        # Pause button
        self.pause_button = QPushButton("Pause Training")
        self.pause_button.clicked.connect(self._on_pause_training)
        self.pause_button.setEnabled(False)
        layout.addWidget(self.pause_button)

        # Cancel button
        self.cancel_button = QPushButton("Cancel Training")
        self.cancel_button.clicked.connect(self._on_cancel_training)
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)

        layout.addStretch()

        # TensorBoard button
        self.tensorboard_button = QPushButton("Open TensorBoard")
        self.tensorboard_button.clicked.connect(self._on_open_tensorboard)
        layout.addWidget(self.tensorboard_button)

        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        return layout

    def _load_default_config(self):
        """Load default configuration from YAML file based on selected model type."""
        if yaml is None:
            logger.warning("PyYAML not installed, using hardcoded defaults")
            return

        model_type = self._get_model_type_key()

        # Map VIEW-aware model types to correct config files
        config_file_map = {
            'view': 'view_classifier_config.yaml',
            'detection': 'yolo_config.yaml',
            'detection_top': 'yolo_config.yaml',      # VIEW-aware TOP detection
            'detection_side': 'yolo_config.yaml',     # VIEW-aware SIDE detection
            'defect': 'defect_classifier_config.yaml',
            'defect_top': 'defect_classifier_config.yaml',    # VIEW-aware TOP defect
            'defect_side': 'defect_classifier_config.yaml'    # VIEW-aware SIDE defect
        }

        config_file = self.config_dir / config_file_map.get(model_type, 'yolo_config.yaml')

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Update UI with config values
            self.epochs_spin.setValue(config.get('epochs', 100))

            # Handle different batch size parameter names
            batch_size = config.get('batch', config.get('batch_size', 16))
            self.batch_size_spin.setValue(batch_size)

            # Handle different learning rate parameter names
            learning_rate = config.get('lr0', config.get('learning_rate', 0.01))
            self.learning_rate_spin.setValue(learning_rate)

            # Device
            device = config.get('device', 'auto')
            device_index = {"auto": 0, "cuda": 1, "cpu": 2}.get(device.lower(), 0)
            self.device_combo.setCurrentIndex(device_index)

            # Advanced options
            patience = config.get('patience', config.get('early_stopping_patience', 20))
            self.patience_spin.setValue(patience)

            self.save_best_checkbox.setChecked(config.get('save_best_only', True))

            # Load model_name based on model type
            model_name = config.get('model_name', None)
            if model_name:
                if model_type in ['detection', 'detection_top', 'detection_side']:
                    index = self.yolo_model_combo.findText(model_name)
                    if index >= 0:
                        self.yolo_model_combo.setCurrentIndex(index)
                elif model_type == 'view':
                    index = self.view_backbone_combo.findText(model_name)
                    if index >= 0:
                        self.view_backbone_combo.setCurrentIndex(index)
                elif model_type in ['defect', 'defect_top', 'defect_side']:
                    index = self.defect_backbone_combo.findText(model_name)
                    if index >= 0:
                        self.defect_backbone_combo.setCurrentIndex(index)

            logger.info(f"Loaded config from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _load_dataset_statistics(self):
        """Load and display dataset statistics."""
        try:
            if not Path(self.db_path).exists():
                self.stats_label.setText("⚠ Database not found. Please annotate some data first.")
                return

            analyzer = DatasetAnalyzer(self.db_path)
            stats = analyzer.get_overall_statistics()

            # Format statistics
            stats_text = (
                f"<b>Dataset Overview:</b><br>"
                f"• Total Images: {stats['total_images']}<br>"
                f"• Total Annotations: {stats['total_annotations']}<br>"
                f"• Avg Annotations/Image: {stats['annotations_per_image']:.2f}<br><br>"
                f"<b>View Distribution:</b><br>"
            )

            for view_type, count in stats['view_distribution'].items():
                percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
                stats_text += f"• {view_type}: {count} ({percentage:.1f}%)<br>"

            stats_text += "<br><b>Defect Distribution:</b><br>"
            for defect_type, count in stats['defect_distribution'].items():
                percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
                stats_text += f"• {defect_type}: {count} ({percentage:.1f}%)<br>"

            # Check for insufficient data
            if stats['total_annotations'] < 50:
                stats_text += "<br><b style='color: red;'>⚠ Warning: Dataset is very small. Consider adding more annotations.</b>"

            self.stats_label.setText(stats_text)

        except Exception as e:
            logger.error(f"Failed to load statistics: {e}")
            self.stats_label.setText(f"⚠ Failed to load statistics: {e}")

    def _get_model_type_key(self) -> str:
        """Get model type key from combo box selection."""
        selection = self.model_type_combo.currentText()

        # VIEW-aware models
        if "Detection" in selection and "TOP" in selection:
            return "detection_top"
        elif "Detection" in selection and "SIDE" in selection:
            return "detection_side"
        elif "Defect" in selection and "TOP" in selection:
            return "defect_top"
        elif "Defect" in selection and "SIDE" in selection:
            return "defect_side"

        # Legacy unified models
        elif "Detection" in selection and "Unified" in selection:
            return "detection"
        elif "Defect" in selection and "Unified" in selection:
            return "defect"

        # View classifier (unchanged)
        elif "View" in selection:
            return "view"

        # Fallback
        else:
            return "defect"

    def _on_model_type_changed(self):
        """Handle model type selection change."""
        # Try to load saved config for this model type first
        # If no saved config, fallback to default config
        self._try_load_saved_config_for_current_model()
        self._update_model_specific_visibility()

    def _try_load_saved_config_for_current_model(self):
        """
        Try to load most recent saved config for current model type.
        If no saved config exists, load default config from YAML.
        """
        model_type = self._get_model_type_key()
        config_dir = Path("training_configs")

        # Try to find most recent config for this model type
        config_pattern = f"{model_type}_config_*.json"
        config_files = list(config_dir.glob(config_pattern)) if config_dir.exists() else []

        if config_files:
            # Use most recent config (sorted by filename which includes timestamp)
            latest_config = sorted(config_files)[-1]
            try:
                with open(latest_config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self._apply_config_to_ui(config)
                self._log_message(f"✓ Loaded saved config: {latest_config.name}")
                return True
            except Exception as e:
                self._log_message(f"⚠ Failed to load saved config: {e}")
                # Fallback to default config

        # No saved config found or failed to load, use default from YAML
        self._load_default_config()
        return False

    def _validate_training_config(self) -> bool:
        """
        Validate training configuration before starting.

        Returns:
            True if valid, False otherwise
        """
        # Check database exists
        if not Path(self.db_path).exists():
            QMessageBox.warning(
                self,
                "No Data",
                "Database not found. Please annotate some images first."
            )
            return False

        # Check sufficient data
        try:
            analyzer = DatasetAnalyzer(self.db_path)
            stats = analyzer.get_overall_statistics()

            if stats['total_annotations'] < 10:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Only {stats['total_annotations']} annotations found.\n"
                    "Please annotate at least 10 images before training."
                )
                return False

            if stats['total_annotations'] < 50:
                reply = QMessageBox.question(
                    self,
                    "Small Dataset",
                    f"Only {stats['total_annotations']} annotations found.\n"
                    "Training with such a small dataset may not produce good results.\n\n"
                    "Do you want to continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return False

        except Exception as e:
            logger.error(f"Validation error: {e}")
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Failed to validate dataset: {e}"
            )
            return False

        return True

    def _get_training_config(self) -> Dict[str, Any]:
        """
        Get current training configuration from UI.

        Returns:
            Dictionary with training parameters
        """
        device_map = {0: "auto", 1: "cuda", 2: "cpu"}
        model_type = self._get_model_type_key()

        # Common parameters
        config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'device': device_map[self.device_combo.currentIndex()],
            'patience': self.patience_spin.value(),
            'save_best_only': self.save_best_checkbox.isChecked(),
            'use_augmentation': self.augmentation_checkbox.isChecked()
        }

        # Model-specific parameters
        if self.model_specific_group.isChecked():
            if model_type in ['detection', 'detection_top', 'detection_side']:
                config.update({
                    'model_name': self.yolo_model_combo.currentText(),  # Base model selection
                    'imgsz': self.yolo_imgsz_spin.value(),
                    'optimizer': self.yolo_optimizer_combo.currentText(),
                    'warmup_epochs': self.yolo_warmup_spin.value(),
                    'conf_threshold': self.yolo_conf_spin.value(),
                    'iou_threshold': self.yolo_iou_spin.value()
                })
            elif model_type == 'view':
                config.update({
                    'model_name': self.view_backbone_combo.currentText(),  # Base model selection
                    'backbone': self.view_backbone_combo.currentText(),  # Keep for backward compatibility
                    'pretrained': self.view_pretrained_checkbox.isChecked(),
                    'scheduler': self.view_scheduler_combo.currentText(),
                    'dropout': self.view_dropout_spin.value(),
                    'weight_decay': self.view_weight_decay_spin.value()
                })
            elif model_type in ['defect', 'defect_top', 'defect_side']:
                config.update({
                    'model_name': self.defect_backbone_combo.currentText(),  # Base model selection
                    'backbone': self.defect_backbone_combo.currentText(),  # Keep for backward compatibility
                    'loss_function': self.defect_loss_combo.currentText(),
                    'focal_gamma': self.defect_focal_gamma_spin.value(),
                    'auto_class_weights': self.defect_class_weights_checkbox.isChecked(),
                    'pass_aug_factor': self.defect_pass_aug_spin.value(),
                    'balanced_sampling': self.defect_balanced_sampling_checkbox.isChecked(),
                    'scheduler': self.defect_scheduler_combo.currentText(),
                    'dropout': self.defect_dropout_spin.value(),
                    'weight_decay': self.defect_weight_decay_spin.value()
                })

        return config

    def _on_start_training(self):
        """Handle start training button click."""
        # Validate configuration
        if not self._validate_training_config():
            return

        # Get configuration
        model_type = self._get_model_type_key()
        self.current_config = self._get_training_config()

        # Clear previous logs
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.metrics_table.setRowCount(0)

        try:
            # Prepare data using full pipeline with wire pairing support
            data_prep = DataPreparator(db_path=self.db_path, random_seed=42)

            # Get wire pairing preference from checkbox
            preserve_wire_pairs = self.preserve_wire_pairs_checkbox.isChecked()

            # Determine if VIEW-aware architecture is needed
            view_aware = model_type in ['detection_top', 'detection_side', 'defect_top', 'defect_side']

            # Prepare complete datasets with optional wire pairing
            self._log_message("Preparing datasets...")
            if preserve_wire_pairs:
                self._log_message("  Using wire-aware split to preserve TOP/SIDE pairs")
            else:
                self._log_message("  Using standard stratified split")

            if view_aware:
                self._log_message("  Using VIEW-aware architecture (separate TOP/SIDE models)")
            else:
                self._log_message("  Using unified architecture (single model for both views)")

            complete_info = data_prep.prepare_full_pipeline(
                output_base_dir='datasets',
                val_ratio=0.2,
                stratify_by='defect_type',
                preserve_wire_pairs=preserve_wire_pairs,
                view_aware=view_aware
            )

            # Extract dataset info based on model type
            # YOLO Detection models
            if model_type in ['detection', 'detection_top', 'detection_side']:
                # Determine dataset key
                if model_type == 'detection':
                    dataset_key = 'yolo_detection'
                elif model_type == 'detection_top':
                    dataset_key = 'yolo_detection_top'
                else:  # detection_side
                    dataset_key = 'yolo_detection_side'

                dataset_info = complete_info[dataset_key]
                data_yaml_path = dataset_info['data_yaml']

                self._log_message(f"YOLO dataset prepared: {data_yaml_path}")

                # Create YOLO worker
                self.worker = TrainingWorker(
                    model_type=model_type,
                    config=self.current_config,
                    database_path=self.db_path,
                    data_yaml_path=data_yaml_path,
                    output_dir=f'runs/{model_type}'
                )

            # Classifier models
            else:
                # Determine dataset key
                if model_type == 'view':
                    dataset_key = 'view_classifier'
                elif model_type == 'defect':
                    dataset_key = 'defect_classifier'
                elif model_type == 'defect_top':
                    dataset_key = 'defect_classifier_top'
                else:  # defect_side
                    dataset_key = 'defect_classifier_side'

                dataset_info = complete_info[dataset_key]
                train_dir = dataset_info['train_dir']
                val_dir = dataset_info['val_dir']

                self._log_message(f"{model_type.replace('_', ' ').capitalize()} classifier dataset prepared")
                self._log_message(f"  Train: {train_dir}")
                self._log_message(f"  Val: {val_dir}")

                # Create classifier worker
                self.worker = TrainingWorker(
                    model_type=model_type,
                    config=self.current_config,
                    database_path=self.db_path,
                    train_data=train_dir,
                    val_data=val_dir,
                    output_dir=f'runs/{model_type}'
                )

            # Connect signals (note the actual signal signatures from TrainingWorker)
            self.worker.progress_updated.connect(self._on_progress_updated)
            self.worker.epoch_completed.connect(self._on_epoch_completed)
            self.worker.training_finished.connect(self._on_training_finished)
            self.worker.training_error.connect(self._on_training_error)
            self.worker.log_message.connect(self._on_log_message)
            self.worker.state_changed.connect(self._on_state_changed)

            # Start training
            self.worker.start()
            self.is_training = True
            self.start_time = datetime.now()

            self._update_button_states()
            self._log_message("=" * 60)
            self._log_message(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self._log_message(f"Model Type: {model_type}")
            self._log_message(f"Configuration: {self.current_config}")
            self._log_message("=" * 60)

        except Exception as e:
            logger.error(f"Failed to start training: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Training Error",
                f"Failed to start training:\n\n{str(e)}"
            )

    def _on_pause_training(self):
        """Handle pause training button click."""
        if self.worker and self.is_training:
            if self.pause_button.text() == "Pause Training":
                self.worker.request_pause()
                self.pause_button.setText("Resume Training")
                self._log_message("Training pause requested...")
            else:
                self.worker.request_resume()
                self.pause_button.setText("Pause Training")
                self._log_message("Training resume requested...")

    def _on_cancel_training(self):
        """Handle cancel training button click."""
        if self.worker and self.is_training:
            reply = QMessageBox.question(
                self,
                "Cancel Training",
                "Are you sure you want to cancel training?\nProgress will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.worker.request_cancel()
                self._log_message("Training cancellation requested...")
                self.is_training = False
                self._update_button_states()

    def _on_open_tensorboard(self):
        """Handle open TensorBoard button click."""
        runs_dir = Path("runs")

        if not runs_dir.exists():
            QMessageBox.information(
                self,
                "No Training Runs",
                "No training runs found. TensorBoard logs will appear in the 'runs/' directory after training starts."
            )
            return

        # Start TensorBoard
        if self.tensorboard_manager.is_running():
            # Already running, just open browser
            url = self.tensorboard_manager.get_tensorboard_url()
            if url:
                import webbrowser
                webbrowser.open(url)
        else:
            success = self.tensorboard_manager.start_tensorboard(
                logdir=str(runs_dir),
                auto_open=True
            )

            if success:
                self._log_message("TensorBoard opened in browser")
            else:
                QMessageBox.warning(
                    self,
                    "TensorBoard Error",
                    "Failed to start TensorBoard. Make sure it's installed:\npip install tensorboard"
                )

    def _on_save_config(self):
        """Handle save configuration button click."""
        try:
            # Get current configuration from UI
            config = self._collect_current_config()

            # Create configs directory if it doesn't exist
            config_save_dir = Path("training_configs")
            config_save_dir.mkdir(exist_ok=True)

            # Suggest default filename based on model type and timestamp
            model_type = self._get_model_type_key()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"{model_type}_config_{timestamp}.json"
            default_path = config_save_dir / default_filename

            # Show file dialog
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Training Configuration",
                str(default_path),
                "JSON Files (*.json);;All Files (*)"
            )

            if not filename:
                return  # User cancelled

            # Save configuration
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            QMessageBox.information(
                self,
                "Config Saved",
                f"Training configuration saved to:\n{filename}"
            )
            self._log_message(f"Configuration saved to {Path(filename).name}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save configuration:\n{e}"
            )

    def _on_load_config(self):
        """Handle load configuration button click."""
        try:
            # Create configs directory if it doesn't exist
            config_save_dir = Path("training_configs")
            config_save_dir.mkdir(exist_ok=True)

            # Show file dialog
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load Training Configuration",
                str(config_save_dir),
                "JSON Files (*.json);;All Files (*)"
            )

            if not filename:
                return  # User cancelled

            # Load configuration
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Apply configuration to UI
            self._apply_config_to_ui(config)

            QMessageBox.information(
                self,
                "Config Loaded",
                f"Training configuration loaded from:\n{filename}"
            )
            self._log_message(f"Configuration loaded from {Path(filename).name}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load configuration:\n{e}"
            )

    def _collect_current_config(self) -> Dict[str, Any]:
        """Collect current configuration from all UI widgets."""
        model_type = self._get_model_type_key()

        config = {
            'model_type': model_type,
            'common': {
                'epochs': self.epochs_spin.value(),
                'batch_size': self.batch_size_spin.value(),
                'learning_rate': self.learning_rate_spin.value(),
                'device': self.device_combo.currentText(),
                'patience': self.patience_spin.value(),
                'save_best_only': self.save_best_checkbox.isChecked(),
                'use_augmentation': self.augmentation_checkbox.isChecked(),
                'preserve_wire_pairs': self.preserve_wire_pairs_checkbox.isChecked()
            }
        }

        # Add model-specific parameters
        # Detection models (unified, TOP, SIDE all share YOLO parameters)
        if model_type in ['detection', 'detection_top', 'detection_side']:
            config['yolo'] = {
                'model_name': self.yolo_model_combo.currentText(),  # Base model selection
                'imgsz': self.yolo_imgsz_spin.value(),
                'optimizer': self.yolo_optimizer_combo.currentText(),
                'warmup_epochs': self.yolo_warmup_spin.value(),
                'conf_threshold': self.yolo_conf_spin.value(),
                'iou_threshold': self.yolo_iou_spin.value()
            }
        elif model_type == 'view':
            config['view'] = {
                'model_name': self.view_backbone_combo.currentText(),  # Base model (backbone)
                'backbone': self.view_backbone_combo.currentText(),  # Keep for backward compatibility
                'pretrained': self.view_pretrained_checkbox.isChecked(),
                'scheduler': self.view_scheduler_combo.currentText(),
                'dropout': self.view_dropout_spin.value(),
                'weight_decay': self.view_weight_decay_spin.value()
            }
        # Defect models (unified, TOP, SIDE all share defect parameters)
        elif model_type in ['defect', 'defect_top', 'defect_side']:
            config['defect'] = {
                'model_name': self.defect_backbone_combo.currentText(),  # Base model (backbone)
                'backbone': self.defect_backbone_combo.currentText(),  # Keep for backward compatibility
                'loss_function': self.defect_loss_combo.currentText(),
                'focal_gamma': self.defect_focal_gamma_spin.value(),
                'use_class_weights': self.defect_class_weights_checkbox.isChecked(),
                'pass_augmentation_factor': self.defect_pass_aug_spin.value(),
                'balanced_sampling': self.defect_balanced_sampling_checkbox.isChecked(),
                'scheduler': self.defect_scheduler_combo.currentText(),
                'dropout': self.defect_dropout_spin.value(),
                'weight_decay': self.defect_weight_decay_spin.value()
            }

        return config

    def _apply_config_to_ui(self, config: Dict[str, Any]):
        """Apply loaded configuration to UI widgets."""
        # Note: We DON'T change the model type combo box here
        # User has already selected the model type they want
        # This method only applies the configuration parameters to the UI
        model_type = config.get('model_type', 'detection')

        # Apply common parameters
        common = config.get('common', {})
        if 'epochs' in common:
            self.epochs_spin.setValue(common['epochs'])
        if 'batch_size' in common:
            self.batch_size_spin.setValue(common['batch_size'])
        if 'learning_rate' in common:
            self.learning_rate_spin.setValue(common['learning_rate'])
        if 'device' in common:
            index = self.device_combo.findText(common['device'])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        if 'patience' in common:
            self.patience_spin.setValue(common['patience'])
        if 'save_best_only' in common:
            self.save_best_checkbox.setChecked(common['save_best_only'])
        if 'use_augmentation' in common:
            self.augmentation_checkbox.setChecked(common['use_augmentation'])
        if 'preserve_wire_pairs' in common:
            self.preserve_wire_pairs_checkbox.setChecked(common['preserve_wire_pairs'])

        # Apply model-specific parameters
        # Detection models (unified, TOP, SIDE all share YOLO parameters)
        if model_type in ['detection', 'detection_top', 'detection_side'] and 'yolo' in config:
            yolo = config['yolo']
            # Load model_name (base model)
            if 'model_name' in yolo:
                index = self.yolo_model_combo.findText(yolo['model_name'])
                if index >= 0:
                    self.yolo_model_combo.setCurrentIndex(index)
            if 'imgsz' in yolo:
                self.yolo_imgsz_spin.setValue(yolo['imgsz'])
            if 'optimizer' in yolo:
                index = self.yolo_optimizer_combo.findText(yolo['optimizer'])
                if index >= 0:
                    self.yolo_optimizer_combo.setCurrentIndex(index)
            if 'warmup_epochs' in yolo:
                self.yolo_warmup_spin.setValue(yolo['warmup_epochs'])
            if 'conf_threshold' in yolo:
                self.yolo_conf_spin.setValue(yolo['conf_threshold'])
            if 'iou_threshold' in yolo:
                self.yolo_iou_spin.setValue(yolo['iou_threshold'])

        elif model_type == 'view' and 'view' in config:
            view = config['view']
            # Load model_name (prefer model_name, fallback to backbone for backward compatibility)
            model_name = view.get('model_name', view.get('backbone', 'resnet50'))
            index = self.view_backbone_combo.findText(model_name)
            if index >= 0:
                self.view_backbone_combo.setCurrentIndex(index)
            if 'pretrained' in view:
                self.view_pretrained_checkbox.setChecked(view['pretrained'])
            if 'scheduler' in view:
                index = self.view_scheduler_combo.findText(view['scheduler'])
                if index >= 0:
                    self.view_scheduler_combo.setCurrentIndex(index)
            if 'dropout' in view:
                self.view_dropout_spin.setValue(view['dropout'])
            if 'weight_decay' in view:
                self.view_weight_decay_spin.setValue(view['weight_decay'])

        # Defect models (unified, TOP, SIDE all share defect parameters)
        elif model_type in ['defect', 'defect_top', 'defect_side'] and 'defect' in config:
            defect = config['defect']
            # Load model_name (prefer model_name, fallback to backbone for backward compatibility)
            model_name = defect.get('model_name', defect.get('backbone', 'efficientnet_b3'))
            index = self.defect_backbone_combo.findText(model_name)
            if index >= 0:
                self.defect_backbone_combo.setCurrentIndex(index)
            if 'loss_function' in defect:
                index = self.defect_loss_combo.findText(defect['loss_function'])
                if index >= 0:
                    self.defect_loss_combo.setCurrentIndex(index)
            if 'focal_gamma' in defect:
                self.defect_focal_gamma_spin.setValue(defect['focal_gamma'])
            if 'use_class_weights' in defect:
                self.defect_class_weights_checkbox.setChecked(defect['use_class_weights'])
            if 'pass_augmentation_factor' in defect:
                self.defect_pass_aug_spin.setValue(defect['pass_augmentation_factor'])
            if 'balanced_sampling' in defect:
                self.defect_balanced_sampling_checkbox.setChecked(defect['balanced_sampling'])
            if 'scheduler' in defect:
                index = self.defect_scheduler_combo.findText(defect['scheduler'])
                if index >= 0:
                    self.defect_scheduler_combo.setCurrentIndex(index)
            if 'dropout' in defect:
                self.defect_dropout_spin.setValue(defect['dropout'])
            if 'weight_decay' in defect:
                self.defect_weight_decay_spin.setValue(defect['weight_decay'])

    def _on_progress_updated(self, current_epoch: int, total_epochs: int, progress_percent: float):
        """Handle progress update from worker."""
        self.progress_bar.setValue(int(progress_percent))

    def _on_epoch_completed(self, epoch: int, metrics: Dict[str, float]):
        """Handle epoch completion with metrics update."""
        # Update metrics table
        self.metrics_table.setRowCount(len(metrics))

        for i, (key, value) in enumerate(metrics.items()):
            # Format value
            if isinstance(value, float):
                if 'loss' in key.lower():
                    value_str = f"{value:.4f}"
                elif 'accuracy' in key.lower() or 'recall' in key.lower() or 'precision' in key.lower():
                    value_str = f"{value:.2%}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            self.metrics_table.setItem(i, 0, QTableWidgetItem(key.replace('_', ' ').title()))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value_str))

        # Calculate ETA
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            total_epochs = self.current_config.get('epochs', 100)
            if epoch > 0:
                avg_time_per_epoch = elapsed / epoch
                remaining_epochs = total_epochs - epoch
                eta_seconds = remaining_epochs * avg_time_per_epoch

                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                seconds = int(eta_seconds % 60)
                eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                # Add ETA to metrics table
                row = self.metrics_table.rowCount()
                self.metrics_table.insertRow(row)
                self.metrics_table.setItem(row, 0, QTableWidgetItem("ETA"))
                self.metrics_table.setItem(row, 1, QTableWidgetItem(eta_str))

    def _on_log_message(self, level: str, message: str):
        """Handle log message from worker."""
        self._log_message(f"[{level}] {message}")

    def _on_state_changed(self, state):
        """Handle training state change from worker."""
        from src.gui.training_worker import TrainingState

        # Update button states based on training state
        if state == TrainingState.RUNNING:
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.pause_button.setText("Pause Training")
            self.cancel_button.setEnabled(True)
            self.close_button.setEnabled(False)

            # Disable configuration inputs during training
            self.model_type_combo.setEnabled(False)
            self.epochs_spin.setEnabled(False)
            self.batch_size_spin.setEnabled(False)
            self.learning_rate_spin.setEnabled(False)
            self.device_combo.setEnabled(False)
            self.advanced_group.setEnabled(False)

        elif state == TrainingState.PAUSED:
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.pause_button.setText("Resume Training")
            self.cancel_button.setEnabled(True)
            self.close_button.setEnabled(False)

            # Keep configuration inputs disabled during pause
            self.model_type_combo.setEnabled(False)
            self.epochs_spin.setEnabled(False)
            self.batch_size_spin.setEnabled(False)
            self.learning_rate_spin.setEnabled(False)
            self.device_combo.setEnabled(False)
            self.advanced_group.setEnabled(False)

        elif state == TrainingState.IDLE:
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Pause Training")
            self.cancel_button.setEnabled(False)
            self.close_button.setEnabled(True)

            # Re-enable configuration inputs
            self.model_type_combo.setEnabled(True)
            self.epochs_spin.setEnabled(True)
            self.batch_size_spin.setEnabled(True)
            self.learning_rate_spin.setEnabled(True)
            self.device_combo.setEnabled(True)
            self.advanced_group.setEnabled(True)

        elif state == TrainingState.CANCELLED:
            # Cancelled state will transition to IDLE soon
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.close_button.setEnabled(False)

    def _on_training_finished(self, success: bool, model_path: str, final_metrics: Dict[str, float]):
        """Handle training completion."""
        self.is_training = False
        self._update_button_states()

        if success:
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)

            self._log_message("=" * 60)
            self._log_message("Training completed successfully!")
            self._log_message(f"Best model saved to: {model_path}")
            self._log_message(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            if final_metrics:
                self._log_message(f"Final metrics: {final_metrics}")
            self._log_message("=" * 60)

            # Check if we're in batch training mode
            if self.is_batch_training:
                self.current_queue_index += 1
                if self.current_queue_index < len(self.training_queue):
                    # Continue with next model in queue
                    self._log_message(f"\n{'='*60}")
                    self._log_message(f"📋 Batch Training Progress: {self.current_queue_index}/{len(self.training_queue)} completed")
                    self._log_message(f"{'='*60}\n")
                    self._start_next_training_in_queue()
                else:
                    # All trainings completed
                    self.is_batch_training = False
                    self._log_message("\n" + "=" * 60)
                    self._log_message("🎉 ALL MODEL TRAINING COMPLETED!")
                    self._log_message(f"Successfully trained {len(self.training_queue)} models")
                    self._log_message("=" * 60)
                    QMessageBox.information(
                        self,
                        "Batch Training Complete",
                        f"All {len(self.training_queue)} VIEW-aware models trained successfully!\n\n"
                        "You can now use the complete inference system."
                    )
            else:
                # Single training mode - show normal completion dialog
                QMessageBox.information(
                    self,
                    "Training Complete",
                    f"Training completed successfully!\n\n"
                    f"Best model saved to:\n{model_path}\n\n"
                    f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                )
        else:
            self._log_message("=" * 60)
            self._log_message("Training was cancelled or did not complete")
            self._log_message("=" * 60)

            # Cancel batch training if in progress
            if self.is_batch_training:
                self.is_batch_training = False
                self._log_message("Batch training cancelled.")
                QMessageBox.warning(
                    self,
                    "Batch Training Cancelled",
                    f"Training of model {self.current_queue_index + 1}/{len(self.training_queue)} failed.\n"
                    "Batch training has been stopped."
                )

    def _on_training_error(self, error_message: str, traceback_str: str):
        """Handle training error with helpful suggestions."""
        self.is_training = False
        self._update_button_states()

        self._log_message("=" * 60)
        self._log_message(f"Training failed: {error_message}")
        self._log_message("Traceback:")
        self._log_message(traceback_str)
        self._log_message("=" * 60)

        # Provide helpful suggestions based on error type
        suggestion = self._get_error_suggestion(error_message, traceback_str)

        error_details = f"Training failed with error:\n\n{error_message}"
        if suggestion:
            error_details += f"\n\nSuggested fix:\n{suggestion}"
        error_details += "\n\nCheck the training log for full details."

        QMessageBox.critical(
            self,
            "Training Failed",
            error_details
        )

    def _get_error_suggestion(self, error_message: str, traceback_str: str) -> str:
        """
        Get helpful suggestion based on error type.

        Args:
            error_message: Error message string
            traceback_str: Full traceback string

        Returns:
            Suggestion string, or empty if no specific suggestion
        """
        error_lower = error_message.lower()
        trace_lower = traceback_str.lower()

        # GPU out of memory error
        if "out of memory" in error_lower or "cuda out of memory" in trace_lower:
            return (
                "GPU ran out of memory. Try one of these solutions:\n"
                "1. Reduce batch size (try 8 or 4)\n"
                "2. Use CPU instead of CUDA (slower but works)\n"
                "3. Close other GPU-using applications\n"
                "4. Restart the application"
            )

        # Insufficient data error
        if "insufficient" in error_lower or "not enough" in error_lower:
            return (
                "Not enough training data. You need:\n"
                "• At least 10 annotations per class\n"
                "• Preferably 50+ annotations for good results\n"
                "→ Use the annotation tool to label more images"
            )

        # Missing files error
        if "no such file" in error_lower or "filenotfounderror" in trace_lower:
            return (
                "Required files are missing. This could mean:\n"
                "1. Models not trained yet → Train models first\n"
                "2. Dataset not prepared → Run data preparation\n"
                "3. Configuration file missing → Check config directory"
            )

        # TensorBoard port conflict
        if "address already in use" in error_lower or "port" in error_lower:
            return (
                "TensorBoard port is already in use. Try:\n"
                "1. Close other TensorBoard instances\n"
                "2. Restart the application\n"
                "3. The system will auto-retry with a different port"
            )

        # CUDA/GPU not available
        if "cuda" in error_lower and ("not available" in error_lower or "no cuda" in error_lower):
            return (
                "CUDA/GPU not available. Solutions:\n"
                "1. Switch device to 'Auto' or 'CPU'\n"
                "2. Install CUDA toolkit and compatible PyTorch\n"
                "3. Check if GPU is properly installed\n"
                "→ CPU training will be slower but will work"
            )

        # Permission errors
        if "permission" in error_lower or "access denied" in error_lower:
            return (
                "Permission error accessing files. Try:\n"
                "1. Run as administrator\n"
                "2. Check file/folder permissions\n"
                "3. Close other applications using these files"
            )

        # Import errors
        if "importerror" in trace_lower or "modulenotfounderror" in trace_lower:
            return (
                "Missing required Python packages. Try:\n"
                "1. Install missing package: pip install <package-name>\n"
                "2. Reinstall requirements: pip install -r requirements.txt\n"
                "3. Check virtual environment is activated"
            )

        # Default: no specific suggestion
        return ""

    def _log_message(self, message: str):
        """Append message to log viewer."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def _check_for_checkpoint(self):
        """Check for existing checkpoints and offer to resume."""
        # Get current model type key
        model_type = self._get_model_type_key()

        # Check if checkpoint exists
        if not self.checkpoint_manager.has_checkpoint(model_type):
            return

        # Get checkpoint info
        checkpoint_info = self.checkpoint_manager.get_checkpoint_info(model_type)
        if not checkpoint_info:
            return

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(checkpoint_info['timestamp'])
            time_str = timestamp.strftime('%Y-%m-%d %H:%M')
        except:
            time_str = checkpoint_info.get('timestamp', 'Unknown')

        # Show resume dialog
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Resume Training?")
        msg.setText("Found checkpoint from previous training session")
        msg.setInformativeText(
            f"Model Type: {checkpoint_info['model_type']}\n"
            f"Last Epoch: {checkpoint_info['epoch']}/{checkpoint_info['total_epochs']}\n"
            f"Date: {time_str}\n\n"
            f"Would you like to resume training from this checkpoint?"
        )

        resume_btn = msg.addButton("Resume Training", QMessageBox.ButtonRole.AcceptRole)
        new_btn = msg.addButton("Start New Training", QMessageBox.ButtonRole.RejectRole)
        discard_btn = msg.addButton("Discard Checkpoint", QMessageBox.ButtonRole.DestructiveRole)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == resume_btn:
            # User wants to resume
            self.resume_checkpoint = checkpoint_info
            self._log_message(f"Will resume training from epoch {checkpoint_info['epoch']}/{checkpoint_info['total_epochs']}")
        elif clicked == discard_btn:
            # User wants to discard checkpoint
            self.checkpoint_manager.delete_checkpoint(model_type)
            self._log_message("Checkpoint discarded")
        else:
            # Start new training - archive old checkpoint
            # CheckpointManager will handle overwriting on new save
            self._log_message("Will start new training (old checkpoint will be replaced)")

    def _update_button_states(self):
        """Update button enabled/disabled states based on training state."""
        self.start_button.setEnabled(not self.is_training)
        self.pause_button.setEnabled(self.is_training)
        self.cancel_button.setEnabled(self.is_training)
        self.close_button.setEnabled(not self.is_training)

        # Disable configuration inputs during training
        self.model_type_combo.setEnabled(not self.is_training)
        self.epochs_spin.setEnabled(not self.is_training)
        self.batch_size_spin.setEnabled(not self.is_training)
        self.learning_rate_spin.setEnabled(not self.is_training)
        self.device_combo.setEnabled(not self.is_training)
        self.advanced_group.setEnabled(not self.is_training)

    def _on_train_all_models(self):
        """Handle train all VIEW-aware models button click."""
        # Validate configuration
        if not self._validate_training_config():
            return

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Train All VIEW-aware Models",
            "This will train all 5 VIEW-aware models in sequence:\n\n"
            "1. View Classifier\n"
            "2. Detection Model (YOLO) - TOP\n"
            "3. Detection Model (YOLO) - SIDE\n"
            "4. Defect Classifier - TOP\n"
            "5. Defect Classifier - SIDE\n\n"
            "This may take 3.5-5 hours on GPU.\n\n"
            "The system will load saved configs for each model if available,\n"
            "otherwise use default settings.\n\n"
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Clear previous logs
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.metrics_table.setRowCount(0)

        # Setup training queue
        self.training_queue = [
            'view',
            'detection_top',
            'detection_side',
            'defect_top',
            'defect_side'
        ]
        self.current_queue_index = 0
        self.is_batch_training = True

        # Load configs for each model type
        self._load_batch_training_configs()

        # Log batch training start
        self._log_message("=" * 60)
        self._log_message("🚀 BATCH TRAINING: All VIEW-aware Models")
        self._log_message("=" * 60)
        self._log_message(f"Total models to train: {len(self.training_queue)}")
        self._log_message("Training sequence:")
        for i, model_type in enumerate(self.training_queue, 1):
            self._log_message(f"  {i}. {self._get_model_type_display_name(model_type)}")
        self._log_message("=" * 60 + "\n")

        # Start first training
        self._start_next_training_in_queue()

    def _load_batch_training_configs(self):
        """Load saved configs for each model type in the training queue."""
        config_dir = Path("training_configs")
        self.batch_training_configs = {}

        for model_type in self.training_queue:
            # Try to find most recent config for this model type
            config_pattern = f"{model_type}_config_*.json"
            config_files = list(config_dir.glob(config_pattern)) if config_dir.exists() else []

            if config_files:
                # Use most recent config (sorted by filename which includes timestamp)
                latest_config = sorted(config_files)[-1]
                try:
                    with open(latest_config, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    self.batch_training_configs[model_type] = config
                    self._log_message(f"✓ Loaded config for {model_type}: {latest_config.name}")
                except Exception as e:
                    self._log_message(f"⚠ Failed to load config for {model_type}: {e}")
                    self.batch_training_configs[model_type] = None
            else:
                self._log_message(f"ℹ No saved config found for {model_type}, will use UI settings")
                self.batch_training_configs[model_type] = None

        self._log_message("")

    def _start_next_training_in_queue(self):
        """Start training the next model in the queue."""
        if self.current_queue_index >= len(self.training_queue):
            return

        model_type = self.training_queue[self.current_queue_index]

        self._log_message("\n" + "=" * 60)
        self._log_message(f"📦 Starting Training {self.current_queue_index + 1}/{len(self.training_queue)}")
        self._log_message(f"Model: {self._get_model_type_display_name(model_type)}")
        self._log_message("=" * 60 + "\n")

        # Set model type in UI
        self._set_model_type_in_ui(model_type)

        # Load config if available
        if self.batch_training_configs.get(model_type):
            config = self.batch_training_configs[model_type]
            self._apply_config_to_ui(config)
            self._log_message(f"Using saved configuration for {model_type}")
        else:
            self._log_message(f"Using default/UI configuration for {model_type}")

        # Start training (reuse existing logic)
        try:
            # Get configuration
            self.current_config = self._get_training_config()

            # Prepare data
            data_prep = DataPreparator(db_path=self.db_path, random_seed=42)
            preserve_wire_pairs = self.preserve_wire_pairs_checkbox.isChecked()

            # In batch training mode, always use VIEW-aware to prepare all 5 datasets
            # In single training mode, use VIEW-aware only for view-specific models
            if self.is_batch_training:
                view_aware = True  # Always use VIEW-aware in batch mode
            else:
                view_aware = model_type in ['detection_top', 'detection_side', 'defect_top', 'defect_side']

            self._log_message("Preparing datasets...")
            complete_info = data_prep.prepare_full_pipeline(
                output_base_dir='datasets',
                val_ratio=0.2,
                stratify_by='defect_type',
                preserve_wire_pairs=preserve_wire_pairs,
                view_aware=view_aware
            )

            # Extract dataset info and create worker
            if model_type in ['detection', 'detection_top', 'detection_side']:
                # Determine dataset key
                if model_type == 'detection':
                    dataset_key = 'yolo_detection'
                elif model_type == 'detection_top':
                    dataset_key = 'yolo_detection_top'
                else:  # detection_side
                    dataset_key = 'yolo_detection_side'

                dataset_info = complete_info[dataset_key]
                data_yaml_path = dataset_info['data_yaml']

                self.worker = TrainingWorker(
                    model_type=model_type,
                    config=self.current_config,
                    database_path=self.db_path,
                    data_yaml_path=data_yaml_path,
                    output_dir=f'runs/{model_type}'
                )
            else:
                # Classifier models
                if model_type == 'view':
                    dataset_key = 'view_classifier'
                elif model_type == 'defect':
                    dataset_key = 'defect_classifier'
                elif model_type == 'defect_top':
                    dataset_key = 'defect_classifier_top'
                else:  # defect_side
                    dataset_key = 'defect_classifier_side'

                dataset_info = complete_info[dataset_key]
                train_dir = dataset_info['train_dir']
                val_dir = dataset_info['val_dir']

                self.worker = TrainingWorker(
                    model_type=model_type,
                    config=self.current_config,
                    database_path=self.db_path,
                    train_data=train_dir,
                    val_data=val_dir,
                    output_dir=f'runs/{model_type}'
                )

            # Connect signals
            self.worker.progress_updated.connect(self._on_progress_updated)
            self.worker.epoch_completed.connect(self._on_epoch_completed)
            self.worker.training_finished.connect(self._on_training_finished)
            self.worker.training_error.connect(self._on_training_error)
            self.worker.log_message.connect(self._on_log_message)
            self.worker.state_changed.connect(self._on_state_changed)

            # Start training
            self.worker.start()
            self.is_training = True
            self.start_time = datetime.now()
            self._update_button_states()

        except Exception as e:
            logger.error(f"Failed to start training for {model_type}: {e}", exc_info=True)
            self._log_message(f"ERROR: Failed to start training: {e}")
            self.is_batch_training = False
            QMessageBox.critical(
                self,
                "Training Error",
                f"Failed to start training for {model_type}:\n{e}"
            )

    def _get_model_type_display_name(self, model_type: str) -> str:
        """Get display name for model type."""
        display_names = {
            'view': 'View Classifier',
            'detection_top': 'Detection Model (YOLO) - TOP',
            'detection_side': 'Detection Model (YOLO) - SIDE',
            'defect_top': 'Defect Classifier - TOP',
            'defect_side': 'Defect Classifier - SIDE',
            'detection': 'Detection Model (YOLO) - Unified',
            'defect': 'Defect Classifier - Unified'
        }
        return display_names.get(model_type, model_type)

    def _set_model_type_in_ui(self, model_type: str):
        """Set the model type combo box to match the given model type."""
        combo_index_map = {
            'view': 0,
            'detection_top': 1,
            'detection_side': 2,
            'defect_top': 3,
            'defect_side': 4,
            'detection': 6,  # Unified
            'defect': 7      # Unified
        }
        index = combo_index_map.get(model_type, 0)
        self.model_type_combo.setCurrentIndex(index)

    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.is_training:
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still in progress. Are you sure you want to close?\n"
                "This will cancel the training.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self.worker:
                    self.worker.cancel()
                    self.worker.wait()  # Wait for thread to finish
                event.accept()
            else:
                event.ignore()
        else:
            # Stop TensorBoard if running
            if self.tensorboard_manager.is_running():
                self.tensorboard_manager.stop_tensorboard()
            event.accept()


if __name__ == "__main__":
    """Test the training dialog."""
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    dialog = TrainingDialog(
        db_path="annotations.db",
        config_dir="configs",
        data_dir="data"
    )
    dialog.show()

    sys.exit(app.exec())
