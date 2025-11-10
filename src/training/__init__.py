"""
Training module for Wire Loop X-ray Classification.

This module provides comprehensive training functionality for three models:
1. YOLO Detection (YOLOv8)
2. View Classifier (ResNet18)
3. Defect Classifier (EfficientNet-B0)

Key features:
- Class imbalance handling (PASS class: 6 samples)
- TensorBoard integration
- Model versioning
- GPU/CPU compatibility
"""

from .yolo_trainer import YOLOTrainer
from .view_classifier import ViewClassifier
from .defect_classifier import DefectClassifier

__all__ = [
    'YOLOTrainer',
    'ViewClassifier',
    'DefectClassifier'
]

__version__ = '1.0.0'
