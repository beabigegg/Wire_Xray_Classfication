"""
Model loader for VIEW-aware inference system.

Handles loading of trained models:
- View classifier (.pth) - Classifies TOP/SIDE
- YOLO detection models (.pt) - Separate models for TOP and SIDE views
- Defect classifiers (.pth) - Separate models for TOP and SIDE views
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
import timm
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and manages VIEW-aware trained models for inference.

    Architecture:
    1. View Classification (full image) → TOP or SIDE
    2. YOLO Detection (view-specific model) → Bounding box
    3. Defect Classification (view-specific model) → Defect type
    """

    def __init__(
        self,
        view_classifier_path: Optional[str] = None,
        yolo_top_path: Optional[str] = None,
        yolo_side_path: Optional[str] = None,
        defect_top_path: Optional[str] = None,
        defect_side_path: Optional[str] = None,
        # Backward compatibility parameters
        yolo_path: Optional[str] = None,
        defect_classifier_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize VIEW-aware model loader.

        Args:
            view_classifier_path: Path to View classifier model (.pth file)
            yolo_top_path: Path to YOLO detection model for TOP view (.pt file)
            yolo_side_path: Path to YOLO detection model for SIDE view (.pt file)
            defect_top_path: Path to Defect classifier for TOP view (.pth file)
            defect_side_path: Path to Defect classifier for SIDE view (.pth file)

            # Backward compatibility (old architecture)
            yolo_path: Path to unified YOLO model (used for both views if view-specific models not provided)
            defect_classifier_path: Path to unified Defect classifier (used for both views if view-specific models not provided)

            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        # Backward compatibility: use unified models if view-specific ones not provided
        self.view_classifier_path = Path(view_classifier_path) if view_classifier_path else None

        # YOLO paths
        self.yolo_top_path = Path(yolo_top_path) if yolo_top_path else (Path(yolo_path) if yolo_path else None)
        self.yolo_side_path = Path(yolo_side_path) if yolo_side_path else (Path(yolo_path) if yolo_path else None)

        # Defect classifier paths
        self.defect_top_path = Path(defect_top_path) if defect_top_path else (Path(defect_classifier_path) if defect_classifier_path else None)
        self.defect_side_path = Path(defect_side_path) if defect_side_path else (Path(defect_classifier_path) if defect_classifier_path else None)

        # Validate model paths
        self._validate_paths()

        # Determine device
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")

        # Initialize models as None
        self.view_model = None
        self.yolo_top_model = None
        self.yolo_side_model = None
        self.defect_top_model = None
        self.defect_side_model = None

        # Class mappings (must match PyTorch ImageFolder alphabetical order)
        self.view_classes = ["SIDE", "TOP"]  # Alphabetical order: SIDE=0, TOP=1
        self.defect_classes = ["PASS", "沖線", "晃動", "碰觸"]  # 0=PASS, 1=沖線, 2=晃動, 3=碰觸

    def _validate_paths(self):
        """Validate that all required model files exist."""
        if not self.view_classifier_path or not self.view_classifier_path.exists():
            raise FileNotFoundError(
                f"View classifier model not found at: {self.view_classifier_path}\n"
                f"This model is required for VIEW-aware architecture."
            )

        if not self.yolo_top_path or not self.yolo_top_path.exists():
            raise FileNotFoundError(
                f"YOLO TOP model not found at: {self.yolo_top_path}\n"
                f"Expected path: {self.yolo_top_path.absolute() if self.yolo_top_path else 'None'}"
            )

        if not self.yolo_side_path or not self.yolo_side_path.exists():
            raise FileNotFoundError(
                f"YOLO SIDE model not found at: {self.yolo_side_path}\n"
                f"Expected path: {self.yolo_side_path.absolute() if self.yolo_side_path else 'None'}"
            )

        if not self.defect_top_path or not self.defect_top_path.exists():
            raise FileNotFoundError(
                f"Defect TOP classifier not found at: {self.defect_top_path}\n"
                f"Expected path: {self.defect_top_path.absolute() if self.defect_top_path else 'None'}"
            )

        if not self.defect_side_path or not self.defect_side_path.exists():
            raise FileNotFoundError(
                f"Defect SIDE classifier not found at: {self.defect_side_path}\n"
                f"Expected path: {self.defect_side_path.absolute() if self.defect_side_path else 'None'}"
            )

    def _get_device(self, device: str) -> torch.device:
        """
        Determine the device to use.

        Args:
            device: Device specification ('auto', 'cuda', or 'cpu')

        Returns:
            torch.device object
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA not available, using CPU")
                return torch.device("cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available")
            return torch.device("cuda")
        elif device == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cuda', or 'cpu'")

    def load_models(self) -> Tuple[torch.nn.Module, YOLO, YOLO, torch.nn.Module, torch.nn.Module]:
        """
        Load all VIEW-aware models.

        Returns:
            Tuple of (view_model, yolo_top_model, yolo_side_model, defect_top_model, defect_side_model)
        """
        logger.info("Loading VIEW-aware models...")

        # Load View classifier (always needed first)
        self.view_model = self._load_view_classifier()

        # Load YOLO models (TOP and SIDE)
        self.yolo_top_model = self._load_yolo(self.yolo_top_path, "TOP")
        self.yolo_side_model = self._load_yolo(self.yolo_side_path, "SIDE")

        # Load Defect classifiers (TOP and SIDE)
        self.defect_top_model = self._load_defect_classifier(self.defect_top_path, "TOP")
        self.defect_side_model = self._load_defect_classifier(self.defect_side_path, "SIDE")

        logger.info("All VIEW-aware models loaded successfully")
        return self.view_model, self.yolo_top_model, self.yolo_side_model, self.defect_top_model, self.defect_side_model

    def _load_yolo(self, model_path: Path, view_type: str) -> YOLO:
        """Load view-specific YOLO detection model."""
        logger.info(f"Loading YOLO {view_type} model from {model_path}")
        try:
            model = YOLO(str(model_path))
            # Move to device (YOLO handles this internally)
            if self.device.type == "cuda":
                model.to(self.device)
            logger.info(f"YOLO {view_type} model loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO {view_type} model: {e}")

    def _load_view_classifier(self) -> torch.nn.Module:
        """Load View classifier model (dynamically determines architecture from checkpoint)."""
        logger.info(f"Loading View classifier from {self.view_classifier_path}")
        try:
            # Load checkpoint first to determine model architecture
            checkpoint = torch.load(self.view_classifier_path, map_location=self.device)

            # Try to get model_name from checkpoint (fallback to resnet18 for backward compatibility)
            if 'model_name' in checkpoint:
                model_name = checkpoint['model_name']
                logger.info(f"Creating model architecture: {model_name}")
            else:
                model_name = 'resnet18'
                logger.warning(f"model_name not found in checkpoint, defaulting to {model_name}")

            # Create model architecture with 2 classes (TOP, SIDE)
            model = timm.create_model(model_name, pretrained=False, num_classes=2)

            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()

            logger.info(f"View classifier ({model_name}) loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load View classifier: {e}")

    def _load_defect_classifier(self, model_path: Path, view_type: str) -> torch.nn.Module:
        """Load view-specific Defect classifier model (dynamically determines architecture from checkpoint)."""
        logger.info(f"Loading Defect {view_type} classifier from {model_path}")
        try:
            # Load checkpoint first to determine model architecture
            checkpoint = torch.load(model_path, map_location=self.device)

            # Try to get model_name from checkpoint (fallback to efficientnet_b0 for backward compatibility)
            if 'model_name' in checkpoint:
                model_name = checkpoint['model_name']
                logger.info(f"Creating model architecture: {model_name}")
            else:
                model_name = 'efficientnet_b0'
                logger.warning(f"model_name not found in checkpoint, defaulting to {model_name}")

            # Create model architecture with 4 classes (PASS, 沖線, 晃動, 碰觸)
            model = timm.create_model(model_name, pretrained=False, num_classes=4)

            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()

            logger.info(f"Defect {view_type} classifier ({model_name}) loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Defect {view_type} classifier: {e}")

    def get_yolo_model(self, view: str) -> YOLO:
        """
        Get the appropriate YOLO model based on view type.

        Args:
            view: View type ('TOP' or 'SIDE')

        Returns:
            YOLO model for the specified view
        """
        if view.upper() == 'TOP':
            return self.yolo_top_model
        elif view.upper() == 'SIDE':
            return self.yolo_side_model
        else:
            raise ValueError(f"Invalid view type: {view}. Must be 'TOP' or 'SIDE'")

    def get_defect_model(self, view: str) -> torch.nn.Module:
        """
        Get the appropriate Defect classifier based on view type.

        Args:
            view: View type ('TOP' or 'SIDE')

        Returns:
            Defect classifier model for the specified view
        """
        if view.upper() == 'TOP':
            return self.defect_top_model
        elif view.upper() == 'SIDE':
            return self.defect_side_model
        else:
            raise ValueError(f"Invalid view type: {view}. Must be 'TOP' or 'SIDE'")

    def get_view_class_name(self, class_idx: int) -> str:
        """Get view class name from index."""
        return self.view_classes[class_idx]

    def get_defect_class_name(self, class_idx: int) -> str:
        """Get defect class name from index."""
        return self.defect_classes[class_idx]
