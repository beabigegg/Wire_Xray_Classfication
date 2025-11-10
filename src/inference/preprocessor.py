"""
Image preprocessing for inference.

Handles preprocessing for all three model stages:
- YOLO detection
- View classification
- Defect classification
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for inference."""

    def __init__(self, expected_size: int = 1004):
        """
        Initialize preprocessor.

        Args:
            expected_size: Expected input image size (default: 1004x1004)
        """
        self.expected_size = expected_size

        # Define transforms for classifiers (ImageNet normalization)
        self.classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and validate image.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (H, W, C) in RGB format

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image dimensions are invalid
            RuntimeError: If image cannot be decoded
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                if image.mode == 'L':  # Grayscale
                    logger.debug(f"Converting grayscale image to RGB: {image_path.name}")
                    image = image.convert('RGB')
                elif image.mode == 'RGBA':
                    logger.debug(f"Converting RGBA image to RGB: {image_path.name}")
                    image = image.convert('RGB')
                else:
                    logger.warning(f"Converting {image.mode} image to RGB: {image_path.name}")
                    image = image.convert('RGB')

            # Validate dimensions
            width, height = image.size
            if width != self.expected_size or height != self.expected_size:
                raise ValueError(
                    f"Invalid image dimensions: {width}x{height}. "
                    f"Expected: {self.expected_size}x{self.expected_size}"
                )

            # Convert to numpy array
            image_array = np.array(image)

            logger.debug(f"Loaded image: {image_path.name}, shape: {image_array.shape}")
            return image_array

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise RuntimeError(f"Failed to decode image {image_path}: {e}")

    def preprocess_for_yolo(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO detection.

        Args:
            image: Image array (H, W, C) in RGB format

        Returns:
            Preprocessed image as numpy array
        """
        # YOLO handles preprocessing internally, just return the image
        return image

    def preprocess_for_classifier(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for View/Defect classifiers.

        Args:
            image: Image array (H, W, C) in RGB format

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Apply transforms
        tensor = self.classifier_transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def crop_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop image to bounding box.

        Args:
            image: Full image array (H, W, C)
            bbox: Bounding box (x1, y1, x2, y2) in pixels

        Returns:
            Cropped image array
        """
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Crop image
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            raise ValueError(f"Invalid bounding box results in empty crop: ({x1}, {y1}, {x2}, {y2})")

        return cropped

    @staticmethod
    def validate_image_format(image_path: str) -> bool:
        """
        Check if image format is supported.

        Args:
            image_path: Path to image file

        Returns:
            True if format is supported
        """
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp'}
        ext = Path(image_path).suffix.lower()
        return ext in supported_formats
