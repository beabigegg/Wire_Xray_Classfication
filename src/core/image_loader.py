"""
Image loader module for X-ray images.

This module handles loading, validation, and preprocessing of X-ray images
for the Wire Loop annotation system.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


class ImageLoadError(Exception):
    """Exception raised when image loading fails."""
    pass


class ValidationError(Exception):
    """Exception raised when image validation fails."""
    pass


class ImageLoader:
    """
    Image loader for X-ray images.

    Handles loading PNG, JPEG, and BMP images, validates dimensions,
    and converts 24-bit RGB to grayscale.
    """

    EXPECTED_WIDTH = 1004
    EXPECTED_HEIGHT = 1004
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp'}

    @staticmethod
    def load(filepath: str) -> np.ndarray:
        """
        Load an X-ray image and convert to grayscale.

        Args:
            filepath: Path to the image file

        Returns:
            Grayscale image as numpy array with shape (1004, 1004)

        Raises:
            FileNotFoundError: If the file does not exist
            ImageLoadError: If the image cannot be loaded
            ValidationError: If the image dimensions are invalid
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")

        # Check file extension
        ext = Path(filepath).suffix.lower()
        if ext not in ImageLoader.SUPPORTED_FORMATS:
            raise ImageLoadError(
                f"Unsupported image format: {ext}. "
                f"Supported formats: {', '.join(ImageLoader.SUPPORTED_FORMATS)}"
            )

        try:
            # OpenCV on Windows has issues with non-ASCII (e.g., Chinese) paths
            # Use numpy + cv2.imdecode to handle Unicode paths correctly
            try:
                # Read file as binary
                with open(filepath, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                # Decode image from bytes
                image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            except Exception:
                # Fallback to standard cv2.imread for ASCII paths
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is None:
                # Try with PIL as fallback
                try:
                    pil_image = Image.open(filepath).convert('L')
                    image = np.array(pil_image)
                except Exception as e:
                    raise ImageLoadError(f"Failed to load image with both CV2 and PIL: {e}")

        except Exception as e:
            raise ImageLoadError(f"Failed to load image: {e}")

        # Validate dimensions
        if not ImageLoader.validate(image):
            raise ValidationError(
                f"Invalid image dimensions: {image.shape}. "
                f"Expected: ({ImageLoader.EXPECTED_HEIGHT}, {ImageLoader.EXPECTED_WIDTH})"
            )

        return image

    @staticmethod
    def validate(image: np.ndarray) -> bool:
        """
        Validate image dimensions.

        Args:
            image: Image as numpy array

        Returns:
            True if valid, False otherwise
        """
        if image is None or not isinstance(image, np.ndarray):
            return False

        # Check dimensions
        if len(image.shape) == 2:
            # Grayscale image
            height, width = image.shape
        elif len(image.shape) == 3:
            # Color image (should not happen after our loading, but handle it)
            height, width, _ = image.shape
        else:
            return False

        return (height == ImageLoader.EXPECTED_HEIGHT and
                width == ImageLoader.EXPECTED_WIDTH)

    @staticmethod
    def get_metadata(filepath: str) -> Dict[str, any]:
        """
        Get image metadata.

        Args:
            filepath: Path to the image file

        Returns:
            Dictionary containing metadata:
                - filename: Filename without path
                - filepath: Full file path
                - width: Image width
                - height: Image height
                - size_bytes: File size in bytes
                - format: Image format (extension)

        Raises:
            FileNotFoundError: If the file does not exist
            ImageLoadError: If metadata cannot be extracted
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")

        try:
            # Get file info
            path_obj = Path(filepath)
            filename = path_obj.name
            file_format = path_obj.suffix.lower()
            size_bytes = path_obj.stat().st_size

            # Get image dimensions using PIL (faster than loading full image)
            with Image.open(filepath) as img:
                width, height = img.size

            return {
                'filename': filename,
                'filepath': str(path_obj.absolute()),
                'width': width,
                'height': height,
                'size_bytes': size_bytes,
                'format': file_format
            }

        except Exception as e:
            raise ImageLoadError(f"Failed to get image metadata: {e}")

    @staticmethod
    def load_with_metadata(filepath: str) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Load image and get metadata in one call.

        Args:
            filepath: Path to the image file

        Returns:
            Tuple of (image array, metadata dict)

        Raises:
            FileNotFoundError: If the file does not exist
            ImageLoadError: If the image cannot be loaded
            ValidationError: If the image dimensions are invalid
        """
        metadata = ImageLoader.get_metadata(filepath)
        image = ImageLoader.load(filepath)
        return image, metadata

    @staticmethod
    def is_supported_format(filepath: str) -> bool:
        """
        Check if file format is supported.

        Args:
            filepath: Path to check

        Returns:
            True if format is supported, False otherwise
        """
        ext = Path(filepath).suffix.lower()
        return ext in ImageLoader.SUPPORTED_FORMATS
