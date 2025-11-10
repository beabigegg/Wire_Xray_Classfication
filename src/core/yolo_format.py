"""
YOLO format handler for reading and writing annotation files.

This module handles conversion between pixel coordinates used in the database
and normalized YOLO coordinates used in .txt files.
"""

from pathlib import Path
from typing import Dict, List, Tuple


class YOLOFormatError(Exception):
    """Exception raised for YOLO format errors."""
    pass


class YOLOFormatHandler:
    """
    Handler for YOLO format annotation files.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized to [0, 1] range.
    """

    # Class mappings for view types and defect types
    VIEW_TYPES = {'TOP': 0, 'SIDE': 1}
    DEFECT_TYPES = {'PASS': 0, '沖線': 1, '晃動': 2, '碰觸': 3}

    # Reverse mappings
    VIEW_TYPES_REV = {v: k for k, v in VIEW_TYPES.items()}
    DEFECT_TYPES_REV = {v: k for k, v in DEFECT_TYPES.items()}

    @staticmethod
    def read(label_file: str, image_width: int, image_height: int) -> List[Dict]:
        """
        Read YOLO format annotation file and convert to pixel coordinates.

        Args:
            label_file: Path to YOLO .txt file
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            List of annotation dictionaries with pixel coordinates:
                - bbox: dict with x, y, width, height (pixel coordinates)
                - view_type: str (TOP or SIDE)
                - defect_type: str (PASS, 沖線, 晃動, or 碰觸)
                - class_id: int (for reference)

        Raises:
            FileNotFoundError: If label file doesn't exist
            YOLOFormatError: If format is invalid
        """
        if not Path(label_file).exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        annotations = []

        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments

                    parts = line.split()
                    if len(parts) != 5:
                        raise YOLOFormatError(
                            f"Invalid YOLO format at line {line_num}: "
                            f"expected 5 values, got {len(parts)}"
                        )

                    try:
                        class_id = int(parts[0])
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])
                    except ValueError as e:
                        raise YOLOFormatError(
                            f"Invalid numeric values at line {line_num}: {e}"
                        )

                    # Validate normalized coordinates
                    if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and
                            0 < width_norm <= 1 and 0 < height_norm <= 1):
                        raise YOLOFormatError(
                            f"Coordinates out of range [0, 1] at line {line_num}"
                        )

                    # Convert to pixel coordinates
                    bbox = YOLOFormatHandler.denormalize(
                        {
                            'x_center': x_center_norm,
                            'y_center': y_center_norm,
                            'width': width_norm,
                            'height': height_norm
                        },
                        image_width,
                        image_height
                    )

                    # For now, we treat class_id as a combined identifier
                    # In a real system, you might encode both view and defect in the class_id
                    # For this MVP, we'll use default values and rely on database
                    # The class_id could be used to encode defect type primarily

                    # Map class_id to defect_type (0-3)
                    defect_type = YOLOFormatHandler.DEFECT_TYPES_REV.get(
                        class_id % 4, 'PASS'
                    )
                    # For simplicity, default view_type to TOP
                    # In practice, this should be stored in database
                    view_type = 'TOP'

                    annotations.append({
                        'bbox': bbox,
                        'view_type': view_type,
                        'defect_type': defect_type,
                        'class_id': class_id
                    })

        except Exception as e:
            if isinstance(e, (FileNotFoundError, YOLOFormatError)):
                raise
            raise YOLOFormatError(f"Failed to read YOLO file: {e}")

        return annotations

    @staticmethod
    def write(
        label_file: str,
        annotations: List[Dict],
        image_width: int,
        image_height: int
    ):
        """
        Write annotations to YOLO format file.

        Args:
            label_file: Path to output YOLO .txt file
            annotations: List of annotation dictionaries with:
                - bbox: dict with x, y, width, height (pixel coordinates)
                - view_type: str (TOP or SIDE)
                - defect_type: str (PASS, 沖線, 晃動, or 碰觸)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Raises:
            YOLOFormatError: If writing fails
        """
        try:
            # Create directory if it doesn't exist
            Path(label_file).parent.mkdir(parents=True, exist_ok=True)

            with open(label_file, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    # Get class ID from defect type
                    defect_type = ann.get('defect_type', 'PASS')
                    class_id = YOLOFormatHandler.DEFECT_TYPES.get(defect_type, 0)

                    # Normalize coordinates
                    bbox_norm = YOLOFormatHandler.normalize(
                        ann['bbox'],
                        image_width,
                        image_height
                    )

                    # Write in YOLO format
                    line = (
                        f"{class_id} "
                        f"{bbox_norm['x_center']:.6f} "
                        f"{bbox_norm['y_center']:.6f} "
                        f"{bbox_norm['width']:.6f} "
                        f"{bbox_norm['height']:.6f}\n"
                    )
                    f.write(line)

        except Exception as e:
            raise YOLOFormatError(f"Failed to write YOLO file: {e}")

    @staticmethod
    def normalize(bbox: Dict[str, float], image_width: int, image_height: int) -> Dict[str, float]:
        """
        Convert pixel coordinates to normalized YOLO format.

        Args:
            bbox: Bounding box with x, y, width, height (pixel coordinates)
                 x, y are top-left corner coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Dictionary with normalized coordinates:
                - x_center: normalized center x (0-1)
                - y_center: normalized center y (0-1)
                - width: normalized width (0-1)
                - height: normalized height (0-1)

        Raises:
            YOLOFormatError: If coordinates are invalid
        """
        if image_width <= 0 or image_height <= 0:
            raise YOLOFormatError("Image dimensions must be positive")

        if bbox['width'] <= 0 or bbox['height'] <= 0:
            raise YOLOFormatError("Bounding box dimensions must be positive")

        # Convert top-left corner to center
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2

        # Normalize
        x_center_norm = center_x / image_width
        y_center_norm = center_y / image_height
        width_norm = bbox['width'] / image_width
        height_norm = bbox['height'] / image_height

        # Clamp to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        return {
            'x_center': x_center_norm,
            'y_center': y_center_norm,
            'width': width_norm,
            'height': height_norm
        }

    @staticmethod
    def denormalize(
        bbox_norm: Dict[str, float],
        image_width: int,
        image_height: int
    ) -> Dict[str, float]:
        """
        Convert normalized YOLO coordinates to pixel coordinates.

        Args:
            bbox_norm: Normalized bounding box with:
                - x_center: normalized center x (0-1)
                - y_center: normalized center y (0-1)
                - width: normalized width (0-1)
                - height: normalized height (0-1)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Dictionary with pixel coordinates:
                - x: top-left x (pixels)
                - y: top-left y (pixels)
                - width: width (pixels)
                - height: height (pixels)

        Raises:
            YOLOFormatError: If coordinates are invalid
        """
        if image_width <= 0 or image_height <= 0:
            raise YOLOFormatError("Image dimensions must be positive")

        # Denormalize
        center_x = bbox_norm['x_center'] * image_width
        center_y = bbox_norm['y_center'] * image_height
        width = bbox_norm['width'] * image_width
        height = bbox_norm['height'] * image_height

        # Convert center to top-left corner
        x = center_x - width / 2
        y = center_y - height / 2

        # Clamp to image boundaries
        x = max(0.0, min(image_width - width, x))
        y = max(0.0, min(image_height - height, y))

        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }

    @staticmethod
    def get_class_id(view_type: str, defect_type: str) -> int:
        """
        Get YOLO class ID from view and defect types.

        For simplicity, we use defect_type as the primary class ID.
        View type is stored separately in the database.

        Args:
            view_type: View type (TOP or SIDE)
            defect_type: Defect type (PASS, 沖線, 晃動, 碰觸)

        Returns:
            Class ID (0-3 for defect types)
        """
        return YOLOFormatHandler.DEFECT_TYPES.get(defect_type, 0)

    @staticmethod
    def validate_coordinates(bbox: Dict[str, float]) -> bool:
        """
        Validate bounding box coordinates.

        Args:
            bbox: Bounding box dictionary

        Returns:
            True if valid, False otherwise
        """
        required_keys = {'x', 'y', 'width', 'height'}
        if not all(key in bbox for key in required_keys):
            return False

        # Check positive dimensions
        if bbox['width'] <= 0 or bbox['height'] <= 0:
            return False

        # Check non-negative position
        if bbox['x'] < 0 or bbox['y'] < 0:
            return False

        return True
