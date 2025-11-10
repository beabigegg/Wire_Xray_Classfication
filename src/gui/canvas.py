"""
Annotation canvas widget for drawing and editing bounding boxes.

This module provides a PyQt6 widget for displaying images and
managing bounding box annotations with mouse and keyboard interaction.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap, QImage, QPen, QColor, QFont, QBrush


class BoundingBox:
    """Represents a bounding box annotation."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        annotation_id: Optional[int] = None,
        view_type: str = "TOP",
        defect_type: str = "PASS"
    ):
        """
        Initialize bounding box.

        Args:
            x: Top-left x coordinate (pixels)
            y: Top-left y coordinate (pixels)
            width: Width (pixels)
            height: Height (pixels)
            annotation_id: Database annotation ID
            view_type: View type (TOP or SIDE)
            defect_type: Defect type
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.annotation_id = annotation_id
        self.view_type = view_type
        self.defect_type = defect_type

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }

    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside bounding box."""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def get_rect(self) -> QRect:
        """Get QRect representation."""
        return QRect(int(self.x), int(self.y), int(self.width), int(self.height))


class AnnotationCanvas(QWidget):
    """
    Canvas widget for image display and bounding box annotation.

    Supports:
    - Image display with zoom and pan
    - Drawing new bounding boxes
    - Selecting and editing existing bounding boxes
    - Deleting bounding boxes
    """

    # Signals
    bbox_created = pyqtSignal(dict)  # Emitted when new bbox is created
    bbox_selected = pyqtSignal(object)  # Emitted when bbox is selected
    bbox_modified = pyqtSignal(object)  # Emitted when bbox is modified
    bbox_deleted = pyqtSignal(object)  # Emitted when bbox is deleted

    def __init__(self, parent=None):
        """Initialize canvas widget."""
        super().__init__(parent)

        # Canvas state
        self.image: Optional[np.ndarray] = None
        self.pixmap: Optional[QPixmap] = None
        self.bboxes: List[BoundingBox] = []
        self.selected_bbox: Optional[BoundingBox] = None

        # Drawing state
        self.is_drawing = False
        self.draw_start: Optional[QPoint] = None
        self.draw_current: Optional[QPoint] = None

        # Interaction mode
        self.drawing_enabled = False

        # Colors and styling
        self.bbox_color = QColor("#00ff00")
        self.bbox_selected_color = QColor("#ff0000")
        self.bbox_line_width = 2
        self.label_font_size = 10

        # Widget setup
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_image(self, image: np.ndarray):
        """
        Load and display an image.

        Args:
            image: Grayscale image as numpy array (H, W)
        """
        self.image = image
        self.bboxes.clear()
        self.selected_bbox = None
        self.is_drawing = False

        # Convert numpy array to QPixmap
        if len(image.shape) == 2:
            # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(
                image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
        else:
            # Should not happen, but handle RGB just in case
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(
                image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )

        self.pixmap = QPixmap.fromImage(q_image)
        self.update()

    def add_bbox(self, bbox_dict: Dict, annotation_id: Optional[int] = None):
        """
        Add a bounding box to the canvas.

        Args:
            bbox_dict: Dictionary with x, y, width, height, view_type, defect_type
            annotation_id: Database annotation ID
        """
        bbox = BoundingBox(
            x=bbox_dict['x'],
            y=bbox_dict['y'],
            width=bbox_dict['width'],
            height=bbox_dict['height'],
            annotation_id=annotation_id,
            view_type=bbox_dict.get('view_type', 'TOP'),
            defect_type=bbox_dict.get('defect_type', 'PASS')
        )
        self.bboxes.append(bbox)
        self.update()

    def clear_bboxes(self):
        """Remove all bounding boxes."""
        self.bboxes.clear()
        self.selected_bbox = None
        self.update()

    def delete_selected_bbox(self):
        """Delete the currently selected bounding box."""
        if self.selected_bbox and self.selected_bbox in self.bboxes:
            self.bboxes.remove(self.selected_bbox)
            self.bbox_deleted.emit(self.selected_bbox)
            self.selected_bbox = None
            self.update()

    def enable_drawing(self, enabled: bool = True):
        """
        Enable or disable drawing mode.

        Args:
            enabled: True to enable drawing, False to disable
        """
        self.drawing_enabled = enabled
        if not enabled:
            self.is_drawing = False
            self.draw_start = None
            self.draw_current = None
            self.update()

    def paintEvent(self, event):
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw image
        if self.pixmap:
            painter.drawPixmap(0, 0, self.pixmap)

        # Draw existing bounding boxes
        for bbox in self.bboxes:
            is_selected = (bbox == self.selected_bbox)
            self._draw_bbox(painter, bbox, is_selected)

        # Draw current drawing bbox
        if self.is_drawing and self.draw_start and self.draw_current:
            self._draw_drawing_bbox(painter)

    def _draw_bbox(self, painter: QPainter, bbox: BoundingBox, is_selected: bool):
        """
        Draw a bounding box with label.

        Args:
            painter: QPainter instance
            bbox: BoundingBox to draw
            is_selected: Whether this bbox is selected
        """
        # Set pen color
        color = self.bbox_selected_color if is_selected else self.bbox_color
        pen = QPen(color, self.bbox_line_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Draw rectangle
        rect = bbox.get_rect()
        painter.drawRect(rect)

        # Draw label
        label_text = f"{bbox.view_type} - {bbox.defect_type}"
        font = QFont("Arial", self.label_font_size)
        painter.setFont(font)

        # Label background
        label_rect = painter.fontMetrics().boundingRect(label_text)
        label_rect.moveTo(int(bbox.x), int(bbox.y) - label_rect.height() - 2)
        painter.fillRect(label_rect, QBrush(color))

        # Label text
        painter.setPen(QPen(QColor("#000000")))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)

    def _draw_drawing_bbox(self, painter: QPainter):
        """Draw the bounding box being drawn."""
        pen = QPen(self.bbox_color, self.bbox_line_width, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        x1 = min(self.draw_start.x(), self.draw_current.x())
        y1 = min(self.draw_start.y(), self.draw_current.y())
        x2 = max(self.draw_start.x(), self.draw_current.x())
        y2 = max(self.draw_start.y(), self.draw_current.y())

        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.pos()

        if self.drawing_enabled:
            # Start drawing new bbox
            self.is_drawing = True
            self.draw_start = pos
            self.draw_current = pos
        else:
            # Check if clicking on existing bbox
            clicked_bbox = self._find_bbox_at(pos.x(), pos.y())
            if clicked_bbox:
                self.selected_bbox = clicked_bbox
                self.bbox_selected.emit(clicked_bbox)
            else:
                self.selected_bbox = None

        self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.is_drawing:
            self.draw_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.is_drawing and self.draw_start and self.draw_current:
            # Finish drawing
            x1 = min(self.draw_start.x(), self.draw_current.x())
            y1 = min(self.draw_start.y(), self.draw_current.y())
            x2 = max(self.draw_start.x(), self.draw_current.x())
            y2 = max(self.draw_start.y(), self.draw_current.y())

            width = x2 - x1
            height = y2 - y1

            # Validate bbox: check size, positivity, and image boundaries
            is_valid = self._validate_bbox(x1, y1, width, height)

            if is_valid:
                bbox_dict = {
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(width),
                    'height': float(height),
                    'view_type': 'TOP',
                    'defect_type': 'PASS'
                }
                self.bbox_created.emit(bbox_dict)

            # Reset drawing state
            self.is_drawing = False
            self.draw_start = None
            self.draw_current = None
            self.drawing_enabled = False
            self.update()

    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() == Qt.Key.Key_Delete and self.selected_bbox:
            self.delete_selected_bbox()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel drawing
            if self.is_drawing:
                self.is_drawing = False
                self.draw_start = None
                self.draw_current = None
                self.drawing_enabled = False
                self.update()

    def _validate_bbox(self, x: float, y: float, width: float, height: float) -> bool:
        """
        Validate bounding box coordinates and dimensions.

        Args:
            x: Top-left X coordinate
            y: Top-left Y coordinate
            width: Box width
            height: Box height

        Returns:
            True if bbox is valid, False otherwise
        """
        # Require image to be loaded for validation
        if self.image is None:
            return False

        # Check minimum size
        if width < 10 or height < 10:
            return False

        # Check positivity
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return False

        # Check image boundaries if image is loaded
        if self.image is not None:
            # self.image is a NumPy array with shape (height, width, channels)
            img_height, img_width = self.image.shape[:2]

            # Ensure bbox doesn't overflow image bounds
            if x >= img_width or y >= img_height:
                return False
            if (x + width) > img_width or (y + height) > img_height:
                return False

        return True

    def _find_bbox_at(self, x: float, y: float) -> Optional[BoundingBox]:
        """
        Find bounding box at given position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            BoundingBox if found, None otherwise
        """
        # Search in reverse order (last drawn on top)
        for bbox in reversed(self.bboxes):
            if bbox.contains(x, y):
                return bbox
        return None

    def update_selected_bbox_labels(self, view_type: str, defect_type: str):
        """
        Update labels of selected bounding box.

        Args:
            view_type: New view type
            defect_type: New defect type
        """
        if self.selected_bbox:
            self.selected_bbox.view_type = view_type
            self.selected_bbox.defect_type = defect_type
            self.bbox_modified.emit(self.selected_bbox)
            self.update()

    def get_bboxes(self) -> List[Dict]:
        """
        Get all bounding boxes as dictionaries.

        Returns:
            List of bbox dictionaries
        """
        return [
            {
                'x': bbox.x,
                'y': bbox.y,
                'width': bbox.width,
                'height': bbox.height,
                'view_type': bbox.view_type,
                'defect_type': bbox.defect_type,
                'annotation_id': bbox.annotation_id
            }
            for bbox in self.bboxes
        ]
