"""
Main annotation window for the Wire Loop annotation tool.

This module provides the main GUI window with image list, canvas,
and properties panel for annotating X-ray images.
"""

import os
from pathlib import Path
from typing import List, Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QRadioButton, QButtonGroup,
    QLabel, QFileDialog, QMessageBox, QGroupBox, QStatusBar, QToolBar,
    QComboBox, QDialog
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QKeySequence, QColor

from src.core.image_loader import ImageLoader, ImageLoadError, ValidationError
from src.core.database import Database, DatabaseError
from src.core.yolo_format import YOLOFormatHandler
from src.core.config import Config
from src.gui.canvas import AnnotationCanvas
from src.gui.tensorboard_manager import TensorBoardManager
from src.gui.training_dialog import TrainingDialog
from src.gui.model_selector_dialog import ModelSelectorDialog
from src.gui.model_comparison_dialog import ModelComparisonDialog
from src.gui.annotator_dialog import AnnotatorDialog
from src.inference.model_loader import ModelLoader
from src.inference.inference_pipeline import InferencePipeline


class AnnotationWindow(QMainWindow):
    """
    Main window for annotation tool.

    Provides three-panel layout:
    - Left: Image list
    - Center: Annotation canvas
    - Right: Properties panel (view type, defect type)
    """

    def __init__(self, config: Config, db: Database):
        """
        Initialize annotation window.

        Args:
            config: Configuration object
            db: Database instance
        """
        super().__init__()

        self.config = config
        self.db = db

        # State
        self.current_folder: Optional[str] = None
        self.image_files: List[str] = []
        self.current_image_index: int = -1
        self.current_image_id: Optional[int] = None
        self.annotator: str = config.default_annotator

        # GUI integration components
        self.tensorboard_manager = TensorBoardManager()
        self.training_dialog = None
        self.model_loader = None
        self.inference_pipeline = None
        self.pending_prediction = None  # Store prediction for accept/modify

        # Setup UI
        self.setWindowTitle(config.get('gui.window_title', 'Wire Loop Annotation Tool'))
        self.setGeometry(100, 100, config.get('gui.window_width', 1600),
                        config.get('gui.window_height', 900))

        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_central_widget()
        self._create_status_bar()

        # Connect signals
        self._connect_signals()

    def _create_actions(self):
        """Create menu and toolbar actions."""
        # File actions
        self.action_open = QAction("Open Folder", self)
        self.action_open.setShortcut(QKeySequence(self.config.get('shortcuts.open', 'Ctrl+O')))
        self.action_open.triggered.connect(self.open_folder)

        self.action_save = QAction("Save", self)
        self.action_save.setShortcut(QKeySequence(self.config.get('shortcuts.save', 'S')))
        self.action_save.triggered.connect(self.save_annotations)

        self.action_quit = QAction("Quit", self)
        self.action_quit.setShortcut(QKeySequence(self.config.get('shortcuts.quit', 'Ctrl+Q')))
        self.action_quit.triggered.connect(self.close)

        # Edit actions
        self.action_delete = QAction("Delete", self)
        self.action_delete.setShortcut(QKeySequence(self.config.get('shortcuts.delete', 'Delete')))
        self.action_delete.triggered.connect(self.delete_selected_annotation)

        # Navigation actions
        self.action_next = QAction("Next Image", self)
        self.action_next.setShortcut(QKeySequence(self.config.get('shortcuts.next', 'D')))
        self.action_next.triggered.connect(self.next_image)

        self.action_prev = QAction("Previous Image", self)
        self.action_prev.setShortcut(QKeySequence(self.config.get('shortcuts.prev', 'A')))
        self.action_prev.triggered.connect(self.previous_image)

        # Drawing action
        self.action_draw = QAction("Draw Bbox", self)
        self.action_draw.setShortcut(QKeySequence(self.config.get('shortcuts.draw', 'W')))
        self.action_draw.triggered.connect(self.enable_drawing)

        # Training actions
        self.action_train_detection = QAction("Train Detection Model", self)
        self.action_train_detection.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self.action_train_detection.triggered.connect(lambda: self.open_training_dialog('detection'))

        self.action_train_view = QAction("Train View Classifier", self)
        self.action_train_view.setShortcut(QKeySequence("Ctrl+Shift+V"))
        self.action_train_view.triggered.connect(lambda: self.open_training_dialog('view'))

        self.action_train_defect = QAction("Train Defect Classifier", self)
        self.action_train_defect.setShortcut(QKeySequence("Ctrl+Shift+F"))
        self.action_train_defect.triggered.connect(lambda: self.open_training_dialog('defect'))

        self.action_train_all = QAction("Train All Models", self)
        self.action_train_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.action_train_all.triggered.connect(self.train_all_models)

        self.action_open_tensorboard = QAction("Open TensorBoard", self)
        self.action_open_tensorboard.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.action_open_tensorboard.triggered.connect(self.open_tensorboard)

        self.action_manage_models = QAction("Manage Models...", self)
        self.action_manage_models.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self.action_manage_models.triggered.connect(self.open_model_selector)

        # Model comparison action
        self.action_compare_models = QAction("Compare Models...", self)
        self.action_compare_models.setShortcut(QKeySequence("Ctrl+M"))
        self.action_compare_models.triggered.connect(self._open_model_comparison)

        # Inference actions
        self.action_infer_current = QAction("Run on Current Image", self)
        self.action_infer_current.setShortcut(QKeySequence("Ctrl+I"))
        self.action_infer_current.triggered.connect(self.infer_current_image)

        self.action_infer_batch = QAction("Run Batch Inference...", self)
        self.action_infer_batch.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.action_infer_batch.triggered.connect(self.infer_batch)

        # Settings actions
        self.action_change_annotator = QAction("Change Annotator...", self)
        self.action_change_annotator.triggered.connect(self.change_annotator)

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction(self.action_delete)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction(self.action_next)
        view_menu.addAction(self.action_prev)

        # Training menu (add before Help menu)
        training_menu = menubar.addMenu("Training")
        training_menu.addAction(self.action_train_detection)
        training_menu.addAction(self.action_train_view)
        training_menu.addAction(self.action_train_defect)
        training_menu.addAction(self.action_train_all)
        training_menu.addSeparator()
        training_menu.addAction(self.action_open_tensorboard)
        training_menu.addAction(self.action_compare_models)
        training_menu.addAction(self.action_manage_models)

        # Inference menu
        inference_menu = menubar.addMenu("Inference")
        inference_menu.addAction(self.action_infer_current)
        inference_menu.addAction(self.action_infer_batch)
        inference_menu.addSeparator()
        inference_menu.addAction(self.action_manage_models)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction(self.action_change_annotator)

        # Help menu
        help_menu = menubar.addMenu("Help")
        action_about = QAction("About", self)
        action_about.triggered.connect(self.show_about)
        help_menu.addAction(action_about)

    def _create_tool_bar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)

        # Add actions to toolbar
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_save)
        toolbar.addSeparator()
        toolbar.addAction(self.action_draw)
        toolbar.addAction(self.action_delete)
        toolbar.addSeparator()
        toolbar.addAction(self.action_prev)
        toolbar.addAction(self.action_next)

    def _create_central_widget(self):
        """Create central widget with three-panel layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        # Left panel: Image list
        left_panel = self._create_image_list_panel()
        layout.addWidget(left_panel, stretch=1)

        # Center panel: Canvas
        self.canvas = AnnotationCanvas()
        layout.addWidget(self.canvas, stretch=3)

        # Right panel: Properties
        right_panel = self._create_properties_panel()
        layout.addWidget(right_panel, stretch=1)

    def _create_image_list_panel(self) -> QWidget:
        """Create image list panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Images")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Filter and sort controls
        controls_layout = QHBoxLayout()

        # Filter dropdown
        filter_label = QLabel("Filter:")
        controls_layout.addWidget(filter_label)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Annotated", "Unannotated"])
        self.filter_combo.setCurrentText(
            self.config.get('gui.image_list_filter', 'All')
        )
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        controls_layout.addWidget(self.filter_combo)

        controls_layout.addStretch()

        # Sort dropdown
        sort_label = QLabel("Sort:")
        controls_layout.addWidget(sort_label)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Date", "Status"])
        self.sort_combo.setCurrentText(
            self.config.get('gui.image_list_sort', 'Name')
        )
        self.sort_combo.currentTextChanged.connect(self.on_sort_changed)
        controls_layout.addWidget(self.sort_combo)

        layout.addLayout(controls_layout)

        # Image list
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        layout.addWidget(self.image_list)

        # Progress label
        self.progress_label = QLabel("0 / 0 images")
        layout.addWidget(self.progress_label)

        return panel

    def _create_properties_panel(self) -> QWidget:
        """Create properties panel for annotation labels."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Properties")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # View Type group
        view_group_box = QGroupBox("View Type")
        view_layout = QVBoxLayout()
        self.view_button_group = QButtonGroup()

        self.view_buttons = {}
        for i, view_type in enumerate(self.config.view_types):
            radio = QRadioButton(view_type)
            self.view_buttons[view_type] = radio
            self.view_button_group.addButton(radio, i)
            view_layout.addWidget(radio)

            if view_type == self.config.get('annotation.default_view'):
                radio.setChecked(True)

        view_group_box.setLayout(view_layout)
        layout.addWidget(view_group_box)

        # Defect Type group
        defect_group_box = QGroupBox("Defect Type")
        defect_layout = QVBoxLayout()
        self.defect_button_group = QButtonGroup()

        self.defect_buttons = {}
        for i, defect_type in enumerate(self.config.defect_types):
            radio = QRadioButton(defect_type)
            self.defect_buttons[defect_type] = radio
            self.defect_button_group.addButton(radio, i)
            defect_layout.addWidget(radio)

            if defect_type == self.config.get('annotation.default_defect'):
                radio.setChecked(True)

        defect_group_box.setLayout(defect_layout)
        layout.addWidget(defect_group_box)

        # Apply button
        self.apply_button = QPushButton("Apply Labels")
        self.apply_button.clicked.connect(self.apply_labels_to_selected)
        layout.addWidget(self.apply_button)

        # Spacer
        layout.addStretch()

        # Connect signals
        self.view_button_group.buttonClicked.connect(self.on_labels_changed)
        self.defect_button_group.buttonClicked.connect(self.on_labels_changed)

        return panel

    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Left side: general messages
        self.status_bar.showMessage("Ready")

        # Right side: permanent annotator label
        self.annotator_label = QLabel(f"Annotator: {self.annotator}")
        self.annotator_label.setStyleSheet("QLabel { padding: 2px 10px; background-color: #e8f4f8; border-radius: 3px; }")
        self.status_bar.addPermanentWidget(self.annotator_label)

    def _connect_signals(self):
        """Connect canvas signals."""
        self.canvas.bbox_created.connect(self.on_bbox_created)
        self.canvas.bbox_selected.connect(self.on_bbox_selected)
        self.canvas.bbox_modified.connect(self.on_bbox_modified)
        self.canvas.bbox_deleted.connect(self.on_bbox_deleted)

    def open_folder(self):
        """Open folder dialog and load images."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            self.config.data_dir
        )

        if not folder:
            return

        self.current_folder = folder
        self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder: str):
        """
        Load all supported images from folder.

        Args:
            folder: Path to folder containing images
        """
        self.image_files.clear()
        self.image_list.clear()

        # Find all supported image files
        supported_exts = self.config.get('image.supported_formats', [])
        for ext in supported_exts:
            pattern = f"*{ext}"
            files = list(Path(folder).glob(pattern))
            self.image_files.extend([str(f) for f in files])

        # Apply initial sort
        self._sort_image_files()

        # Populate list widget with annotation info
        self._populate_image_list()

        # Update progress
        self.update_progress_label()

        # Load first image
        if self.image_files:
            self.image_list.setCurrentRow(0)
            self.status_bar.showMessage(f"Loaded {len(self.image_files)} images from {folder}")
        else:
            self.status_bar.showMessage(f"No images found in {folder}")

    def _sort_image_files(self):
        """Sort image files based on current sort option."""
        sort_by = self.sort_combo.currentText() if hasattr(self, 'sort_combo') else 'Name'

        if sort_by == 'Name':
            self.image_files.sort()
        elif sort_by == 'Date':
            # Sort by file modification time
            self.image_files.sort(key=lambda f: os.path.getmtime(f))
        elif sort_by == 'Status':
            # Sort by annotation status (annotated first)
            def sort_key(filepath):
                img_record = self.db.get_image_by_filepath(filepath)
                if img_record:
                    count = self.db.get_annotation_count(img_record['id'])
                    return (0 if count > 0 else 1, os.path.basename(filepath))
                return (1, os.path.basename(filepath))

            try:
                self.image_files.sort(key=sort_key)
            except Exception:
                # Fallback to name sorting if database query fails
                self.image_files.sort()

    def _populate_image_list(self):
        """Populate image list with annotation info."""
        self.image_list.clear()

        filter_by = self.filter_combo.currentText() if hasattr(self, 'filter_combo') else 'All'

        for filepath in self.image_files:
            filename = os.path.basename(filepath)

            # Get annotation count
            annotation_count = 0
            try:
                img_record = self.db.get_image_by_filepath(filepath)
                if img_record:
                    annotation_count = self.db.get_annotation_count(img_record['id'])
            except Exception:
                # If database query fails, just show without count
                pass

            # Apply filter
            is_annotated = annotation_count > 0
            if filter_by == 'Annotated' and not is_annotated:
                continue
            elif filter_by == 'Unannotated' and is_annotated:
                continue

            # Create list item with annotation info
            checkmark = "✓ " if is_annotated else "  "
            display_text = f"{checkmark}{filename} ({annotation_count})"

            item = QListWidgetItem(display_text)
            # Store filepath as item data for easy retrieval
            item.setData(Qt.ItemDataRole.UserRole, filepath)

            # Color-code annotated items
            if is_annotated:
                item.setForeground(QColor('#4CAF50'))  # Green for annotated

            self.image_list.addItem(item)

    def on_filter_changed(self, filter_text: str):
        """Handle filter combo box change."""
        # Save preference
        self.config.set('gui.image_list_filter', filter_text)
        self.config.save()

        # Repopulate list
        self._populate_image_list()
        self.update_progress_label()

        # Select first item if available
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    def on_sort_changed(self, sort_text: str):
        """Handle sort combo box change."""
        # Save preference
        self.config.set('gui.image_list_sort', sort_text)
        self.config.save()

        # Re-sort and repopulate
        self._sort_image_files()
        self._populate_image_list()
        self.update_progress_label()

        # Select first item if available
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    def _update_list_item_display(self, item: QListWidgetItem):
        """
        Update a list item's display with current annotation info.

        Args:
            item: The list item to update
        """
        if not item:
            return

        filepath = item.data(Qt.ItemDataRole.UserRole)
        if not filepath:
            return

        filename = os.path.basename(filepath)

        # Get current annotation count
        annotation_count = 0
        try:
            img_record = self.db.get_image_by_filepath(filepath)
            if img_record:
                annotation_count = self.db.get_annotation_count(img_record['id'])
        except Exception:
            pass

        # Update display text
        is_annotated = annotation_count > 0
        checkmark = "✓ " if is_annotated else "  "
        display_text = f"{checkmark}{filename} ({annotation_count})"
        item.setText(display_text)

        # Update color
        if is_annotated:
            item.setForeground(QColor('#4CAF50'))  # Green for annotated
        else:
            item.setForeground(QColor('#000000'))  # Black for unannotated

    def on_image_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """
        Handle image selection change.

        Args:
            current: Current list item
            previous: Previous list item
        """
        if previous:
            # Save annotations for previous image
            self.save_current_image_annotations()
            # Update the previous item's display to show new annotation count
            self._update_list_item_display(previous)
            # Update progress counter after saving
            self.update_progress_label()

        if not current:
            return

        # Get filepath from item data
        filepath = current.data(Qt.ItemDataRole.UserRole)
        if filepath:
            # Update current_image_index to match filepath
            if filepath in self.image_files:
                self.current_image_index = self.image_files.index(filepath)
            self.load_image(filepath)

    def load_image(self, filepath: str):
        """
        Load and display an image.

        Args:
            filepath: Path to image file
        """
        try:
            # Load image
            image = ImageLoader.load(filepath)

            # Add to database if not exists
            # IMPORTANT: Use filepath (not filename) to avoid conflicts
            # when different folders contain files with the same name
            filename = os.path.basename(filepath)
            img_record = self.db.get_image_by_filepath(filepath)

            if img_record:
                self.current_image_id = img_record['id']
            else:
                self.current_image_id = self.db.add_image(
                    filename=filename,
                    filepath=filepath,
                    width=image.shape[1],
                    height=image.shape[0]
                )

            # Display image
            self.canvas.load_image(image)

            # Load existing annotations
            self.load_annotations()

            self.status_bar.showMessage(f"Loaded: {filename}")

        except (ImageLoadError, ValidationError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
        except DatabaseError as e:
            QMessageBox.critical(self, "Database Error", f"Database error: {e}")

    def load_annotations(self):
        """Load annotations for current image from database."""
        if not self.current_image_id:
            return

        try:
            # Clear canvas
            self.canvas.clear_bboxes()

            # Load from database
            annotations = self.db.get_annotations(self.current_image_id)

            # Add to canvas
            for ann in annotations:
                bbox_dict = {
                    'x': ann['bbox_x'],
                    'y': ann['bbox_y'],
                    'width': ann['bbox_width'],
                    'height': ann['bbox_height'],
                    'view_type': ann['view_type'],
                    'defect_type': ann['defect_type']
                }
                self.canvas.add_bbox(bbox_dict, ann['id'])

        except DatabaseError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load annotations: {e}")

    def save_current_image_annotations(self):
        """Save annotations for current image to database and YOLO file."""
        if not self.current_image_id:
            return

        try:
            # Get all bboxes from canvas
            bboxes = self.canvas.get_bboxes()

            # Validate all bboxes before saving
            invalid_bboxes = []
            for bbox in bboxes:
                if not self._validate_bbox_data(bbox):
                    invalid_bboxes.append(bbox)

            # Warn user if invalid bboxes detected
            if invalid_bboxes:
                QMessageBox.warning(
                    self,
                    "Invalid Annotations",
                    f"Detected {len(invalid_bboxes)} invalid bounding box(es) with negative or overflowing coordinates. These will be skipped."
                )
                # Filter out invalid bboxes
                bboxes = [bbox for bbox in bboxes if self._validate_bbox_data(bbox)]

            # Get current annotations from database
            existing_anns = self.db.get_annotations(self.current_image_id)
            existing_ids = {ann['id'] for ann in existing_anns}

            # Track which annotations still exist
            current_ids = set()

            # Update or create annotations
            for bbox in bboxes:
                ann_id = bbox.get('annotation_id')

                if ann_id and ann_id in existing_ids:
                    # Update existing
                    self.db.update_annotation(
                        ann_id,
                        bbox_x=bbox['x'],
                        bbox_y=bbox['y'],
                        bbox_width=bbox['width'],
                        bbox_height=bbox['height'],
                        view_type=bbox['view_type'],
                        defect_type=bbox['defect_type']
                    )
                    current_ids.add(ann_id)
                else:
                    # Create new
                    new_id = self.db.add_annotation(
                        image_id=self.current_image_id,
                        bbox={
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height']
                        },
                        view_type=bbox['view_type'],
                        defect_type=bbox['defect_type'],
                        annotator=self.annotator
                    )
                    current_ids.add(new_id)

            # Delete removed annotations
            for ann_id in existing_ids - current_ids:
                self.db.delete_annotation(ann_id)

            # Write YOLO file
            self.write_yolo_file()

        except DatabaseError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to save annotations: {e}")

    def _validate_bbox_data(self, bbox: dict) -> bool:
        """
        Validate bounding box data before persistence.

        Args:
            bbox: Bounding box dictionary with x, y, width, height

        Returns:
            True if bbox is valid, False otherwise
        """
        try:
            x = float(bbox.get('x', -1))
            y = float(bbox.get('y', -1))
            width = float(bbox.get('width', 0))
            height = float(bbox.get('height', 0))

            # Check positivity
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                return False

            # Check minimum size
            if width <= 10 or height <= 10:
                return False

            # Check image boundaries against actual image dimensions
            if self.current_image_id:
                img_info = self.db.get_image_by_id(self.current_image_id)
                if img_info:
                    img_width = img_info['width']
                    img_height = img_info['height']

                    # Ensure bbox is within image boundaries
                    if x >= img_width or y >= img_height:
                        return False
                    if (x + width) > img_width or (y + height) > img_height:
                        return False

            return True

        except (ValueError, TypeError, KeyError):
            return False

    def write_yolo_file(self):
        """Write annotations to YOLO format file."""
        if not self.current_image_id:
            return

        try:
            # Get image info
            img_record = self.db.get_image_by_id(self.current_image_id)
            if not img_record:
                return

            # Get annotations
            annotations = self.db.get_annotations(self.current_image_id)

            # Convert to YOLO format
            yolo_annotations = []
            for ann in annotations:
                yolo_annotations.append({
                    'bbox': {
                        'x': ann['bbox_x'],
                        'y': ann['bbox_y'],
                        'width': ann['bbox_width'],
                        'height': ann['bbox_height']
                    },
                    'view_type': ann['view_type'],
                    'defect_type': ann['defect_type']
                })

            # Write to file with unique name based on image_id to avoid conflicts
            # when multiple folders contain files with the same name
            filename_without_ext = Path(img_record['filename']).stem
            # Use image_id prefix to ensure uniqueness
            unique_filename = f"img_{img_record['id']}_{filename_without_ext}.txt"
            label_file = os.path.join(
                self.config.labels_dir,
                unique_filename
            )

            YOLOFormatHandler.write(
                label_file,
                yolo_annotations,
                img_record['width'],
                img_record['height']
            )

        except Exception as e:
            print(f"Warning: Failed to write YOLO file: {e}")

    def save_annotations(self):
        """Save current annotations (triggered by Save action)."""
        self.save_current_image_annotations()
        self.update_progress_label()
        self.status_bar.showMessage("Annotations saved")

    def on_bbox_created(self, bbox_dict: dict):
        """
        Handle new bbox creation.

        Args:
            bbox_dict: Bounding box dictionary from canvas
        """
        # Apply current labels
        view_type = self.get_selected_view_type()
        defect_type = self.get_selected_defect_type()

        bbox_dict['view_type'] = view_type
        bbox_dict['defect_type'] = defect_type

        # Add to canvas
        self.canvas.add_bbox(bbox_dict)

        self.status_bar.showMessage(f"Created bbox: {view_type} - {defect_type}")

    def on_bbox_selected(self, bbox):
        """
        Handle bbox selection.

        Args:
            bbox: Selected BoundingBox object
        """
        # Update radio buttons to match selected bbox
        self.set_view_type(bbox.view_type)
        self.set_defect_type(bbox.defect_type)

        self.status_bar.showMessage(f"Selected: {bbox.view_type} - {bbox.defect_type}")

    def on_bbox_modified(self, bbox):
        """Handle bbox modification."""
        self.status_bar.showMessage("Bbox modified")

    def on_bbox_deleted(self, bbox):
        """Handle bbox deletion."""
        self.status_bar.showMessage("Bbox deleted")

    def on_labels_changed(self):
        """Handle label change in properties panel."""
        # Auto-apply to selected bbox if any
        if self.canvas.selected_bbox:
            view_type = self.get_selected_view_type()
            defect_type = self.get_selected_defect_type()
            self.canvas.update_selected_bbox_labels(view_type, defect_type)

    def apply_labels_to_selected(self):
        """Apply current labels to selected bbox."""
        if not self.canvas.selected_bbox:
            self.status_bar.showMessage("No bbox selected")
            return

        view_type = self.get_selected_view_type()
        defect_type = self.get_selected_defect_type()
        self.canvas.update_selected_bbox_labels(view_type, defect_type)

        self.status_bar.showMessage(f"Applied: {view_type} - {defect_type}")

    def enable_drawing(self):
        """Enable drawing mode."""
        self.canvas.enable_drawing(True)
        self.status_bar.showMessage("Drawing mode enabled - click and drag to draw bbox")

    def delete_selected_annotation(self):
        """Delete selected annotation."""
        self.canvas.delete_selected_bbox()

    def next_image(self):
        """Load next image in the filtered list."""
        current_row = self.image_list.currentRow()
        if current_row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(current_row + 1)

    def previous_image(self):
        """Load previous image in the filtered list."""
        current_row = self.image_list.currentRow()
        if current_row > 0:
            self.image_list.setCurrentRow(current_row - 1)

    def get_selected_view_type(self) -> str:
        """Get currently selected view type."""
        for view_type, button in self.view_buttons.items():
            if button.isChecked():
                return view_type
        return self.config.get('annotation.default_view', 'TOP')

    def get_selected_defect_type(self) -> str:
        """Get currently selected defect type."""
        for defect_type, button in self.defect_buttons.items():
            if button.isChecked():
                return defect_type
        return self.config.get('annotation.default_defect', 'PASS')

    def set_view_type(self, view_type: str):
        """Set view type selection."""
        if view_type in self.view_buttons:
            self.view_buttons[view_type].setChecked(True)

    def set_defect_type(self, defect_type: str):
        """Set defect type selection."""
        if defect_type in self.defect_buttons:
            self.defect_buttons[defect_type].setChecked(True)

    def update_progress_label(self):
        """Update progress label."""
        total = len(self.image_files)
        visible = self.image_list.count()

        # Count annotated images (in all files, not just visible)
        annotated = 0
        for filepath in self.image_files:
            # Use filepath instead of filename to avoid conflicts
            img_record = self.db.get_image_by_filepath(filepath)
            if img_record:
                count = self.db.get_annotation_count(img_record['id'])
                if count > 0:
                    annotated += 1

        # Show different text based on whether filter is active
        filter_active = hasattr(self, 'filter_combo') and self.filter_combo.currentText() != 'All'
        if filter_active and visible < total:
            self.progress_label.setText(f"{annotated} / {total} annotated | Showing {visible} filtered")
        else:
            self.progress_label.setText(f"{annotated} / {total} images annotated")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About",
            f"{self.config.get('application.name')}\n"
            f"Version {self.config.get('application.version')}\n\n"
            "Wire Loop X-ray annotation tool for semiconductor packaging."
        )

    def change_annotator(self):
        """Change annotator identity."""
        dialog = AnnotatorDialog(current_annotator=self.annotator, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_annotator = dialog.get_annotator()
            if new_annotator:
                self.annotator = new_annotator
                self.annotator_label.setText(f"Annotator: {self.annotator}")
                # Save to config for persistence
                self.config.set('annotation.default_annotator', new_annotator)
                self.config.save()
                self.status_bar.showMessage(f"Annotator changed to: {new_annotator}", 3000)

    def open_training_dialog(self, model_type: str):
        """Open training dialog for specific model type."""
        try:
            # Create and show training dialog
            self.training_dialog = TrainingDialog(
                db_path=str(self.db.db_path),
                config_dir="configs",
                data_dir="data",
                parent=self
            )
            # Set model type
            model_types = {'detection': 0, 'view': 1, 'defect': 2}
            if model_type in model_types:
                self.training_dialog.model_type_combo.setCurrentIndex(model_types[model_type])

            self.training_dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open training dialog: {e}")

    def train_all_models(self):
        """Train all three models sequentially with wire pairing preserved."""
        # Ask user for confirmation
        reply = QMessageBox.question(
            self,
            "Train All Models",
            "This will train all three models sequentially:\n\n"
            "1. YOLO Detection (wire detection)\n"
            "2. View Classifier (TOP/SIDE)\n"
            "3. Defect Classifier (PASS/defects)\n\n"
            "Wire pairing will be preserved for all models.\n\n"
            "This follows the inference pipeline logic:\n"
            "Detect → Classify View → Classify Defect\n\n"
            "This may take a considerable amount of time.\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Model types to train in order (logical sequence)
        # 1. YOLO first to ensure wire detection works
        # 2. View classifier on detected regions
        # 3. Defect classifier on detected regions
        model_types = [
            ('detection', 'YOLO Detection', 0),
            ('view', 'View Classifier', 1),
            ('defect', 'Defect Classifier', 2)
        ]

        failed_models = []

        for model_type, model_name, combo_index in model_types:
            try:
                self.status_bar.showMessage(f"Starting {model_name} training...")

                # Create training dialog
                training_dialog = TrainingDialog(
                    db_path=str(self.db.db_path),
                    config_dir="configs",
                    data_dir="data",
                    parent=self
                )

                # Set model type
                training_dialog.model_type_combo.setCurrentIndex(combo_index)

                # Enable wire pairing by default
                if hasattr(training_dialog, 'preserve_wire_pairs_checkbox'):
                    training_dialog.preserve_wire_pairs_checkbox.setChecked(True)

                # Execute dialog
                result = training_dialog.exec()

                if result == QDialog.DialogCode.Accepted:
                    self.status_bar.showMessage(f"{model_name} training completed successfully")
                else:
                    # User cancelled or dialog was closed
                    reply = QMessageBox.question(
                        self,
                        "Training Cancelled",
                        f"{model_name} training was cancelled.\n\n"
                        f"Do you want to continue with remaining models?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        break
                    failed_models.append(model_name)

            except Exception as e:
                # Error during training
                reply = QMessageBox.critical(
                    self,
                    "Training Error",
                    f"Failed to train {model_name}:\n{e}\n\n"
                    f"Do you want to continue with remaining models?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    break
                failed_models.append(model_name)

        # Show completion summary
        if failed_models:
            QMessageBox.warning(
                self,
                "Training Completed with Errors",
                f"Training completed with some failures:\n\n"
                f"Failed models: {', '.join(failed_models)}\n\n"
                f"Please check the logs and train failed models individually."
            )
        else:
            QMessageBox.information(
                self,
                "Training Complete",
                "All three models have been trained successfully!\n\n"
                "Wire pairing was preserved throughout all training phases.\n\n"
                "You can now use the trained models for inference."
            )

    def open_tensorboard(self):
        """Open TensorBoard in browser."""
        try:
            if not self.tensorboard_manager.is_running():
                success = self.tensorboard_manager.start_tensorboard("runs", auto_open=True)
                if success:
                    self.statusBar().showMessage(
                        f"TensorBoard started on {self.tensorboard_manager.get_tensorboard_url()}",
                        5000
                    )
                else:
                    QMessageBox.warning(self, "TensorBoard", "Failed to start TensorBoard")
            else:
                # Already running, just open browser
                import webbrowser
                webbrowser.open(self.tensorboard_manager.get_tensorboard_url())
                self.statusBar().showMessage(
                    f"TensorBoard: {self.tensorboard_manager.get_tensorboard_url()}",
                    3000
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open TensorBoard: {e}")

    def open_model_selector(self):
        """Open model selector dialog."""
        try:
            dialog = ModelSelectorDialog(self.db, models_dir='models', parent=self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open model selector: {e}")

    def _open_model_comparison(self):
        """Open model comparison dialog."""
        from PyQt6.QtWidgets import QInputDialog

        # Ask user which model type to compare
        model_types = ['detection', 'view', 'defect']
        model_type, ok = QInputDialog.getItem(
            self,
            "Select Model Type",
            "Which model type do you want to compare?",
            model_types,
            0,
            False
        )

        if ok and model_type:
            try:
                dialog = ModelComparisonDialog(model_type, self.db, self)
                dialog.exec()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Comparison Error",
                    f"Failed to open model comparison:\n{e}"
                )

    def infer_current_image(self):
        """Run inference on current image and show prediction for review."""
        if self.current_image_index < 0:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        # Check if image already has annotation
        image_path = self.image_files[self.current_image_index]
        image_name = Path(image_path).name

        # Get image record and check for existing annotations
        image_record = self.db.get_image_by_filepath(image_path)
        if image_record:
            existing_annotations = self.db.get_annotations(image_record['id'])
            if existing_annotations:
                existing_annotation = existing_annotations[0]  # Use first annotation
                reply = QMessageBox.question(
                    self,
                    "Existing Annotation",
                    f"This image already has an annotation.\n\n"
                    f"Current: View={existing_annotation['view_type']}, "
                    f"Defect={existing_annotation['defect_type']}\n\n"
                    f"Do you want to overwrite with model prediction?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

        # Load models if not already loaded
        if self.model_loader is None or self.inference_pipeline is None:
            try:
                self._load_inference_models()
            except Exception as e:
                QMessageBox.critical(self, "Model Load Error", f"Failed to load models:\n{e}")
                return

        # Run inference
        try:
            result = self.inference_pipeline.infer_single_image(image_path)

            if not result['success']:
                QMessageBox.warning(self, "Inference Failed", f"Failed to run inference:\n{result.get('error', 'Unknown error')}")
                return

            primary = result.get('primary_result')
            if not primary:
                QMessageBox.warning(self, "No Detection", "No wire loop detected in the image.")
                return

            # Store prediction for potential modification
            self.pending_prediction = {
                'image_path': image_path,
                'bbox': primary['bbox'],
                'view_type': primary['view'],
                'view_confidence': primary['view_confidence'],
                'defect_type': primary['defect'],
                'defect_confidence': primary['defect_confidence'],
                'model_versions': {
                    'detection': 'active',
                    'view': 'active',
                    'defect': 'active'
                }
            }

            # Show prediction dialog
            self._show_prediction_dialog(result)

        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"Failed to run inference:\n{e}")

    def _load_inference_models(self):
        """Load VIEW-aware inference models from active model versions."""
        try:
            # Get active model paths from database
            active_models = self.db.get_active_models()

            if not active_models:
                raise ValueError("No active models found. Please train models or select active models first.")

            # Check if we have VIEW-aware models (new architecture)
            has_view_aware = (
                active_models.get('detection_top') and
                active_models.get('detection_side') and
                active_models.get('defect_top') and
                active_models.get('defect_side')
            )

            # Create ModelLoader (supports both old and new architecture)
            if has_view_aware:
                logger.info("Loading VIEW-aware models (new architecture)")
                self.model_loader = ModelLoader(
                    view_classifier_path=active_models.get('view'),
                    yolo_top_path=active_models.get('detection_top'),
                    yolo_side_path=active_models.get('detection_side'),
                    defect_top_path=active_models.get('defect_top'),
                    defect_side_path=active_models.get('defect_side')
                )
            else:
                # Fallback to old architecture (backward compatibility)
                logger.warning("Using unified models (old architecture) - Consider training VIEW-specific models for better accuracy")
                self.model_loader = ModelLoader(
                    view_classifier_path=active_models.get('view'),
                    yolo_path=active_models.get('detection'),
                    defect_classifier_path=active_models.get('defect')
                )

            # Create InferencePipeline
            self.inference_pipeline = InferencePipeline(self.model_loader)

            arch_type = "VIEW-aware" if has_view_aware else "unified"
            self.statusBar().showMessage(f"Models loaded successfully ({arch_type} architecture)", 3000)

        except Exception as e:
            raise RuntimeError(f"Failed to load inference models: {e}")

    def _show_prediction_dialog(self, result: dict):
        """
        Show prediction dialog with Accept/Modify/Cancel options.

        Args:
            result: Inference result dictionary
        """
        from PyQt6.QtWidgets import QDialog, QTextEdit

        dialog = QDialog(self)
        dialog.setWindowTitle("Model Prediction")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Prediction info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)

        primary = result['primary_result']
        bbox = primary['bbox']
        bbox_conf = primary['bbox_confidence']

        pred_info = f"""<h3>Model Prediction Results</h3>
<b>Detection:</b><br/>
&nbsp;&nbsp;Bounding Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]<br/>
&nbsp;&nbsp;Confidence: {bbox_conf:.2%}<br/>
<br/>
<b>View Classification:</b><br/>
&nbsp;&nbsp;Predicted: <b>{primary['view']}</b><br/>
&nbsp;&nbsp;Confidence: {primary['view_confidence']:.2%}<br/>
<br/>
<b>Defect Classification:</b><br/>
&nbsp;&nbsp;Predicted: <b>{primary['defect']}</b><br/>
&nbsp;&nbsp;Confidence: {primary['defect_confidence']:.2%}<br/>
<br/>
<b>Processing Time:</b> {result['processing_time']:.3f}s<br/>
<br/>
<i>You can:</i><br/>
• Click <b>Accept</b> to save this prediction as annotation<br/>
• Click <b>Modify</b> to adjust the prediction before saving<br/>
• Click <b>Cancel</b> to discard this prediction
"""
        info_text.setHtml(pred_info)
        layout.addWidget(info_text)

        # Buttons
        button_layout = QHBoxLayout()

        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(lambda: self._accept_prediction(dialog))

        modify_button = QPushButton("Modify")
        modify_button.clicked.connect(lambda: self._modify_prediction(dialog))

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)

        button_layout.addWidget(accept_button)
        button_layout.addWidget(modify_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        dialog.exec()

    def _accept_prediction(self, dialog):
        """Accept the prediction and save to database."""
        if self.pending_prediction is None:
            return

        try:
            # Save prediction to database
            image_name = Path(self.pending_prediction['image_path']).name

            # Get or create image_id
            if self.current_image_id is None:
                self.current_image_id = self.db.add_image(image_name, self.current_folder)

            # Save annotation
            self.db.add_annotation(
                image_id=self.current_image_id,
                bbox=self.pending_prediction['bbox'],
                view_type=self.pending_prediction['view_type'],
                defect_type=self.pending_prediction['defect_type'],
                annotator=self.annotator,
                model_predicted=True,
                model_version=f"det:{self.pending_prediction['model_versions']['detection']},"
                             f"view:{self.pending_prediction['model_versions']['view']},"
                             f"def:{self.pending_prediction['model_versions']['defect']}"
            )

            # Update UI
            self.canvas.set_bbox(self.pending_prediction['bbox'])
            self.view_buttons[self.pending_prediction['view_type']].setChecked(True)
            self.defect_buttons[self.pending_prediction['defect_type']].setChecked(True)

            self.statusBar().showMessage(f"Prediction accepted and saved for {image_name}", 3000)

            # Clear pending prediction
            self.pending_prediction = None
            dialog.accept()

            # Move to next image
            self.next_image()

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save prediction:\n{e}")

    def _modify_prediction(self, dialog):
        """Close dialog and let user modify the prediction manually."""
        if self.pending_prediction is None:
            return

        try:
            # Apply prediction to UI for manual modification
            self.canvas.set_bbox(self.pending_prediction['bbox'])
            self.view_buttons[self.pending_prediction['view_type']].setChecked(True)
            self.defect_buttons[self.pending_prediction['defect_type']].setChecked(True)

            dialog.accept()

            QMessageBox.information(
                self,
                "Modify Prediction",
                "Prediction loaded into the annotation panel.\n\n"
                "You can now:\n"
                "• Adjust the bounding box on the image\n"
                "• Change the view type or defect type\n"
                "• Press 'S' to save when done"
            )

            # Don't clear pending_prediction yet - keep for reference
            # User will save manually with modified values

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply prediction:\n{e}")

    def infer_batch(self):
        """Run batch inference on folder."""
        from PyQt6.QtWidgets import QProgressDialog
        from src.inference.batch_processor import BatchProcessor
        from src.inference.result_exporter import ResultExporter
        from src.gui.batch_results_dialog import BatchResultsDialog

        # Select folder
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for Batch Inference")
        if not folder:
            return

        # Load models if needed
        if self.model_loader is None or self.inference_pipeline is None:
            try:
                self._load_inference_models()
            except Exception as e:
                QMessageBox.critical(self, "Model Load Error", f"Failed to load models:\n{e}")
                return

        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if Path(f).suffix.lower() in image_extensions
        ]

        if not image_files:
            QMessageBox.warning(self, "No Images", "No image files found in selected folder.")
            return

        # Create progress dialog
        progress = QProgressDialog("Processing images...", "Cancel", 0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        # Create batch processor
        batch_processor = BatchProcessor(self.inference_pipeline)

        # Process images
        try:
            results = []
            for i, image_path in enumerate(image_files):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Processing {Path(image_path).name}...")

                try:
                    result = self.inference_pipeline.infer_single_image(image_path)
                    # Add full image path for preview
                    result['image_path'] = image_path
                    results.append(result)
                except Exception as e:
                    results.append({
                        'image_name': Path(image_path).name,
                        'image_path': image_path,
                        'success': False,
                        'error': str(e),
                        'detections': [],
                        'primary_result': None,
                        'processing_time': 0.0
                    })

            progress.setValue(len(image_files))

            if not results:
                QMessageBox.warning(self, "No Results", "No results to display.")
                return

            # Export results to CSV
            output_path = os.path.join(folder, 'inference_results.csv')
            ResultExporter.export_csv(results, output_path)

            # Show results dialog
            results_dialog = BatchResultsDialog(results, self)
            results_dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Batch Inference Error", f"Failed to process images:\n{e}")

    def closeEvent(self, event):
        """Handle window close event - cleanup TensorBoard."""
        # Save current annotations
        self.save_current_image_annotations()

        # Stop TensorBoard if running
        if self.tensorboard_manager.is_running():
            self.tensorboard_manager.stop_tensorboard()

        # Close database
        self.db.close()

        event.accept()
