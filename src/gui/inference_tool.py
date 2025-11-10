"""
GUI tool for inference on Wire Loop X-ray images.

Provides a graphical interface similar to the annotation tool for running inference
and visualizing results.
"""

import logging
from pathlib import Path
from typing import Optional, List
import sys

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QProgressBar, QGroupBox,
    QFileDialog, QMessageBox, QComboBox, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
import numpy as np
from PIL import Image

from src.inference import ModelLoader, InferencePipeline, BatchProcessor, ResultExporter

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Background worker for running single-image inference."""

    progress = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)  # inference result
    finished = pyqtSignal(list)  # all results
    error = pyqtSignal(str)  # error message

    def __init__(self, pipeline: InferencePipeline, image_paths: List[str]):
        super().__init__()
        self.pipeline = pipeline
        self.image_paths = image_paths
        self.results = []

    def run(self):
        """Run inference on all images."""
        try:
            for i, image_path in enumerate(self.image_paths):
                result = self.pipeline.infer_single_image(image_path)
                self.results.append(result)
                self.result_ready.emit(result)
                self.progress.emit(i + 1, len(self.image_paths))

            self.finished.emit(self.results)
        except Exception as e:
            logger.error(f"Inference worker error: {e}")
            self.error.emit(str(e))


class PairedInferenceWorker(QThread):
    """Background worker for running paired TOP/SIDE inference."""

    progress = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)  # paired inference result
    finished = pyqtSignal(list)  # all results
    error = pyqtSignal(str)  # error message

    def __init__(self, pipeline: InferencePipeline, image_dir: str, combination_strategy: str = "worst_case"):
        super().__init__()
        self.pipeline = pipeline
        self.image_dir = image_dir
        self.combination_strategy = combination_strategy
        self.results = []

    def run(self):
        """Run paired inference on all wire pairs in directory."""
        try:
            # Use batch pairing method from pipeline
            results = self.pipeline.infer_batch_with_pairing(
                self.image_dir,
                self.combination_strategy
            )

            # Emit progress and results
            for i, result in enumerate(results):
                self.results.append(result)
                self.result_ready.emit(result)
                self.progress.emit(i + 1, len(results))

            self.finished.emit(self.results)
        except Exception as e:
            logger.error(f"Paired inference worker error: {e}")
            self.error.emit(str(e))


class InferenceCanvas(QLabel):
    """Canvas for displaying images with inference results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(800, 800)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")

        self.original_image = None
        self.display_pixmap = None
        self.inference_result = None

    def load_image(self, image_path: str):
        """Load and display image."""
        try:
            self.original_image = Image.open(image_path)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')

            self.update_display()

        except Exception as e:
            logger.error(f"Failed to load image: {e}")

    def set_inference_result(self, result: dict):
        """Set inference result and update display."""
        self.inference_result = result
        self.update_display()

    def update_display(self):
        """Update the displayed image with inference results."""
        if self.original_image is None:
            return

        # Convert PIL image to QImage
        img_array = np.array(self.original_image)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Create pixmap
        pixmap = QPixmap.fromImage(q_image)

        # Draw inference results if available
        if self.inference_result and self.inference_result.get('success'):
            painter = QPainter(pixmap)

            primary_result = self.inference_result.get('primary_result')
            if primary_result and primary_result.get('bbox'):
                bbox = primary_result['bbox']

                # Draw bounding box
                pen = QPen(QColor(0, 255, 0), 3)
                painter.setPen(pen)
                x1, y1, x2, y2 = bbox
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                # Draw label
                view = primary_result.get('view', 'Unknown')
                defect = primary_result.get('defect', 'Unknown')
                view_conf = primary_result.get('view_confidence', 0)
                defect_conf = primary_result.get('defect_confidence', 0)

                label_text = f"View: {view} ({view_conf:.2f})\nDefect: {defect} ({defect_conf:.2f})"

                # Draw text background
                font = QFont("Arial", 14, QFont.Weight.Bold)
                painter.setFont(font)

                text_rect = painter.fontMetrics().boundingRect(0, 0, 500, 100,
                                                               Qt.TextFlag.TextWordWrap, label_text)
                text_rect.moveTopLeft(painter.viewport().topLeft())
                text_rect.adjust(-5, -5, 5, 5)

                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.TextFlag.TextWordWrap, label_text)

            painter.end()

        # Scale to fit widget
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.display_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)


class InferenceTool(QMainWindow):
    """Main GUI window for inference tool."""

    def __init__(self):
        super().__init__()

        self.image_dir = None
        self.image_files = []
        self.current_index = 0

        self.pipeline = None
        self.worker = None
        self.inference_results = []

        self.init_ui()
        self.setup_logging()

    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("Wire Loop X-ray Inference Tool")
        self.setGeometry(100, 100, 1600, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Image list and controls
        left_panel = self.create_left_panel()

        # Center panel - Canvas
        self.canvas = InferenceCanvas()

        # Right panel - Results
        right_panel = self.create_right_panel()

        # Add panels to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_left_panel(self) -> QWidget:
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Model loading group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()

        self.load_models_btn = QPushButton("Load Models")
        self.load_models_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.load_models_btn)

        self.model_status_label = QLabel("Models: Not Loaded")
        self.model_status_label.setWordWrap(True)
        model_layout.addWidget(self.model_status_label)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Image loading group
        image_group = QGroupBox("Images")
        image_layout = QVBoxLayout()

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)
        image_layout.addWidget(self.open_folder_btn)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        image_layout.addWidget(self.image_list)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Inference controls
        inference_group = QGroupBox("Inference")
        inference_layout = QVBoxLayout()

        # Inference mode selector
        mode_label = QLabel("Inference Mode:")
        inference_layout.addWidget(mode_label)

        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItem("Single Image", "single")
        self.inference_mode_combo.addItem("Paired TOP/SIDE (Recommended)", "paired")
        self.inference_mode_combo.setCurrentIndex(1)  # Default to paired mode
        self.inference_mode_combo.setToolTip(
            "Single Image: Process each image independently\n"
            "Paired TOP/SIDE: Combine TOP and SIDE views of same wire\n"
            "  - Requires filenames like: {wire_id}_TOP.jpg, {wire_id}_SIDE.jpg\n"
            "  - Provides more accurate defect detection\n"
            "  - Recommended for production use"
        )
        inference_layout.addWidget(self.inference_mode_combo)

        # Combination strategy selector (only for paired mode)
        strategy_label = QLabel("Combination Strategy:")
        inference_layout.addWidget(strategy_label)

        self.combination_strategy_combo = QComboBox()
        self.combination_strategy_combo.addItem("Worst Case (Fail if either defect)", "worst_case")
        self.combination_strategy_combo.addItem("Highest Confidence", "confidence")
        self.combination_strategy_combo.setCurrentIndex(0)  # Default to worst_case
        self.combination_strategy_combo.setToolTip(
            "Worst Case: Select more severe defect (PASS < 沖線 < 晃動 < 碰觸)\n"
            "  - Recommended for quality control\n"
            "Highest Confidence: Use prediction with higher confidence\n"
            "  - May miss defects visible in one view"
        )
        inference_layout.addWidget(self.combination_strategy_combo)

        # Update strategy combo visibility based on mode
        def update_strategy_visibility():
            is_paired = self.inference_mode_combo.currentData() == "paired"
            strategy_label.setVisible(is_paired)
            self.combination_strategy_combo.setVisible(is_paired)

        self.inference_mode_combo.currentIndexChanged.connect(update_strategy_visibility)
        update_strategy_visibility()

        self.infer_current_btn = QPushButton("Infer Current Image")
        self.infer_current_btn.clicked.connect(self.infer_current_image)
        self.infer_current_btn.setEnabled(False)
        inference_layout.addWidget(self.infer_current_btn)

        self.infer_all_btn = QPushButton("Infer All Images")
        self.infer_all_btn.clicked.connect(self.infer_all_images)
        self.infer_all_btn.setEnabled(False)
        inference_layout.addWidget(self.infer_all_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        inference_layout.addWidget(self.progress_bar)

        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)

        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(self.export_json)
        self.export_json_btn.setEnabled(False)
        export_layout.addWidget(self.export_json_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()

        return panel

    def create_right_panel(self) -> QWidget:
        """Create right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Results display
        results_group = QGroupBox("Inference Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        return panel

    def setup_logging(self):
        """Setup logging display in status bar."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def find_latest_model(self, model_dir: str, extension: str) -> Optional[str]:
        """
        Find the latest model file in a directory.

        Args:
            model_dir: Directory to search
            extension: File extension (e.g., '.pt', '.pth')

        Returns:
            Path to latest model file or None
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            return None

        # Find all model files with the extension
        model_files = list(model_path.glob(f'*{extension}'))

        if not model_files:
            return None

        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return str(model_files[0])

    def load_models(self):
        """Load inference models."""
        try:
            self.statusBar().showMessage("Loading models...")

            # Find latest model files
            yolo_path = self.find_latest_model("models/detection", ".pt")
            view_path = self.find_latest_model("models/view", ".pt")
            defect_path = self.find_latest_model("models/defect", ".pt")

            # Check if models exist
            if not yolo_path or not view_path or not defect_path:
                missing = []
                if not yolo_path:
                    missing.append("Detection model (models/detection/)")
                if not view_path:
                    missing.append("View classifier (models/view/)")
                if not defect_path:
                    missing.append("Defect classifier (models/defect/)")

                QMessageBox.warning(
                    self,
                    "Models Not Found",
                    f"Trained models not found:\n" + "\n".join(f"- {m}" for m in missing) +
                    "\n\nPlease train models first using train.bat"
                )
                return

            # Load models
            model_loader = ModelLoader(
                yolo_path=yolo_path,
                view_classifier_path=view_path,
                defect_classifier_path=defect_path,
                device="auto"
            )

            self.pipeline = InferencePipeline(model_loader)

            # Show loaded model info
            status_text = "Models: Loaded ✓\n"
            status_text += f"Device: {self.pipeline.device}\n\n"
            status_text += f"Detection:\n  {Path(yolo_path).name}\n"
            status_text += f"View:\n  {Path(view_path).name}\n"
            status_text += f"Defect:\n  {Path(defect_path).name}"

            self.model_status_label.setText(status_text)
            self.infer_current_btn.setEnabled(True)
            self.infer_all_btn.setEnabled(True)

            self.statusBar().showMessage("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load models:\n{str(e)}")
            self.statusBar().showMessage("Failed to load models")

    def open_folder(self):
        """Open folder containing images."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")

        if folder:
            self.image_dir = Path(folder)
            self.load_images()

    def load_images(self):
        """Load images from selected folder."""
        if not self.image_dir:
            return

        # Find image files
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        self.image_files = []

        for ext in supported_formats:
            self.image_files.extend(self.image_dir.glob(f'*{ext}'))
            self.image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))

        self.image_files.sort()

        # Update list widget
        self.image_list.clear()
        for img_file in self.image_files:
            item = QListWidgetItem(img_file.name)
            self.image_list.addItem(item)

        self.statusBar().showMessage(f"Loaded {len(self.image_files)} images")

        # Select first image
        if self.image_files:
            self.image_list.setCurrentRow(0)

    def on_image_selected(self, index: int):
        """Handle image selection."""
        if 0 <= index < len(self.image_files):
            self.current_index = index
            image_path = str(self.image_files[index])
            self.canvas.load_image(image_path)

            # Show cached result if available
            if index < len(self.inference_results):
                result = self.inference_results[index]
                self.canvas.set_inference_result(result)
                self.display_result(result)
            else:
                self.results_text.clear()

    def infer_current_image(self):
        """Run inference on current image."""
        if not self.pipeline:
            QMessageBox.warning(self, "Warning", "Please load models first")
            return

        if not self.image_files:
            QMessageBox.warning(self, "Warning", "Please open a folder with images first")
            return

        try:
            self.statusBar().showMessage("Running inference...")

            image_path = str(self.image_files[self.current_index])
            result = self.pipeline.infer_single_image(image_path)

            # Update results
            if self.current_index < len(self.inference_results):
                self.inference_results[self.current_index] = result
            else:
                while len(self.inference_results) < self.current_index:
                    self.inference_results.append(None)
                self.inference_results.append(result)

            # Display result
            self.canvas.set_inference_result(result)
            self.display_result(result)

            self.export_csv_btn.setEnabled(True)
            self.export_json_btn.setEnabled(True)

            self.statusBar().showMessage("Inference completed")

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            QMessageBox.critical(self, "Error", f"Inference failed:\n{str(e)}")

    def infer_all_images(self):
        """Run inference on all images (single or paired mode)."""
        if not self.pipeline:
            QMessageBox.warning(self, "Warning", "Please load models first")
            return

        if not self.image_files:
            QMessageBox.warning(self, "Warning", "Please open a folder with images first")
            return

        # Disable buttons during inference
        self.infer_current_btn.setEnabled(False)
        self.infer_all_btn.setEnabled(False)

        # Get inference mode
        inference_mode = self.inference_mode_combo.currentData()

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        if inference_mode == "paired":
            # Paired inference mode
            combination_strategy = self.combination_strategy_combo.currentData()

            # Estimate number of wire pairs (total images / 2)
            estimated_pairs = len(self.image_files) // 2
            self.progress_bar.setMaximum(estimated_pairs)

            # Create paired inference worker
            self.worker = PairedInferenceWorker(
                self.pipeline,
                str(self.image_dir),
                combination_strategy
            )

            self.statusBar().showMessage(f"Running paired inference with {combination_strategy} strategy...")
        else:
            # Single image mode
            self.progress_bar.setMaximum(len(self.image_files))

            # Create single inference worker
            image_paths = [str(p) for p in self.image_files]
            self.worker = InferenceWorker(self.pipeline, image_paths)

            self.statusBar().showMessage("Running single-image inference...")

        # Connect signals
        self.worker.progress.connect(self.on_inference_progress)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"Processing image {current}/{total}")

    def on_result_ready(self, result: dict):
        """Handle single result."""
        pass  # Results are collected in worker

    def on_inference_finished(self, results: List[dict]):
        """Handle inference completion."""
        self.inference_results = results

        # Re-enable buttons
        self.infer_current_btn.setEnabled(True)
        self.infer_all_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)

        # Hide progress bar
        self.progress_bar.setVisible(False)

        # Update statistics
        self.update_statistics()

        # Show current image result
        if self.current_index < len(results):
            result = results[self.current_index]
            self.canvas.set_inference_result(result)
            self.display_result(result)

        self.statusBar().showMessage(f"Inference completed: {len(results)} images processed")

        QMessageBox.information(
            self,
            "Inference Complete",
            f"Successfully processed {len(results)} images"
        )

    def on_inference_error(self, error_msg: str):
        """Handle inference error."""
        self.infer_current_btn.setEnabled(True)
        self.infer_all_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(self, "Inference Error", error_msg)
        self.statusBar().showMessage("Inference failed")

    def display_result(self, result: dict):
        """Display inference result in text widget (single or paired)."""
        if not result:
            return

        # Check if this is a paired result (has wire_id, top_result, side_result)
        if 'wire_id' in result and 'top_result' in result and 'side_result' in result:
            # Paired result display
            text = f"=== PAIRED INFERENCE RESULT ===\n\n"
            text += f"Wire ID: {result['wire_id']}\n"
            text += f"Processing Time: {result['processing_time']:.3f}s\n\n"

            if result['success']:
                combined = result.get('combined_result')

                # Show combined decision
                text += f">>> FINAL DECISION <<<\n"
                text += f"Defect: {combined['defect_type']}\n"
                text += f"Confidence: {combined['defect_confidence']:.4f}\n"
                text += f"Decisive View: {combined['decisive_view']}\n\n"

                # Show TOP view result
                text += "--- TOP View ---\n"
                top_result = result['top_result']
                if top_result['success'] and top_result['primary_result']:
                    top_primary = top_result['primary_result']
                    text += f"View: {top_primary['view']}\n"
                    text += f"  Confidence: {top_primary['view_confidence']:.4f}\n"
                    text += f"Defect: {top_primary['defect']}\n"
                    text += f"  Confidence: {top_primary['defect_confidence']:.4f}\n\n"
                else:
                    text += "No detection or error\n\n"

                # Show SIDE view result
                text += "--- SIDE View ---\n"
                side_result = result['side_result']
                if side_result['success'] and side_result['primary_result']:
                    side_primary = side_result['primary_result']
                    text += f"View: {side_primary['view']}\n"
                    text += f"  Confidence: {side_primary['view_confidence']:.4f}\n"
                    text += f"Defect: {side_primary['defect']}\n"
                    text += f"  Confidence: {side_primary['defect_confidence']:.4f}\n"
                else:
                    text += "No detection or error\n"
            else:
                text += f"Error: {result.get('error', 'Unknown error')}\n"

        else:
            # Single image result display
            text = f"Image: {result.get('image_name', 'Unknown')}\n"
            text += f"Processing Time: {result['processing_time']:.3f}s\n\n"

            if result['success']:
                primary = result.get('primary_result')
                if primary and primary.get('bbox'):
                    bbox = primary['bbox']
                    text += f"Bounding Box: ({bbox[0]}, {bbox[1]}) -> ({bbox[2]}, {bbox[3]})\n"
                    text += f"  Confidence: {primary['bbox_confidence']:.4f}\n\n"
                    text += f"View: {primary['view']}\n"
                    text += f"  Confidence: {primary['view_confidence']:.4f}\n\n"
                    text += f"Defect: {primary['defect']}\n"
                    text += f"  Confidence: {primary['defect_confidence']:.4f}\n"
                else:
                    text += "No detection found\n"
            else:
                text += f"Error: {result.get('error', 'Unknown error')}\n"

        self.results_text.setPlainText(text)

    def update_statistics(self):
        """Update statistics display (single or paired results)."""
        if not self.inference_results:
            return

        # Check if results are paired or single
        is_paired = (self.inference_results and
                    'wire_id' in self.inference_results[0] and
                    'combined_result' in self.inference_results[0])

        if is_paired:
            # Statistics for paired results
            total = len(self.inference_results)
            successful = sum(1 for r in self.inference_results if r['success'])

            # Count by combined defect type
            defect_counts = {}
            decisive_view_counts = {}

            for result in self.inference_results:
                if result['success'] and result['combined_result']:
                    combined = result['combined_result']
                    defect = combined['defect_type']
                    decisive_view = combined.get('decisive_view')

                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
                    if decisive_view:
                        decisive_view_counts[decisive_view] = decisive_view_counts.get(decisive_view, 0) + 1

            # Build statistics text
            text = f"=== PAIRED INFERENCE STATISTICS ===\n\n"
            text += f"Total Wire Pairs: {total}\n"
            text += f"Successful: {successful}\n"
            text += f"Failed: {total - successful}\n\n"

            if defect_counts:
                text += "Combined Defect Distribution:\n"
                for defect, count in sorted(defect_counts.items()):
                    pct = count / successful * 100 if successful > 0 else 0
                    text += f"  {defect}: {count} ({pct:.1f}%)\n"
                text += "\n"

            if decisive_view_counts:
                text += "Decisive View Distribution:\n"
                for view, count in sorted(decisive_view_counts.items()):
                    pct = count / successful * 100 if successful > 0 else 0
                    text += f"  {view}: {count} ({pct:.1f}%)\n"

        else:
            # Statistics for single image results
            total = len(self.inference_results)
            successful = sum(1 for r in self.inference_results if r['success'])
            detected = sum(1 for r in self.inference_results
                          if r['success'] and r.get('primary_result') and r['primary_result'].get('bbox'))

            # Count by view and defect
            view_counts = {}
            defect_counts = {}

            for result in self.inference_results:
                if result['success'] and result.get('primary_result') and result['primary_result'].get('bbox'):
                    view = result['primary_result']['view']
                    defect = result['primary_result']['defect']
                    view_counts[view] = view_counts.get(view, 0) + 1
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1

            # Build statistics text
            text = f"Total Images: {total}\n"
            text += f"Successful: {successful}\n"
            text += f"Detected: {detected}\n"
            text += f"No Detection: {successful - detected}\n\n"

            if view_counts:
                text += "View Distribution:\n"
                for view, count in sorted(view_counts.items()):
                    pct = count / detected * 100 if detected > 0 else 0
                    text += f"  {view}: {count} ({pct:.1f}%)\n"
                text += "\n"

            if defect_counts:
                text += "Defect Distribution:\n"
                for defect, count in sorted(defect_counts.items()):
                    pct = count / detected * 100 if detected > 0 else 0
                    text += f"  {defect}: {count} ({pct:.1f}%)\n"

        self.stats_text.setPlainText(text)

    def export_csv(self):
        """Export results to CSV."""
        if not self.inference_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "inference_results.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                ResultExporter.export_csv(self.inference_results, file_path)
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")

    def export_json(self):
        """Export results to JSON."""
        if not self.inference_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON",
            "inference_results.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                ResultExporter.export_json(self.inference_results, file_path)
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export JSON:\n{str(e)}")


def main():
    """Main entry point."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = InferenceTool()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
