"""
Batch inference results viewer dialog.

This dialog displays batch inference results in a table with image preview
and detailed information for each result.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QComboBox, QSplitter, QWidget,
    QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
import cv2
import numpy as np


class BatchResultsDialog(QDialog):
    """
    Dialog for viewing batch inference results.

    Features:
    - Table view of all results
    - Image preview with bounding box
    - Detailed result information
    - Filter by status/defect type
    - Export filtered results
    """

    def __init__(self, results: List[Dict], parent=None):
        """
        Initialize batch results dialog.

        Args:
            results: List of inference results
            parent: Parent widget
        """
        super().__init__(parent)
        self.results = results
        self.filtered_results = results.copy()
        self.current_index = -1

        self._init_ui()
        self._populate_table()

    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("Batch Inference Results")
        self.setMinimumSize(1200, 800)

        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>Batch Inference Results</h2>")
        layout.addWidget(title)

        # Summary
        summary_layout = self._create_summary()
        layout.addLayout(summary_layout)

        # Filter section
        filter_layout = self._create_filter_section()
        layout.addLayout(filter_layout)

        # Main content: table + preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Results table
        table_widget = self._create_table_widget()
        splitter.addWidget(table_widget)

        # Right: Image preview and details
        preview_widget = self._create_preview_widget()
        splitter.addWidget(preview_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Action buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_summary(self) -> QHBoxLayout:
        """Create summary section."""
        layout = QHBoxLayout()

        total = len(self.results)
        success = sum(1 for r in self.results if r['success'])
        failed = total - success

        # Count defect types
        defect_counts = {}
        for result in self.results:
            if result['success'] and result.get('primary_result'):
                defect = result['primary_result']['defect']
                defect_counts[defect] = defect_counts.get(defect, 0) + 1

        summary_text = f"<b>Total:</b> {total} | <b>Success:</b> {success} | <b>Failed:</b> {failed}"
        if defect_counts:
            defect_str = " | ".join([f"<b>{k}:</b> {v}" for k, v in defect_counts.items()])
            summary_text += f" | {defect_str}"

        summary_label = QLabel(summary_text)
        layout.addWidget(summary_label)
        layout.addStretch()

        return layout

    def _create_filter_section(self) -> QHBoxLayout:
        """Create filter section."""
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Filter:"))

        # Status filter
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Success Only", "Failed Only"])
        self.status_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.status_filter)

        # Defect type filter
        self.defect_filter = QComboBox()
        defect_types = set()
        for result in self.results:
            if result['success'] and result.get('primary_result'):
                defect_types.add(result['primary_result']['defect'])

        self.defect_filter.addItem("All Defects")
        self.defect_filter.addItems(sorted(defect_types))
        self.defect_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.defect_filter)

        layout.addStretch()

        return layout

    def _create_table_widget(self) -> QWidget:
        """Create table widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Image", "View", "View Conf", "Defect", "Defect Conf", "Status"
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self.table)
        widget.setLayout(layout)

        return widget

    def _create_preview_widget(self) -> QWidget:
        """Create preview widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Image preview
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setScaledContents(False)
        preview_layout.addWidget(self.image_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Details
        details_group = QGroupBox("Detection Details")
        details_layout = QVBoxLayout()

        self.details_label = QLabel("Select an image to view details")
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # Navigation
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self._show_previous)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self._show_next)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        widget.setLayout(layout)
        return widget

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        # Export filtered results
        export_button = QPushButton("Export Filtered Results")
        export_button.clicked.connect(self._export_filtered_results)
        layout.addWidget(export_button)

        layout.addStretch()

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        return layout

    def _populate_table(self):
        """Populate table with filtered results."""
        self.table.setRowCount(len(self.filtered_results))

        for row, result in enumerate(self.filtered_results):
            # Image name
            image_item = QTableWidgetItem(result['image_name'])
            self.table.setItem(row, 0, image_item)

            if result['success'] and result.get('primary_result'):
                primary = result['primary_result']

                # View
                view_item = QTableWidgetItem(primary['view'])
                view_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 1, view_item)

                # View confidence
                view_conf_item = QTableWidgetItem(f"{primary['view_confidence']:.2%}")
                view_conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 2, view_conf_item)

                # Defect
                defect_item = QTableWidgetItem(primary['defect'])
                defect_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color code by defect type
                if primary['defect'] == 'PASS':
                    defect_item.setBackground(QColor(144, 238, 144))  # Light green
                elif primary['defect'] == 'no_detection':
                    defect_item.setBackground(QColor(211, 211, 211))  # Light gray
                else:
                    defect_item.setBackground(QColor(255, 182, 193))  # Light red

                self.table.setItem(row, 3, defect_item)

                # Defect confidence
                defect_conf_item = QTableWidgetItem(f"{primary['defect_confidence']:.2%}")
                defect_conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 4, defect_conf_item)

                # Status
                status_item = QTableWidgetItem("✓ Success")
                status_item.setForeground(QColor(0, 128, 0))
            else:
                # Failed result
                for col in [1, 2, 3, 4]:
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, col, item)

                status_item = QTableWidgetItem("✗ Failed")
                status_item.setForeground(QColor(255, 0, 0))

            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, status_item)

    def _apply_filters(self):
        """Apply filters to results."""
        status_filter = self.status_filter.currentText()
        defect_filter = self.defect_filter.currentText()

        self.filtered_results = []

        for result in self.results:
            # Apply status filter
            if status_filter == "Success Only" and not result['success']:
                continue
            if status_filter == "Failed Only" and result['success']:
                continue

            # Apply defect filter
            if defect_filter != "All Defects":
                if not result['success'] or not result.get('primary_result'):
                    continue
                if result['primary_result']['defect'] != defect_filter:
                    continue

            self.filtered_results.append(result)

        # Repopulate table
        self._populate_table()

    def _on_selection_changed(self):
        """Handle selection change."""
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            return

        row = self.table.currentRow()
        if 0 <= row < len(self.filtered_results):
            self.current_index = row
            self._show_result(row)

            # Update navigation buttons
            self.prev_button.setEnabled(row > 0)
            self.next_button.setEnabled(row < len(self.filtered_results) - 1)

    def _show_result(self, index: int):
        """Show result at index."""
        if not (0 <= index < len(self.filtered_results)):
            return

        result = self.filtered_results[index]

        # Load and display image with bbox
        if result['success'] and result.get('primary_result'):
            primary = result['primary_result']

            # Display image with bbox if we have the path
            if result.get('image_path'):
                self._display_image_with_bbox(result['image_path'], primary)
            else:
                self.image_label.setText("<center><i>Image preview not available</i></center>")

            # Show details
            bbox_str = f"[{primary['bbox'][0]}, {primary['bbox'][1]}, {primary['bbox'][2]}, {primary['bbox'][3]}]" if primary['bbox'] else "None"

            details_html = f"""
            <h3>{result['image_name']}</h3>
            <p><b>Processing Time:</b> {result['processing_time']:.3f}s</p>
            <hr>
            <p><b>Detection:</b></p>
            <ul>
                <li>Bounding Box: {bbox_str}</li>
                <li>Confidence: {primary['bbox_confidence']:.2%}</li>
            </ul>
            <p><b>View Classification:</b></p>
            <ul>
                <li>Class: <b>{primary['view']}</b></li>
                <li>Confidence: {primary['view_confidence']:.2%}</li>
            </ul>
            <p><b>Defect Classification:</b></p>
            <ul>
                <li>Class: <b style="color: {'green' if primary['defect'] == 'PASS' else 'red'};">{primary['defect']}</b></li>
                <li>Confidence: {primary['defect_confidence']:.2%}</li>
            </ul>
            """

            if len(result['detections']) > 1:
                details_html += f"<p><i>Note: {len(result['detections'])} detections found (showing primary)</i></p>"

            self.details_label.setText(details_html)
        else:
            self.image_label.clear()
            self.image_label.setText("<center><i>Image preview not available</i></center>")

            error_msg = result.get('error', 'Unknown error')
            self.details_label.setText(f"""
            <h3>{result['image_name']}</h3>
            <p style="color: red;"><b>Status:</b> Failed</p>
            <p><b>Error:</b> {error_msg}</p>
            """)

    def _display_image_with_bbox(self, image_path: str, primary: Dict):
        """Display image with bounding box."""
        try:
            if not os.path.exists(image_path):
                self.image_label.setText(f"<center><i>Image file not found:<br/>{image_path}</i></center>")
                return

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.image_label.setText(f"<center><i>Failed to load image:<br/>{image_path}</i></center>")
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Draw bounding box if available
            if primary['bbox']:
                x1, y1, x2, y2 = primary['bbox']

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"{primary['defect']} ({primary['defect_confidence']:.2%})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Get label size
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw label background
                cv2.rectangle(
                    image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )

                # Draw label text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness
                )

            # Convert to QPixmap
            from PyQt6.QtGui import QImage
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self.image_label.setText(f"<center><i>Error displaying image:<br/>{str(e)}</i></center>")

    def _show_previous(self):
        """Show previous result."""
        if self.current_index > 0:
            self.table.selectRow(self.current_index - 1)

    def _show_next(self):
        """Show next result."""
        if self.current_index < len(self.filtered_results) - 1:
            self.table.selectRow(self.current_index + 1)

    def _export_filtered_results(self):
        """Export filtered results to CSV."""
        if not self.filtered_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Filtered Results",
            "filtered_inference_results.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                from src.inference.result_exporter import ResultExporter
                ResultExporter.export_csv(self.filtered_results, file_path)

                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Export Success",
                    f"Filtered results exported to:\n{file_path}\n\n"
                    f"Total: {len(self.filtered_results)} results"
                )
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export results:\n{e}"
                )
