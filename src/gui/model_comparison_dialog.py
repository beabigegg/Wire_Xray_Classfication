"""
Model comparison dialog for comparing trained models.

This module provides a GUI dialog for comparing 2-4 trained models,
showing metrics, deltas, and recommendations.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QCheckBox, QMessageBox, QHeaderView, QTextEdit,
    QGroupBox, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from typing import List, Dict, Optional
import subprocess
from pathlib import Path

from src.training.model_comparator import ModelComparator
from src.core.database import Database


class ModelComparisonDialog(QDialog):
    """
    Dialog for comparing multiple trained models.

    Features:
    - Select 2-4 models to compare
    - View metrics comparison table with deltas
    - Color-coded improvements/regressions
    - Recommendation display
    - View TensorBoard runs
    - Set active model
    """

    def __init__(self, model_type: str, db: Database, parent=None):
        """
        Initialize model comparison dialog.

        Args:
            model_type: Model type - supports VIEW-aware architecture:
                       'view', 'detection', 'detection_top', 'detection_side',
                       'defect', 'defect_top', 'defect_side'
            db: Database instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.model_type = model_type
        self.db = db
        self.comparator = ModelComparator(db)

        # State
        self.available_models = []
        self.selected_models = []
        self.comparison_result = None

        # UI components
        self.model_checkboxes = {}
        self.comparison_table = None
        self.recommendation_text = None

        self._init_ui()
        self._load_models()

    def _init_ui(self):
        """Initialize UI components."""
        display_name = self._get_model_type_display_name()
        self.setWindowTitle(f"Compare {display_name}")
        self.setMinimumSize(900, 700)

        layout = QVBoxLayout()

        # Title
        title = QLabel(f"<h2>Compare {display_name}</h2>")
        layout.addWidget(title)

        # Model selection section
        selection_group = self._create_selection_section()
        layout.addWidget(selection_group)

        # Comparison table section
        table_group = self._create_table_section()
        layout.addWidget(table_group)

        # Recommendation section
        rec_group = self._create_recommendation_section()
        layout.addWidget(rec_group)

        # Action buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_selection_section(self) -> QGroupBox:
        """Create model selection section."""
        group = QGroupBox("Select Models to Compare (2-4 models)")
        layout = QVBoxLayout()

        # Info label
        info = QLabel("Select 2 to 4 models to compare their performance:")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Checkboxes container (will be populated in _load_models)
        self.checkbox_layout = QVBoxLayout()
        layout.addLayout(self.checkbox_layout)

        # Compare button
        self.compare_button = QPushButton("Compare Selected Models")
        self.compare_button.clicked.connect(self._run_comparison)
        self.compare_button.setEnabled(False)
        layout.addWidget(self.compare_button)

        group.setLayout(layout)
        return group

    def _create_table_section(self) -> QGroupBox:
        """Create comparison table section."""
        group = QGroupBox("Comparison Results")
        layout = QVBoxLayout()

        # Create table
        self.comparison_table = QTableWidget()
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.comparison_table)

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Legend:"))

        improvement_label = QLabel("  Green = Improvement  ")
        improvement_label.setStyleSheet("background-color: #90EE90; padding: 2px;")
        legend_layout.addWidget(improvement_label)

        regression_label = QLabel("  Red = Regression  ")
        regression_label.setStyleSheet("background-color: #FFB6C1; padding: 2px;")
        legend_layout.addWidget(regression_label)

        neutral_label = QLabel("  Gray = Neutral  ")
        neutral_label.setStyleSheet("background-color: #D3D3D3; padding: 2px;")
        legend_layout.addWidget(neutral_label)

        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        group.setLayout(layout)
        return group

    def _create_recommendation_section(self) -> QGroupBox:
        """Create recommendation section."""
        group = QGroupBox("Recommendation")
        layout = QVBoxLayout()

        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setMaximumHeight(120)
        layout.addWidget(self.recommendation_text)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        # View TensorBoard button
        self.tensorboard_button = QPushButton("View TensorBoard")
        self.tensorboard_button.clicked.connect(self._open_tensorboard)
        self.tensorboard_button.setEnabled(False)
        layout.addWidget(self.tensorboard_button)

        # Set as Active button
        self.set_active_button = QPushButton("Set Recommended as Active")
        self.set_active_button.clicked.connect(self._set_recommended_active)
        self.set_active_button.setEnabled(False)
        layout.addWidget(self.set_active_button)

        layout.addStretch()

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        return layout

    def _load_models(self):
        """Load available models for this type."""
        try:
            # Get all trained models of this type
            models = self.db.get_trained_models(self.model_type)
            self.available_models = models

            if not models:
                info_label = QLabel("<i>No trained models found. Please train models first.</i>")
                self.checkbox_layout.addWidget(info_label)
                return

            # Create checkboxes for each model
            valid_model_count = 0
            for i, model in enumerate(models):
                # Debug: check model structure
                if 'id' not in model or model['id'] is None:
                    print(f"Warning: Model {i} missing 'id' field. Keys: {list(model.keys())}")
                    continue

                try:
                    checkbox = QCheckBox(self._format_model_name(model))
                    checkbox.stateChanged.connect(self._on_selection_changed)
                    self.model_checkboxes[model['id']] = checkbox
                    self.checkbox_layout.addWidget(checkbox)
                    valid_model_count += 1
                except Exception as e:
                    print(f"Warning: Failed to create checkbox for model {model.get('id', 'unknown')}: {e}")
                    continue

            if valid_model_count == 0:
                info_label = QLabel("<i>No valid models found for comparison.</i>")
                self.checkbox_layout.addWidget(info_label)

        except Exception as e:
            import traceback
            error_detail = f"Failed to load models:\n{e}\n\nDetails:\n{traceback.format_exc()}"
            print(error_detail)
            QMessageBox.critical(self, "Error", error_detail)

    def _get_model_type_display_name(self) -> str:
        """Get display name for model type."""
        display_names = {
            'view': 'View Classifier Models',
            'detection': 'Detection Models (YOLO) - Unified [Legacy]',
            'detection_top': 'Detection Models (YOLO) - TOP View',
            'detection_side': 'Detection Models (YOLO) - SIDE View',
            'defect': 'Defect Classifier Models - Unified [Legacy]',
            'defect_top': 'Defect Classifier Models - TOP View',
            'defect_side': 'Defect Classifier Models - SIDE View'
        }
        return display_names.get(self.model_type, f"{self.model_type.capitalize()} Models")

    def _format_model_name(self, model: Dict) -> str:
        """Format model name for display."""
        # Safely get model name
        name = model.get('model_name', None)
        if not name:
            # Fallback: use model_path filename if available
            model_path = model.get('model_path', '')
            if model_path:
                from pathlib import Path
                name = Path(model_path).stem
            else:
                name = 'Unknown'

        timestamp = model.get('timestamp', '')

        # Get primary metric value
        primary_metric = ModelComparator.get_primary_metric(self.model_type)
        metric_value = model.get(primary_metric, 0.0)

        # Format timestamp
        if timestamp:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                timestamp_str = str(timestamp)[:16] if timestamp else "Unknown"
        else:
            timestamp_str = "Unknown"

        # Format metric value
        if isinstance(metric_value, (int, float)):
            metric_str = f"{metric_value:.3f}"
        else:
            metric_str = str(metric_value)

        return f"{name} - {timestamp_str} ({primary_metric}: {metric_str})"

    def _on_selection_changed(self):
        """Handle selection change."""
        selected = [
            model_id for model_id, checkbox in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

        # Enable compare button if 2-4 models selected
        self.compare_button.setEnabled(2 <= len(selected) <= 4)

        if len(selected) > 4:
            # Automatically uncheck the last one
            for model_id, checkbox in self.model_checkboxes.items():
                if checkbox.isChecked() and model_id == selected[-1]:
                    checkbox.setChecked(False)
                    break

    def _run_comparison(self):
        """Run model comparison."""
        try:
            # Get selected model IDs
            selected_ids = [
                model_id for model_id, checkbox in self.model_checkboxes.items()
                if checkbox.isChecked()
            ]

            if len(selected_ids) < 2:
                QMessageBox.warning(self, "Selection Error", "Please select at least 2 models to compare.")
                return

            # Run comparison
            self.comparison_result = self.comparator.compare_models(
                model_ids=selected_ids,
                model_type=self.model_type
            )

            # Update UI
            self._update_comparison_table()
            self._update_recommendation()

            # Enable action buttons
            self.tensorboard_button.setEnabled(True)
            self.set_active_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"Failed to compare models:\n{e}")

    def _update_comparison_table(self):
        """Update comparison table with results."""
        if not self.comparison_result:
            return

        models = self.comparison_result['models']
        deltas = self.comparison_result.get('deltas', {})
        ranking = self.comparison_result.get('ranking', [])

        # Get all unique metrics
        all_metrics = set()
        for model in models:
            all_metrics.update(model.keys())

        # Exclude non-metric fields
        exclude_fields = {'id', 'model_name', 'timestamp', 'config', 'model_path'}
        metrics = sorted(list(all_metrics - exclude_fields))

        # Setup table
        self.comparison_table.setRowCount(len(metrics))
        self.comparison_table.setColumnCount(len(models) + 1)

        # Set headers
        headers = ['Metric'] + [f"Model {i+1}" for i in range(len(models))]
        self.comparison_table.setHorizontalHeaderLabels(headers)

        # Fill table
        for row, metric in enumerate(metrics):
            # Metric name
            metric_item = QTableWidgetItem(metric.replace('_', ' ').title())
            metric_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            self.comparison_table.setItem(row, 0, metric_item)

            # Model values
            for col, model in enumerate(models):
                value = model.get(metric, 0.0)

                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                # Check if this model is ranked (best for this metric)
                model_id = model.get('id')
                is_best = ranking and model_id and model_id == ranking[0]

                item = QTableWidgetItem(value_str)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color code based on ranking
                if is_best and col == 0:
                    item.setBackground(QColor(144, 238, 144))  # Light green for best

                self.comparison_table.setItem(row, col + 1, item)

    def _update_recommendation(self):
        """Update recommendation text."""
        if not self.comparison_result:
            return

        recommendation = self.comparison_result.get('recommendation', {})
        if not recommendation:
            return

        model_id = recommendation.get('model_id')
        model_name = recommendation.get('model_name', 'Unknown')
        reasoning = recommendation.get('reasoning', 'No reasoning provided')

        # Find model details
        model_details = None
        for model in self.comparison_result['models']:
            if model.get('id') == model_id:
                model_details = model
                break

        # Format recommendation
        html = f"""
        <div style="padding: 10px;">
            <h3 style="color: #2E7D32;">Recommended Model: {model_name}</h3>
            <p><b>Reasoning:</b> {reasoning}</p>
        """

        if model_details:
            primary_metric = ModelComparator.get_primary_metric(self.model_type)
            metric_value = model_details.get(primary_metric, 0.0)
            html += f"<p><b>{primary_metric}:</b> {metric_value:.4f}</p>"

        html += "</div>"

        self.recommendation_text.setHtml(html)

    def _open_tensorboard(self):
        """Open TensorBoard for selected models."""
        if not self.comparison_result:
            return

        try:
            # Get log directories for selected models
            log_dirs = []
            for model in self.comparison_result['models']:
                model_path = model.get('model_path', '')
                if model_path:
                    # Assume TensorBoard logs are in runs/ directory
                    run_dir = Path(model_path).parent.parent
                    if run_dir.exists():
                        log_dirs.append(str(run_dir))

            if not log_dirs:
                QMessageBox.warning(self, "TensorBoard", "No TensorBoard logs found for selected models.")
                return

            # Launch TensorBoard with multiple log dirs
            logdir_arg = ','.join([f"{self.model_type}_{i}:{d}" for i, d in enumerate(log_dirs)])

            # Use subprocess to launch TensorBoard
            subprocess.Popen(
                ['tensorboard', '--logdir_spec', logdir_arg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            QMessageBox.information(
                self,
                "TensorBoard",
                "TensorBoard is starting...\nIt will open in your browser shortly at http://localhost:6006"
            )

        except Exception as e:
            QMessageBox.warning(self, "TensorBoard Error", f"Failed to launch TensorBoard:\n{e}")

    def _set_recommended_active(self):
        """Set recommended model as active."""
        if not self.comparison_result:
            return

        recommendation = self.comparison_result.get('recommendation', {})
        if not recommendation:
            return

        model_id = recommendation.get('model_id')
        model_name = recommendation.get('model_name', 'Unknown')

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Set Active Model",
            f"Set {model_name} as the active {self.model_type} model?\n\n"
            f"This will be used for inference operations.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # set_active_model only takes model_id, not model_type
                # It will automatically get the model_type from the database
                self.db.set_active_model(model_id)
                QMessageBox.information(
                    self,
                    "Success",
                    f"{model_name} is now the active {self.model_type} model."
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set active model:\n{e}")
