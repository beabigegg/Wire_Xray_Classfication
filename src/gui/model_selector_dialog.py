"""
Model selector dialog for managing trained model versions.

This module provides a dialog window for viewing, selecting, and managing
trained model versions across all three model types (Detection, View, Defect).
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QTextBrowser, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.core.database import Database, DatabaseError
from src.training.model_manager import ModelManager, ModelManagerError


class ModelSelectorDialog(QDialog):
    """
    Dialog for viewing and managing model versions.

    Provides three sections (one for each model type) with:
    - List of all model versions
    - Model details panel
    - Management buttons (Set Active, Delete, Refresh)
    """

    def __init__(self, db: Database, models_dir: str = 'models', parent=None):
        """
        Initialize model selector dialog.

        Args:
            db: Database instance for model version management
            models_dir: Base directory for model storage
            parent: Parent widget (optional)
        """
        super().__init__(parent)

        self.db = db
        self.model_manager = ModelManager(models_dir=models_dir)
        self.models_dir = Path(models_dir)

        # Model type configurations
        self.model_types = {
            'detection': {
                'label': 'Detection Model (YOLO)',
                'metric_key': 'map50',
                'metric_label': 'mAP@0.5'
            },
            'view': {
                'label': 'View Classifier (TOP/SIDE)',
                'metric_key': 'accuracy',
                'metric_label': 'Accuracy'
            },
            'defect': {
                'label': 'Defect Classifier',
                'metric_key': 'balanced_accuracy',
                'metric_label': 'Balanced Accuracy'
            }
        }

        # UI components storage
        self.model_lists: Dict[str, QListWidget] = {}
        self.details_panels: Dict[str, QTextBrowser] = {}
        self.set_active_buttons: Dict[str, QPushButton] = {}
        self.delete_buttons: Dict[str, QPushButton] = {}

        # Setup UI
        self.setWindowTitle("Model Version Manager")
        self.setMinimumSize(1200, 600)
        self._setup_ui()

        # Load initial data
        self._refresh_all_models()

    def _setup_ui(self):
        """Create and layout UI components."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create a section for each model type
        for model_type, config in self.model_types.items():
            section = self._create_model_section(model_type, config)
            main_layout.addWidget(section)

        self.setLayout(main_layout)

    def _create_model_section(self, model_type: str, config: Dict[str, str]) -> QGroupBox:
        """
        Create a section for one model type.

        Args:
            model_type: Type of model (detection, view, defect)
            config: Configuration dict with label and metric info

        Returns:
            QGroupBox containing the complete section
        """
        group = QGroupBox(config['label'])
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Model list
        list_label = QLabel("Model Versions:")
        list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(list_label)

        model_list = QListWidget()
        model_list.setMinimumHeight(200)
        model_list.itemClicked.connect(
            lambda item: self._on_model_selected(model_type, item)
        )
        self.model_lists[model_type] = model_list
        layout.addWidget(model_list)

        # Details panel
        details_label = QLabel("Model Details:")
        details_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(details_label)

        details_panel = QTextBrowser()
        details_panel.setMinimumHeight(150)
        details_panel.setMaximumHeight(200)
        details_panel.setOpenExternalLinks(False)
        self.details_panels[model_type] = details_panel
        layout.addWidget(details_panel)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)

        # Set Active button
        set_active_btn = QPushButton("Set as Active")
        set_active_btn.clicked.connect(
            lambda: self._on_set_active_clicked(model_type)
        )
        set_active_btn.setEnabled(False)
        self.set_active_buttons[model_type] = set_active_btn
        button_layout.addWidget(set_active_btn)

        # Delete button
        delete_btn = QPushButton("Delete Model")
        delete_btn.clicked.connect(
            lambda: self._on_delete_clicked(model_type)
        )
        delete_btn.setEnabled(False)
        self.delete_buttons[model_type] = delete_btn
        button_layout.addWidget(delete_btn)

        # Refresh button
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(
            lambda: self._refresh_models(model_type)
        )
        button_layout.addWidget(refresh_btn)

        layout.addLayout(button_layout)

        group.setLayout(layout)
        return group

    def _refresh_all_models(self):
        """Refresh model lists for all model types."""
        for model_type in self.model_types.keys():
            self._refresh_models(model_type)

    def _refresh_models(self, model_type: str):
        """
        Refresh the model list for a specific model type.

        Args:
            model_type: Type of model (detection, view, defect)
        """
        try:
            # Clear current list
            model_list = self.model_lists[model_type]
            model_list.clear()

            # Clear details
            self.details_panels[model_type].clear()
            self.set_active_buttons[model_type].setEnabled(False)
            self.delete_buttons[model_type].setEnabled(False)

            # Get models from database
            db_models = self.db.get_model_versions(model_type=model_type)

            # Get models from filesystem
            fs_models = self.model_manager.list_models(
                model_type=model_type,
                sort_by='date'
            )

            # Merge: prioritize database records but include filesystem-only models
            db_versions = {m['version']: m for m in db_models}

            # Parse metrics from JSON strings in database
            for version, model in db_versions.items():
                if model.get('metrics_json'):
                    try:
                        model['metrics'] = json.loads(model['metrics_json'])
                    except json.JSONDecodeError:
                        model['metrics'] = {}

            # Add filesystem models not in database
            for fs_model in fs_models:
                version = fs_model.get('version', '')
                if version and version not in db_versions:
                    # Add to merged list with is_active=False
                    db_versions[version] = {
                        **fs_model,
                        'is_active': False,
                        'from_fs': True
                    }

            # Sort by date (newest first)
            sorted_models = sorted(
                db_versions.values(),
                key=lambda x: x.get('saved_at', x.get('created_at', '')),
                reverse=True
            )

            # Populate list
            for model in sorted_models:
                version = model.get('version', 'Unknown')
                is_active = model.get('is_active', 0)

                # Format display text
                if is_active:
                    display_text = f"[ACTIVE] {version}"
                else:
                    display_text = version

                # Create list item
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, model)

                # Bold font for active model
                if is_active:
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)

                model_list.addItem(item)

            # Show count in status
            if model_list.count() == 0:
                placeholder_item = QListWidgetItem("No models found")
                placeholder_item.setFlags(Qt.ItemFlag.NoItemFlags)
                model_list.addItem(placeholder_item)

        except (DatabaseError, ModelManagerError) as e:
            QMessageBox.critical(
                self,
                "Error Loading Models",
                f"Failed to load {model_type} models:\n{str(e)}"
            )

    def _on_model_selected(self, model_type: str, item: QListWidgetItem):
        """
        Handle model selection in list.

        Args:
            model_type: Type of model
            item: Selected list item
        """
        # Get model data
        model = item.data(Qt.ItemDataRole.UserRole)
        if not model:
            return

        # Enable buttons
        self.set_active_buttons[model_type].setEnabled(True)
        self.delete_buttons[model_type].setEnabled(True)

        # Display details
        self._display_model_details(model_type, model)

    def _display_model_details(self, model_type: str, model: Dict[str, Any]):
        """
        Display detailed information about a model.

        Args:
            model_type: Type of model
            model: Model metadata dictionary
        """
        details_panel = self.details_panels[model_type]
        config = self.model_types[model_type]

        # Build HTML content
        html = "<html><body style='font-family: monospace; font-size: 10pt;'>"

        # Version and status
        version = model.get('version', 'Unknown')
        is_active = model.get('is_active', 0)
        html += f"<h3 style='margin: 5px 0;'>Version: {version}</h3>"

        if is_active:
            html += "<p style='color: green; font-weight: bold; margin: 5px 0;'>STATUS: ACTIVE</p>"
        else:
            html += "<p style='color: gray; margin: 5px 0;'>Status: Inactive</p>"

        html += "<hr style='margin: 8px 0;'>"

        # Model path
        filepath = model.get('filepath', model.get('model_path', 'Unknown'))
        file_exists = Path(filepath).exists() if filepath != 'Unknown' else False

        html += f"<p style='margin: 3px 0;'><b>Path:</b> {filepath}</p>"
        if not file_exists and filepath != 'Unknown':
            html += "<p style='color: red; margin: 3px 0;'>⚠ File not found on disk</p>"

        # Training date
        saved_at = model.get('saved_at', model.get('created_at', 'Unknown'))
        if saved_at != 'Unknown':
            # Format timestamp
            if 'T' in saved_at:
                date_part, time_part = saved_at.split('T')
                time_part = time_part.split('.')[0]  # Remove microseconds
                saved_at = f"{date_part} {time_part}"

        html += f"<p style='margin: 3px 0;'><b>Trained:</b> {saved_at}</p>"

        # Metrics
        metrics = model.get('metrics', {})
        if metrics:
            html += "<p style='margin: 8px 0 3px 0;'><b>Performance Metrics:</b></p>"
            html += "<ul style='margin: 0; padding-left: 20px;'>"

            # Show main metric first
            main_metric_key = config['metric_key']
            if main_metric_key in metrics:
                value = metrics[main_metric_key]
                label = config['metric_label']
                html += f"<li><b>{label}:</b> {value:.4f} ({value*100:.2f}%)</li>"

            # Show other metrics
            for key, value in metrics.items():
                if key != main_metric_key and isinstance(value, (int, float)):
                    display_key = key.replace('_', ' ').title()
                    if value <= 1.0:  # Assume it's a ratio
                        html += f"<li>{display_key}: {value:.4f} ({value*100:.2f}%)</li>"
                    else:
                        html += f"<li>{display_key}: {value:.4f}</li>"

            html += "</ul>"
        else:
            html += "<p style='margin: 8px 0 3px 0; color: gray;'>No metrics available</p>"

        # Training config
        config_data = model.get('config', {})
        if config_data:
            html += "<p style='margin: 8px 0 3px 0;'><b>Training Configuration:</b></p>"
            html += "<ul style='margin: 0; padding-left: 20px;'>"

            # Show key config items
            key_items = ['epochs', 'batch_size', 'learning_rate', 'lr', 'optimizer']
            for key in key_items:
                if key in config_data:
                    value = config_data[key]
                    display_key = key.replace('_', ' ').title()
                    html += f"<li>{display_key}: {value}</li>"

            html += "</ul>"

        # File size
        if 'file_size_mb' in model:
            size_mb = model['file_size_mb']
            html += f"<p style='margin: 8px 0 3px 0;'><b>File Size:</b> {size_mb:.2f} MB</p>"

        html += "</body></html>"

        details_panel.setHtml(html)

    def _on_set_active_clicked(self, model_type: str):
        """
        Handle Set Active button click.

        Args:
            model_type: Type of model
        """
        # Get selected model
        model_list = self.model_lists[model_type]
        selected_items = model_list.selectedItems()

        if not selected_items:
            return

        model = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not model:
            return

        # Check if already active
        if model.get('is_active', 0):
            QMessageBox.information(
                self,
                "Already Active",
                "This model is already set as active."
            )
            return

        try:
            # Get model ID from database or create record if from filesystem
            model_id = model.get('id')

            if not model_id:
                # Model is from filesystem only, need to save to database
                version = model.get('version', '')
                filepath = model.get('filepath', model.get('model_path', ''))
                model_name = model.get('model_name', 'unknown')
                metrics = model.get('metrics', {})

                model_id = self.db.save_model_version(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    filepath=filepath,
                    metrics=metrics,
                    set_active=True
                )
            else:
                # Update existing record
                success = self.db.set_active_model(model_id)
                if not success:
                    raise DatabaseError("Failed to set active model")

            QMessageBox.information(
                self,
                "Success",
                f"Model version '{model.get('version', 'Unknown')}' is now active."
            )

            # Refresh the list to show updated active status
            self._refresh_models(model_type)

        except DatabaseError as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set active model:\n{str(e)}"
            )

    def _on_delete_clicked(self, model_type: str):
        """
        Handle Delete button click.

        Args:
            model_type: Type of model
        """
        # Get selected model
        model_list = self.model_lists[model_type]
        selected_items = model_list.selectedItems()

        if not selected_items:
            return

        model = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not model:
            return

        version = model.get('version', 'Unknown')
        is_active = model.get('is_active', 0)

        # Build confirmation message
        msg = f"Are you sure you want to delete model version:\n\n{version}\n\n"

        if is_active:
            msg += "⚠ WARNING: This is the currently active model!\n"
            msg += "You will need to select a new active model after deletion.\n\n"

        msg += "This will:\n"
        msg += "• Delete the model file from disk\n"
        msg += "• Remove the database record\n"
        msg += "\nThis action cannot be undone."

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Delete model file
            filepath = model.get('filepath', model.get('model_path', ''))
            if filepath and filepath != 'Unknown':
                try:
                    self.model_manager.delete_model(filepath, delete_metadata=True)
                except ModelManagerError as e:
                    # File might not exist, continue anyway
                    print(f"Warning: Could not delete model file: {e}")

            # Delete database record if exists
            model_id = model.get('id')
            if model_id:
                self.db.delete_model_version(model_id)

            QMessageBox.information(
                self,
                "Success",
                f"Model version '{version}' has been deleted."
            )

            # If we deleted the active model, prompt to select new one
            if is_active:
                QMessageBox.warning(
                    self,
                    "Active Model Deleted",
                    f"The active {model_type} model has been deleted.\n"
                    "Please select a new active model."
                )

            # Refresh the list
            self._refresh_models(model_type)

        except (DatabaseError, ModelManagerError) as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to delete model:\n{str(e)}"
            )


# Test if run directly
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    from src.core.config import Config

    app = QApplication(sys.argv)

    # Create test database and model manager
    db = Database("test_annotations.db")

    # Create dialog
    dialog = ModelSelectorDialog(db, models_dir='models')
    dialog.show()

    sys.exit(app.exec())
