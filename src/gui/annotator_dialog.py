"""
Annotator Selection Dialog

This module provides a dialog for selecting/changing the annotator identity.
Implements OpenSpec requirement for annotator tracking and configurability.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QComboBox, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt
from typing import Optional
import json
from pathlib import Path


class AnnotatorDialog(QDialog):
    """
    Dialog for selecting annotator identity.

    Features:
    - Recent annotators dropdown
    - Custom annotator name input
    - Persistent history
    """

    def __init__(self, current_annotator: str = "", parent=None):
        """
        Initialize annotator dialog.

        Args:
            current_annotator: Current annotator name
            parent: Parent widget
        """
        super().__init__(parent)
        self.current_annotator = current_annotator
        self.selected_annotator = current_annotator
        self.history_file = Path.home() / ".wire_xray_annotators.json"

        self.setWindowTitle("Select Annotator")
        self.setModal(True)
        self.setMinimumWidth(400)

        self._init_ui()
        self._load_history()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Select Annotator Identity")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "Please select or enter your name. This will be used to track "
            "who created each annotation for audit purposes."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(desc_label)

        # Recent annotators section
        recent_label = QLabel("Recent Annotators:")
        layout.addWidget(recent_label)

        self.recent_combo = QComboBox()
        self.recent_combo.setEditable(False)
        self.recent_combo.currentTextChanged.connect(self._on_recent_selected)
        layout.addWidget(self.recent_combo)

        # OR separator
        or_label = QLabel("-- OR --")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label.setStyleSheet("color: gray; margin: 10px 0;")
        layout.addWidget(or_label)

        # Custom name section
        custom_label = QLabel("Enter New Annotator Name:")
        layout.addWidget(custom_label)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., John Doe")
        self.name_input.setText(self.current_annotator)
        self.name_input.textChanged.connect(self._on_name_changed)
        layout.addWidget(self.name_input)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _load_history(self):
        """Load recent annotators from history file."""
        recent_annotators = []

        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    recent_annotators = data.get('recent', [])
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, ignore and start fresh
                pass

        # Add current annotator if not in list
        if self.current_annotator and self.current_annotator not in recent_annotators:
            recent_annotators.insert(0, self.current_annotator)

        # Populate combo box
        if recent_annotators:
            self.recent_combo.addItems(recent_annotators)
            # Set current annotator as selected
            if self.current_annotator in recent_annotators:
                index = recent_annotators.index(self.current_annotator)
                self.recent_combo.setCurrentIndex(index)
        else:
            self.recent_combo.addItem("(No recent annotators)")
            self.recent_combo.setEnabled(False)

    def _save_history(self, annotator: str):
        """
        Save annotator to history.

        Args:
            annotator: Annotator name to save
        """
        recent_annotators = []

        # Load existing history
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    recent_annotators = data.get('recent', [])
            except (json.JSONDecodeError, IOError):
                pass

        # Add new annotator to front (remove if already exists)
        if annotator in recent_annotators:
            recent_annotators.remove(annotator)
        recent_annotators.insert(0, annotator)

        # Keep only last 10
        recent_annotators = recent_annotators[:10]

        # Save
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({'recent': recent_annotators}, f, indent=2)
        except IOError:
            # If can't save, that's ok
            pass

    def _on_recent_selected(self, text: str):
        """Handle recent annotator selection."""
        if text and text != "(No recent annotators)":
            self.name_input.setText(text)

    def _on_name_changed(self, text: str):
        """Handle name input change."""
        self.ok_btn.setEnabled(bool(text.strip()))

    def _on_ok(self):
        """Handle OK button click."""
        name = self.name_input.text().strip()

        if not name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Please enter an annotator name."
            )
            return

        self.selected_annotator = name
        self._save_history(name)
        self.accept()

    def get_annotator(self) -> Optional[str]:
        """
        Get selected annotator name.

        Returns:
            Annotator name if dialog was accepted, None otherwise
        """
        return self.selected_annotator if self.result() == QDialog.DialogCode.Accepted else None
