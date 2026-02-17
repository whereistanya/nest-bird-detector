#!/usr/bin/env python3
"""
Dialog for reporting missed bird detections
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QLineEdit, QTextEdit, QCompleter,
                             QDialogButtonBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
from pathlib import Path
from missed_bird_logger import MissedBirdLogger
from correction_manager import CorrectionManager


class MissedBirdDialog(QDialog):
    """Dialog for reporting a bird that the detector missed"""

    def __init__(self, image: Image.Image, parent=None):
        super().__init__(parent)
        self.image = image
        self.logger = MissedBirdLogger()
        self.correction_manager = CorrectionManager()  # For species list
        self.saved_path = None

        self.setWindowTitle("Report Missed Bird")
        self.setMinimumWidth(500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Info text
        info_label = QLabel(
            "<b>Report a bird that the detector missed</b><br><br>"
            "This frame will be saved for later annotation and training.<br>"
            "If you know the species, enter it below (optional)."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Optional species input
        species_layout = QVBoxLayout()
        species_layout.addWidget(QLabel("Species (optional):"))

        self.species_input = QLineEdit()
        self.species_input.setPlaceholderText("e.g., House Finch, or leave blank if unknown")

        # Add autocomplete from valid species list
        valid_species = self.correction_manager.get_valid_species()
        completer = QCompleter(valid_species)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.species_input.setCompleter(completer)

        species_layout.addWidget(self.species_input)
        layout.addLayout(species_layout)

        # Optional notes
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(QLabel("Notes (optional):"))

        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "e.g., 'Small bird in upper left corner'\n"
            "'Bird on feeder, partially hidden'\n"
            "'Flying through frame'"
        )
        self.notes_input.setMaximumHeight(80)

        notes_layout.addWidget(self.notes_input)
        layout.addLayout(notes_layout)

        # Info about what happens next
        next_steps = QLabel(
            "<small><i>"
            "The frame will be saved to the 'missed_birds/' folder. "
            "You can later annotate these images with bounding boxes "
            "and use them to retrain the detection model."
            "</i></small>"
        )
        next_steps.setWordWrap(True)
        layout.addWidget(next_steps)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_report)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Focus on species input
        self.species_input.setFocus()

    def _save_report(self):
        """Save the missed bird report"""
        species = self.species_input.text().strip() or None
        notes = self.notes_input.toPlainText().strip()

        # Save using logger
        self.saved_path = self.logger.log_missed_bird(
            self.image,
            species=species,
            notes=notes
        )

        # Show confirmation
        QMessageBox.information(
            self,
            "Missed Bird Reported",
            f"Frame saved for later annotation:\n{self.saved_path}\n\n"
            f"Total missed birds logged: {self.logger.get_stats()['total']}"
        )

        self.accept()

    def get_saved_path(self):
        """Return the path where the image was saved"""
        return self.saved_path
