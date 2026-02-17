"""
Feedback widget for reviewing and correcting bird identifications.
Shows recent detections in a scrollable gallery with inline correction UI.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QGridLayout,
                             QLineEdit, QCompleter, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from pathlib import Path
from correction_manager import CorrectionManager
import csv


class DetectionCard(QFrame):
    """Card showing a single detection with inline confirm/correct controls"""

    confirmed = pyqtSignal(str, str, float)  # path, prediction, confidence
    corrected = pyqtSignal(str, str, str, float)  # path, original, corrected, confidence
    exclude_requested = pyqtSignal(str)  # path

    def __init__(self, image_path: str, prediction: str, confidence: float,
                 timestamp: str, valid_species: list, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.prediction = prediction
        self.confidence = confidence
        self.timestamp = timestamp

        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMaximumWidth(250)

        layout = QVBoxLayout()

        # Image
        img_label = QLabel()
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaledToHeight(150, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
        except Exception:
            img_label.setText("No preview")
        layout.addWidget(img_label)

        # Prediction info
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        self._update_info_label()
        layout.addWidget(self.info_label)

        # Confirm / Exclude buttons row
        btn_row = QHBoxLayout()

        confirm_btn = QPushButton("Correct \u2713")
        confirm_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 4px;"
        )
        confirm_btn.setToolTip("Confirm this identification is correct")
        confirm_btn.clicked.connect(
            lambda: self.confirmed.emit(self.image_path, self.prediction, self.confidence)
        )
        btn_row.addWidget(confirm_btn)

        exclude_btn = QPushButton("\u2715")
        exclude_btn.setFixedWidth(30)
        exclude_btn.setStyleSheet(
            "background-color: rgba(255, 0, 0, 0.7); color: white; font-weight: bold; padding: 4px;"
        )
        exclude_btn.setToolTip("Exclude from training (not a bird, bad image, etc.)")
        exclude_btn.clicked.connect(
            lambda: self.exclude_requested.emit(self.image_path)
        )
        btn_row.addWidget(exclude_btn)

        layout.addLayout(btn_row)

        # Correction input row (text field + save)
        correction_row = QHBoxLayout()

        self.species_input = QLineEdit()
        self.species_input.setPlaceholderText("Or type correct species...")
        completer = QCompleter(valid_species)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.species_input.setCompleter(completer)
        self.species_input.returnPressed.connect(self._submit_correction)
        correction_row.addWidget(self.species_input)

        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(45)
        save_btn.clicked.connect(self._submit_correction)
        correction_row.addWidget(save_btn)

        layout.addLayout(correction_row)

        # Feedback label (shown after action)
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_label)

        self.setLayout(layout)

    def _submit_correction(self):
        species = self.species_input.text().strip()
        if species:
            self.corrected.emit(self.image_path, self.prediction, species, self.confidence)

    def show_feedback(self, message: str, color: str = "green"):
        self.feedback_label.setStyleSheet(f"color: {color};")
        self.feedback_label.setText(message)

    def _update_info_label(self):
        """Refresh the prediction info label from current state."""
        self.info_label.setText(
            f"<b>{self.prediction}</b><br>"
            f"{self.confidence:.1%} confidence<br>"
            f"<small>{self.timestamp}</small>"
        )

    def update_prediction(self, new_prediction: str):
        """Update the card to show a corrected prediction."""
        self.prediction = new_prediction
        self._update_info_label()


class FeedbackWidget(QWidget):
    """Widget for reviewing and correcting recent detections"""

    def __init__(self, max_recent=20, parent=None):
        super().__init__(parent)
        self.max_recent = max_recent
        self.correction_manager = CorrectionManager()
        self.recent_detections = []

        self._init_ui()
        self._load_recent_detections()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("<h2>Recent Detections</h2>")
        header_layout.addWidget(title)

        self.stats_label = QLabel("")
        self._update_stats()
        header_layout.addWidget(self.stats_label)

        header_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_recent_detections)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Scrollable area for detection cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.cards_container = QWidget()
        self.cards_layout = QGridLayout()
        self.cards_container.setLayout(self.cards_layout)

        scroll_area.setWidget(self.cards_container)
        layout.addWidget(scroll_area)

        self.setLayout(layout)

    def _load_recent_detections(self):
        """Load recent detections from logs"""
        # Clear existing cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.recent_detections = []

        detections_csv = Path("logs/detections.csv")
        if not detections_csv.exists():
            self.cards_layout.addWidget(
                QLabel("No detections yet. Start monitoring to see detections here!"),
                0, 0
            )
            return

        with open(detections_csv, 'r') as f:
            reader = csv.DictReader(f)
            all_detections = list(reader)

        recent = all_detections[-self.max_recent:]
        recent.reverse()

        valid_species = self.correction_manager.get_valid_species()

        row, col = 0, 0
        cols_per_row = 3

        for detection in recent:
            species_from_csv = detection.get('species_name', '')
            snapshot_file = detection.get('snapshot_file', '')

            # Handle old rows (7 columns): snapshot file is in species_name column
            if species_from_csv and species_from_csv.endswith('.jpg') and not snapshot_file:
                snapshot_file = species_from_csv
                species_from_csv = ''

            snapshot_file = snapshot_file.strip()
            if not snapshot_file or not snapshot_file.endswith('.jpg'):
                continue

            if not Path(snapshot_file).exists():
                continue

            # Check if this has been corrected or excluded
            correction = self.correction_manager.get_correction(snapshot_file)

            if correction and correction.get('status') == 'excluded':
                prediction = "Excluded from training"
                confidence = 0.0
            elif correction:
                prediction = f"{correction['corrected_species']} (corrected)"
                confidence = correction.get('confidence', 0.0)
            else:
                if species_from_csv and species_from_csv not in ['Unknown', '']:
                    prediction = species_from_csv
                else:
                    filename = Path(snapshot_file).stem
                    parts = filename.split('_')
                    if len(parts) >= 3 and parts[0] == 'bird':
                        date_idx = None
                        for i, part in enumerate(parts):
                            if len(part) == 8 and part.isdigit():
                                date_idx = i
                                break
                        if date_idx and date_idx > 1:
                            species_parts = parts[1:date_idx]
                            prediction = ' '.join(species_parts).title()
                        else:
                            prediction = "Unknown"
                    else:
                        prediction = "Unknown"

                confidence = float(detection.get('confidence', 0.0))

            timestamp = detection.get('timestamp', '')

            card = DetectionCard(
                snapshot_file, prediction, confidence, timestamp, valid_species
            )
            card.confirmed.connect(self._handle_confirm)
            card.corrected.connect(self._handle_correction)
            card.exclude_requested.connect(self._handle_exclude)

            self.cards_layout.addWidget(card, row, col)

            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1

            self.recent_detections.append({
                'path': snapshot_file,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': timestamp
            })

        if not self.recent_detections:
            self.cards_layout.addWidget(
                QLabel("No recent detections with images found."),
                0, 0
            )

        self._update_stats()

    def _handle_confirm(self, image_path: str, prediction: str, confidence: float):
        """Mark a detection as confirmed correct"""
        prediction = prediction.replace(" (corrected)", "")
        self.correction_manager.add_correction(
            image_path, prediction, prediction, confidence
        )
        # Find the card and show feedback
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, DetectionCard) and widget.image_path == image_path:
                widget.update_prediction(f"{prediction} (confirmed)")
                widget.show_feedback("Confirmed \u2713")
                break
        self._update_stats()

    def _handle_correction(self, image_path: str, original: str, corrected: str, confidence: float):
        """Save a species correction"""
        original = original.replace(" (corrected)", "")
        is_valid, result = self.correction_manager.validate_species_name(corrected)
        if not is_valid:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, DetectionCard) and widget.image_path == image_path:
                    widget.show_feedback(result, "red")
                    break
            return

        # Extract normalized name
        normalized = result.split("(")[0].strip() if "(" in result else result

        self.correction_manager.add_correction(
            image_path, original, normalized, confidence
        )

        # Find the card and show feedback
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, DetectionCard) and widget.image_path == image_path:
                widget.update_prediction(f"{normalized} (corrected)")
                widget.show_feedback(f"Saved: {normalized} \u2713")
                widget.species_input.clear()
                break
        self._update_stats()

    def _handle_exclude(self, image_path: str):
        """Handle request to exclude image from training"""
        reply = QMessageBox.question(
            self,
            "Exclude from Training",
            "Exclude this image from the training dataset?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.correction_manager.exclude_from_training(image_path)
            self._load_recent_detections()

    def _update_stats(self):
        stats = self.correction_manager.get_stats()
        self.stats_label.setText(
            f"<small>Corrections: {stats['total_corrections']} | "
            f"Training ready: {stats['valid_for_training']}</small>"
        )
