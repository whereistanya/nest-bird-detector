"""
PyQt5 GUI Application
Main window for the Nest Bird Detector.
"""
import os
import logging
import threading
from datetime import datetime
from io import BytesIO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QStatusBar, QMessageBox,
    QGroupBox, QGridLayout, QDialog
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage

from PIL import Image, ImageDraw, ImageFont

from config import Config
from nest_client import NestClient
from detector import BirdDetector
from notifications import NotificationManager
from webrtc_manager import WebRTCStreamManager
from hf_species_classifier import HFSpeciesClassifier
from ebird_client import EBirdClient
from feedback_widget import FeedbackWidget
from missed_bird_dialog import MissedBirdDialog
from missed_bird_logger import MissedBirdLogger

logger = logging.getLogger(__name__)


class DetectionWorker(QThread):
    """Worker thread for running detection without blocking UI"""

    detection_complete = pyqtSignal(object, list)  # image, detections
    error_occurred = pyqtSignal(str)

    def __init__(self, webrtc_manager, detector):
        super().__init__()
        self.webrtc_manager = webrtc_manager
        self.detector = detector
        self.running = True

    def run(self):
        """Capture frame from WebRTC stream and run detection"""
        if not self.running:
            return

        try:
            # Capture frame from WebRTC stream
            image = self.webrtc_manager.capture_frame()

            if image is None:
                # Don't report errors during normal operation - frame capture can timeout
                # due to corrupted frames or temporary quality issues
                # Silently skip and try again on next detection cycle
                return

            # Detect birds
            detections = self.detector.detect_birds(image)

            # Emit results
            self.detection_complete.emit(image, detections)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        """Stop the worker"""
        self.running = False


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.config = Config()
        self.nest_client = None
        self.webrtc_manager = None
        self.detector = None
        self.notifier = None
        self.species_classifier = None  # Local species identification classifier
        self.timer = QTimer()
        self.worker = None
        self.detection_lock = threading.Lock()  # Thread-safe lock for detection flag
        self.detection_in_progress = False  # Flag to prevent concurrent detection
        self.stream_connected = False

        # Statistics
        self.total_checks = 0
        self.total_birds_detected = 0
        self.current_image = None
        self.last_good_timestamp = None  # Track when we last had a good frame
        self.current_detections = []  # Track current detections for stale frames
        self.last_detection_time = None  # Track when last bird was detected
        self.last_detection_species = None  # Track species of last detected bird
        self.feedback_window = None  # Feedback window for correcting identifications
        self.missed_bird_logger = MissedBirdLogger()  # Logger for missed detections

        self._init_ui()
        self._init_components()
        self._connect_signals()

        # Auto-connect stream on startup (after a short delay to allow UI to render)
        QTimer.singleShot(500, self._auto_connect_stream)

    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🐦 Nest Bird Detector")
        self.setGeometry(100, 100, 1000, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title = QLabel("Nest Bird Detector")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setText("No image yet\n\nClick 'Start Monitoring' to begin")
        main_layout.addWidget(self.image_label)

        # Controls
        controls = self._create_controls()
        main_layout.addWidget(controls)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_controls(self):
        """Create control panel"""
        group = QGroupBox("Controls")
        layout = QGridLayout()

        # Connect/Disconnect stream button
        self.connect_button = QPushButton("Connect Stream")
        self.connect_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        layout.addWidget(self.connect_button, 0, 0, 1, 2)

        # Start/Stop monitoring button
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.start_button.setEnabled(False)  # Disabled until stream connected
        layout.addWidget(self.start_button, 1, 0, 1, 2)

        # Check interval
        layout.addWidget(QLabel("Check Interval (seconds):"), 2, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(self.config.check_interval)
        layout.addWidget(self.interval_spin, 2, 1)

        # Confidence threshold
        layout.addWidget(QLabel("Confidence Threshold:"), 3, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(30, 90)
        self.confidence_slider.setValue(int(self.config.confidence_threshold * 100))
        self.confidence_label = QLabel(f"{self.config.confidence_threshold:.2f}")
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)
        layout.addLayout(confidence_layout, 3, 1)

        # Statistics
        layout.addWidget(QLabel("Total Checks:"), 4, 0)
        self.checks_label = QLabel("0")
        layout.addWidget(self.checks_label, 4, 1)

        layout.addWidget(QLabel("Birds Detected:"), 5, 0)
        self.birds_label = QLabel("0")
        layout.addWidget(self.birds_label, 5, 1)

        layout.addWidget(QLabel("Last Detection:"), 6, 0)
        self.last_detection_label = QLabel("None")
        layout.addWidget(self.last_detection_label, 6, 1)

        # Test frame button
        self.test_button = QPushButton("Test Frame Capture")
        self.test_button.setEnabled(False)  # Disabled until stream connected
        layout.addWidget(self.test_button, 7, 0, 1, 2)

        # Review detections button
        self.review_button = QPushButton("Review and Correct Detections")
        self.review_button.setStyleSheet("background-color: #2196F3; color: white;")
        layout.addWidget(self.review_button, 8, 0, 1, 2)

        # Report missed bird button
        self.missed_bird_button = QPushButton("Report Missed Bird")
        self.missed_bird_button.setStyleSheet("background-color: #FF9800; color: white;")
        self.missed_bird_button.setEnabled(False)  # Disabled until we have a frame
        layout.addWidget(self.missed_bird_button, 9, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _init_components(self):
        """Initialize Nest client, detector, and notifier"""
        try:
            # Check configuration
            if not self.config.is_configured():
                QMessageBox.critical(
                    self,
                    "Configuration Error",
                    "Nest API credentials not configured.\n\n"
                    "Please run: python3 setup_credentials.py"
                )
                return

            if not self.config.has_tokens():
                QMessageBox.critical(
                    self,
                    "Configuration Error",
                    "No access tokens found.\n\n"
                    "Please run: python3 setup_credentials.py"
                )
                return

            self.nest_client = NestClient(self.config)
            self.webrtc_manager = WebRTCStreamManager(self.nest_client)
            self.detector = BirdDetector(confidence_threshold=self.config.confidence_threshold)
            self.notifier = NotificationManager(
                detector=self.detector
            )

            # Initialize eBird client for location-based species filtering (if configured)
            ebird_client = None
            if self.config.ebird_api_key and self.config.ebird_region_code:
                print("\n" + "="*70)
                print("LOADING EBIRD REGIONAL SPECIES DATA")
                print("="*70)
                try:
                    ebird_client = EBirdClient(
                        api_key=self.config.ebird_api_key,
                        region_code=self.config.ebird_region_code
                    )
                except Exception as e:
                    print(f"⚠️  Failed to load eBird data: {e}")
                    print(f"   Continuing without location filtering")
            else:
                print("\n⚠️  eBird API not configured - location filtering disabled")
                print("   To enable, set EBIRD_API_KEY and EBIRD_REGION_CODE in .env")

            print("\n" + "="*70)
            print("LOADING SPECIES CLASSIFIER")
            print("="*70)
            self.species_classifier = HFSpeciesClassifier(ebird_client=ebird_client)

            self.status_bar.showMessage("Initialized successfully - Ready to connect stream")

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize:\n{e}")
            self.status_bar.showMessage("✗ Initialization failed")

    def _connect_signals(self):
        """Connect signals to slots"""
        self.connect_button.clicked.connect(self._toggle_stream)
        self.start_button.clicked.connect(self._toggle_monitoring)
        self.test_button.clicked.connect(self._test_frame_capture)
        self.review_button.clicked.connect(self._open_feedback_window)
        self.missed_bird_button.clicked.connect(self._report_missed_bird)
        self.confidence_slider.valueChanged.connect(self._update_confidence)
        self.interval_spin.valueChanged.connect(self._update_interval)
        self.timer.timeout.connect(self._check_for_birds)

    def _toggle_stream(self):
        """Connect or disconnect WebRTC stream"""
        if self.stream_connected:
            # Disconnect stream
            if self.timer.isActive():
                self._toggle_monitoring()  # Stop monitoring first

            self.status_bar.showMessage("🔄 Disconnecting stream...")
            self.webrtc_manager.stop_stream()
            self.stream_connected = False

            self.connect_button.setText("Connect Stream")
            self.connect_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
            self.start_button.setEnabled(False)
            self.test_button.setEnabled(False)
            self.status_bar.showMessage("✓ Stream disconnected")
        else:
            # Connect stream
            if self.webrtc_manager is None:
                QMessageBox.warning(self, "Error", "WebRTC manager not initialized")
                return

            self.status_bar.showMessage("🔄 Connecting to camera stream... (this may take 10-15 seconds)")
            self.connect_button.setEnabled(False)  # Prevent double-click
            QApplication.processEvents()  # Update UI

            success = self.webrtc_manager.start_stream()

            if success:
                self.stream_connected = True
                self.connect_button.setText("Disconnect Stream")
                self.connect_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f44336; color: white;")
                self.start_button.setEnabled(True)
                self.test_button.setEnabled(True)
                self.status_bar.showMessage("✅ Stream connected successfully!")
            else:
                QMessageBox.critical(
                    self,
                    "Connection Failed",
                    "Failed to connect to camera stream.\n\n"
                    "Common issues:\n"
                    "• ICE/STUN connection problems\n"
                    "• Firewall blocking WebRTC\n"
                    "• Invalid SDP offer/answer"
                )
                self.status_bar.showMessage("✗ Stream connection failed")

            self.connect_button.setEnabled(True)

    def _auto_connect_stream(self):
        """Automatically connect to stream on startup"""
        if not self.stream_connected and self.webrtc_manager is not None:
            print("🔄 Auto-connecting to stream on startup...")
            self._toggle_stream()

    def _toggle_monitoring(self):
        """Start or stop monitoring"""
        if self.timer.isActive():
            # Stop monitoring
            self.timer.stop()
            self.start_button.setText("Start Monitoring")
            self.status_bar.showMessage("Monitoring stopped")
        else:
            # Start monitoring
            if not self.stream_connected:
                QMessageBox.warning(self, "Error", "Please connect to stream first")
                return

            if self.webrtc_manager is None or self.detector is None:
                QMessageBox.warning(self, "Error", "Components not initialized")
                return

            # Update UI immediately for responsive feel
            self.start_button.setText("Stop Monitoring")
            self.start_button.setEnabled(False)  # Disable during first check
            self.status_bar.showMessage("🔄 Starting monitoring (capturing first frame)...")
            QApplication.processEvents()  # Force UI update

            interval_ms = self.interval_spin.value() * 1000
            self.timer.start(interval_ms)

            # Run first check immediately (will re-enable button when done)
            self._check_for_birds()

    def _check_for_birds(self):
        """Check camera for birds (runs periodically)"""
        if self.webrtc_manager is None or self.detector is None:
            return

        if not self.stream_connected:
            self.status_bar.showMessage("⚠️  Stream not connected - skipping check")
            return

        # Thread-safe check-and-set with lock (prevents concurrent workers)
        with self.detection_lock:
            if self.detection_in_progress:
                return  # Already in progress
            self.detection_in_progress = True

        # Log detection cycle start
        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence = self.confidence_slider.value() / 100.0
        print(f"\n🔎 [{timestamp}] Starting detection cycle (check #{self.total_checks + 1})")
        print(f"   Confidence threshold: {confidence:.2f}")
        print(f"   Stream connected: {self.stream_connected}")

        self.status_bar.showMessage("🔄 Checking for birds...")

        # Run detection in worker thread
        self.worker = DetectionWorker(self.webrtc_manager, self.detector)
        self.worker.detection_complete.connect(self._handle_detection_result)
        self.worker.error_occurred.connect(self._handle_detection_error)
        self.worker.finished.connect(self._on_detection_worker_finished)
        self.worker.start()

    def _identify_species_for_detections(self, image, detections):
        """
        Identify species for detected birds using local classifier.

        Args:
            image: PIL Image containing birds
            detections: List of Detection objects

        Returns:
            detections with species_name populated (if identified)
        """
        if not detections or not self.species_classifier:
            return detections

        # Identify species using local model (no cooldown needed)
        print(f"   🔍 Identifying species...")

        for detection in detections:
            try:
                # Identify species for this bird
                species_info = self.species_classifier.identify_species(
                    image=image,
                    bbox=detection.bbox
                )

                if species_info and 'common_name' in species_info:
                    detection.species_name = species_info['common_name']
                    confidence = species_info.get('score', 0)
                    print(f"   ✓ Identified: {detection.species_name} (confidence: {confidence:.2%})")
                else:
                    print(f"   ⚠️  Could not identify species")

            except Exception as e:
                print(f"   ⚠️  Species ID error: {e}")
                # Continue with other detections even if one fails
                continue

        return detections

    def _handle_detection_result(self, image, detections):
        """Handle detection results from worker thread"""
        self.total_checks += 1
        self.checks_label.setText(str(self.total_checks))

        print(f"   Frame captured: {image.size[0]}x{image.size[1]}")

        # Frame processing - proceed with normal update
        print(f"   Detections: {len(detections)} bird(s) found")
        if detections:
            for i, det in enumerate(detections, 1):
                print(f"     {i}. Confidence: {det.confidence:.2f}, BBox: {det.bbox}")

        # Clean up previous image before storing new one
        if self.current_image is not None:
            try:
                self.current_image.close()
            except (OSError, AttributeError) as e:
                # OSError: file operation failed
                # AttributeError: object doesn't have close() method
                print(f"Warning: Could not close previous image: {e}")

        # Store current image, detections, and timestamp
        self.current_image = image
        self.current_detections = detections
        self.last_good_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Enable missed bird button now that we have a frame
        self.missed_bird_button.setEnabled(True)

        # Update display
        self._display_image(image, detections, is_stale=False)

        # Handle detections
        if detections:
            self.total_birds_detected += len(detections)
            self.birds_label.setText(str(self.total_birds_detected))

            # Identify species using local classifier
            detections = self._identify_species_for_detections(image, detections)

            # Update last detection info
            self.last_detection_time = datetime.now()
            # Get species name from first detection that has one
            species_names = [d.species_name for d in detections if d.species_name]
            if species_names:
                self.last_detection_species = species_names[0]
            else:
                self.last_detection_species = None

            # Update last detection label
            time_str = self.last_detection_time.strftime("%I:%M:%S %p")
            if self.last_detection_species:
                self.last_detection_label.setText(f"{time_str} - {self.last_detection_species}")
            else:
                self.last_detection_label.setText(f"{time_str}")

            # Send notifications
            self.notifier.notify_bird_detected(image, detections)

            self.status_bar.showMessage(f"🐦 Found {len(detections)} bird(s)!")
            print(f"   Result: ✓ {len(detections)} bird(s) detected\n")
        else:
            self.status_bar.showMessage(f"✓ Check complete - No birds detected")
            print(f"   Result: ✓ No birds detected\n")

    def _handle_detection_error(self, error_message):
        """Handle detection errors"""
        self.status_bar.showMessage(f"✗ Error: {error_message}")
        print(f"Detection error: {error_message}")

    def _on_detection_worker_finished(self):
        """Called when detection worker finishes (success or error)"""
        with self.detection_lock:
            self.detection_in_progress = False

        # Re-enable the start/stop button after first detection
        if self.timer.isActive():
            self.start_button.setEnabled(True)

    def _add_timestamp_to_image(self, image, is_stale=False, target_display_width=None):
        """Add timestamp overlay to image showing when it was captured

        Args:
            image: PIL Image to add timestamp to
            is_stale: Whether this is a stale frame
            target_display_width: Width in pixels that the image will be displayed at
                                 (used to scale font size appropriately)
        """
        # Create a copy to avoid modifying the original
        img_with_timestamp = image.copy()
        draw = ImageDraw.Draw(img_with_timestamp)

        # Get appropriate timestamp
        if is_stale and self.last_good_timestamp:
            timestamp = self.last_good_timestamp
            text = f"⚠️ STALE FRAME - Last Good: {timestamp}"
            text_color = (255, 100, 100)  # Red for stale frames
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"Last Updated: {timestamp}"
            text_color = (255, 255, 255)  # White for fresh frames

        # Calculate font size based on image width (proportional sizing)
        # If target_display_width is provided, use that; otherwise use image width
        reference_width = target_display_width if target_display_width else image.width
        # Use ~1.75% of display width for font size (half the original size)
        font_size = max(12, int(reference_width * 0.0175))

        # Try to use a fixed-width font, fall back to others if not available
        font = None
        # Try Courier New (standard monospace font on macOS)
        for font_path in [
            "/System/Library/Fonts/Courier.dfont",  # Courier on macOS
            "/System/Library/Fonts/Monaco.dfont",  # Monaco (nice monospace)
            "/Library/Fonts/Courier New.ttf",  # Courier New
            "/System/Library/Fonts/Menlo.ttc",  # Menlo (Consolas equivalent on macOS)
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (IOError, OSError):
                continue

        # Fall back to default if no monospace font found
        if font is None:
            font = ImageFont.load_default()

        # Get text size for background
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position at top-left with some padding
        padding = 8
        x = padding
        y = padding

        # Draw semi-transparent background
        background_box = [
            x - 4,
            y - 2,
            x + text_width + 4,
            y + text_height + 2
        ]
        draw.rectangle(background_box, fill=(0, 0, 0, 180))

        # Draw white text on top
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        return img_with_timestamp

    def _display_image(self, image, detections, is_stale=False):
        """Display image in the GUI with bounding boxes and timestamp"""
        # Draw bounding boxes if birds detected
        if detections:
            display_image = self.detector.get_annotated_image(image, detections)
        else:
            display_image = image

        # Calculate final display size to determine appropriate font size
        label_size = self.image_label.size()
        image_aspect = display_image.width / display_image.height
        label_aspect = label_size.width() / label_size.height()

        # Calculate scaled dimensions (keeping aspect ratio)
        if image_aspect > label_aspect:
            # Image is wider - width constrained
            target_display_width = label_size.width()
        else:
            # Image is taller - height constrained
            target_display_width = int(label_size.height() * image_aspect)

        # Add timestamp overlay with font size scaled to display size
        display_image = self._add_timestamp_to_image(
            display_image,
            is_stale=is_stale,
            target_display_width=target_display_width
        )

        # Convert PIL Image to QPixmap
        buffer = BytesIO()
        display_image.save(buffer, format="JPEG")
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)

        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def _test_frame_capture(self):
        """Test capturing a frame from WebRTC stream"""
        if self.webrtc_manager is None:
            QMessageBox.warning(self, "Error", "WebRTC manager not initialized")
            return

        if not self.stream_connected:
            QMessageBox.warning(self, "Error", "Please connect to stream first")
            return

        self.status_bar.showMessage("Capturing test frame...")

        try:
            image = self.webrtc_manager.capture_frame()

            if image is None:
                # Check if still connected
                if self.webrtc_manager.is_connected():
                    QMessageBox.critical(self, "Test Failed", "Failed to capture frame from stream (timeout)")
                else:
                    QMessageBox.warning(self, "Test Failed", "Stream disconnected during test. Please wait for reconnection.")
                self.status_bar.showMessage("Test failed")
                return

            detections = self.detector.detect_birds(image)

            # Store image so missed bird button works
            self.current_image = image
            self.missed_bird_button.setEnabled(True)

            self._display_image(image, detections)

            # Save frame to disk for offline analysis
            import os
            os.makedirs("snapshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = f"snapshots/test_frame_{timestamp}.jpg"
            image.save(frame_path)

            # If detections found, also save annotated version
            if detections:
                annotated = self.detector.get_annotated_image(image, detections)
                annotated_path = f"snapshots/test_frame_{timestamp}_annotated.jpg"
                annotated.save(annotated_path)

            confidence = self.confidence_slider.value() / 100.0
            msg = f"Frame captured successfully!\n\n"
            msg += f"Detected {len(detections)} bird(s).\n"
            msg += f"Confidence threshold: {confidence:.2f}\n\n"
            msg += f"Saved to: {frame_path}"
            if detections:
                msg += f"\nAnnotated: {annotated_path}"

            QMessageBox.information(
                self,
                "Test Successful",
                msg
            )

            self.status_bar.showMessage(f"Test complete - saved to {frame_path}")

        except Exception as e:
            QMessageBox.critical(self, "Test Failed", f"Error: {e}")
            self.status_bar.showMessage("Test failed")

    def _open_feedback_window(self):
        """Open feedback window for reviewing and correcting detections"""
        # Create window if it doesn't exist or was closed
        if self.feedback_window is None or not self.feedback_window.isVisible():
            self.feedback_window = FeedbackWidget()
            self.feedback_window.setWindowTitle("Review and Correct Detections")
            self.feedback_window.setMinimumSize(900, 600)

        # Show and raise window
        self.feedback_window.show()
        self.feedback_window.raise_()
        self.feedback_window.activateWindow()

    def _report_missed_bird(self):
        """Report a bird that the detector missed"""
        if self.current_image is None:
            QMessageBox.warning(
                self,
                "No Frame Available",
                "No frame available to report.\n\n"
                "Please capture a frame first using 'Test Frame Capture' "
                "or wait for monitoring to capture a frame."
            )
            return

        # Make a copy of the image to avoid it being closed
        image_copy = self.current_image.copy()

        # Open dialog with current frame
        dialog = MissedBirdDialog(image_copy, self)
        if dialog.exec_() == QDialog.Accepted:
            stats = self.missed_bird_logger.get_stats()
            self.status_bar.showMessage(
                f"✓ Missed bird reported (total: {stats['total']})"
            )

    def _update_confidence(self, value):
        """Update confidence threshold"""
        threshold = value / 100.0
        self.confidence_label.setText(f"{threshold:.2f}")

        if self.detector:
            self.detector.set_confidence_threshold(threshold)

        self.config.confidence_threshold = threshold

    def _update_interval(self, value):
        """Update check interval"""
        self.config.check_interval = value

        # Update timer if running
        if self.timer.isActive():
            self.timer.setInterval(value * 1000)

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop monitoring
        if self.timer.isActive():
            self.timer.stop()

        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        # Disconnect WebRTC stream
        if self.stream_connected and self.webrtc_manager:
            self.webrtc_manager.stop_stream()

        # Clean up current image
        if self.current_image is not None:
            try:
                self.current_image.close()
            except (OSError, AttributeError) as e:
                # OSError: file operation failed
                # AttributeError: object doesn't have close() method
                logger.warning(f"Could not close image during shutdown: {e}")

        event.accept()


