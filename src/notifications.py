"""
Notification System
Handles macOS notifications, snapshot saving, and logging.
"""
import os
import stat
import csv
import re
from datetime import datetime
from typing import List, Optional, Dict
from PIL import Image
import time

try:
    import pync
    PYNC_AVAILABLE = True
except ImportError:
    PYNC_AVAILABLE = False
    print("Warning: pync not available. macOS notifications will be disabled.")


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Sanitize a string for safe use in filenames.

    Prevents path traversal and ensures cross-platform compatibility.

    Args:
        name: The string to sanitize (e.g., species name)
        max_length: Maximum length for the sanitized filename part

    Returns:
        Safe filename string, or 'unknown' if input is invalid

    Security:
        - Removes path traversal sequences (../, .., etc.)
        - Removes null bytes
        - Removes/replaces filesystem special characters
        - Prevents reserved OS filenames (CON, PRN, AUX, etc.)
        - Limits length to prevent filesystem issues
    """
    if not name or not isinstance(name, str):
        return "unknown"

    # Remove null bytes
    name = name.replace('\0', '')

    # Convert to lowercase for consistency
    name = name.lower()

    # Remove any path separators and traversal sequences
    # This catches: / \ .. ./ .\ and combinations
    name = name.replace('/', '_')
    name = name.replace('\\', '_')
    name = re.sub(r'\.\.+', '.', name)  # Replace multiple dots with single dot

    # Remove other filesystem special characters
    # Keep: letters, numbers, spaces, hyphens, underscores, dots
    name = re.sub(r'[^a-z0-9\s\-_.]', '_', name)

    # Replace spaces with underscores
    name = name.replace(' ', '_')

    # Remove leading/trailing dots and underscores (can cause issues)
    name = name.strip('._')

    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)

    # Check for reserved Windows filenames
    reserved_names = {
        'con', 'prn', 'aux', 'nul',
        'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9',
        'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
    }

    # Split by dots to check base name without extension
    parts = name.split('.')
    if parts and parts[0].lower() in reserved_names:
        name = f"bird_{name}"

    # Limit length
    if len(name) > max_length:
        name = name[:max_length].rstrip('._')

    # Final safety check - if nothing left, use default
    if not name or name in ['.', '..']:
        return "unknown"

    return name


class NotificationManager:
    """Manages all notification types"""

    def __init__(self, snapshots_dir: str = "snapshots", logs_dir: str = "logs",
                 cooldown: int = 30, max_snapshots: int = 100,
                 detector=None):
        """
        Initialize notification manager.

        Args:
            snapshots_dir: Directory to save bird snapshots
            logs_dir: Directory for log files
            cooldown: Seconds between notifications (to avoid spam)
            max_snapshots: Maximum number of snapshots to keep (default: 100)
            detector: BirdDetector instance for annotating snapshots
        """
        self.snapshots_dir = snapshots_dir
        self.logs_dir = logs_dir
        self.cooldown = cooldown
        self.max_snapshots = max_snapshots
        self.detector = detector

        self.last_notification_time = 0
        self.detection_log_file = os.path.join(logs_dir, "detections.csv")

        # Create directories with secure permissions (owner only - 700)
        os.makedirs(snapshots_dir, mode=0o700, exist_ok=True)
        os.makedirs(logs_dir, mode=0o700, exist_ok=True)

        # Ensure existing directories have secure permissions
        try:
            os.chmod(snapshots_dir, stat.S_IRWXU)  # 700
            os.chmod(logs_dir, stat.S_IRWXU)  # 700
        except Exception as e:
            print(f"Warning: Could not set secure permissions on directories: {e}")

        # Initialize detection log
        self._init_detection_log()

    def _init_detection_log(self):
        """Initialize the detection log CSV file with secure permissions"""
        if not os.path.exists(self.detection_log_file):
            # Create file with 600 permissions from the start (no race window)
            fd = os.open(
                self.detection_log_file,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR
            )
            with os.fdopen(fd, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'confidence',
                    'bbox_x1',
                    'bbox_y1',
                    'bbox_x2',
                    'bbox_y2',
                    'species_name',
                    'snapshot_file'
                ])
        else:
            # Fix permissions on existing file
            try:
                os.chmod(self.detection_log_file, stat.S_IRUSR | stat.S_IWUSR)
            except Exception as e:
                print(f"Warning: Could not set secure permissions on {self.detection_log_file}: {e}")

    def _is_cooldown_active(self) -> bool:
        """Check if we're still in cooldown period"""
        current_time = time.time()
        time_since_last = current_time - self.last_notification_time
        return time_since_last < self.cooldown

    def notify_bird_detected(self, image: Image.Image, detections: List,
                            save_snapshot: bool = True,
                            log_detection: bool = True) -> Optional[str]:
        """
        Send notifications when a bird is detected.

        Args:
            image: The camera image containing the bird
            detections: List of Detection objects (with optional species_name)
            save_snapshot: Whether to save the snapshot
            log_detection: Whether to log to CSV

        Returns:
            Path to saved snapshot (if saved), None otherwise
        """
        if not detections:
            return None

        # Check cooldown
        if self._is_cooldown_active():
            return None

        # Update last notification time
        self.last_notification_time = time.time()

        snapshot_path = None

        # Extract species names for notification
        species_names = [d.species_name for d in detections if d.species_name]

        # 1. Save snapshot
        if save_snapshot:
            snapshot_path = self._save_snapshot(image, detections, species_names)

        # 2. macOS notification (plays system sound via pync)
        self._send_macos_notification(len(detections), snapshot_path, species_names)

        # 3. Log detections
        if log_detection:
            self._log_detections(detections, snapshot_path)

        return snapshot_path

    def _save_snapshot(self, image: Image.Image, detections: List, species_names: List = None) -> str:
        """Save snapshot with bounding boxes and secure permissions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include species name in filename if available
        if species_names and len(species_names) > 0:
            # Use first species name with secure sanitization
            species = sanitize_filename(species_names[0])
            filename = f"bird_{species}_{timestamp}.jpg"
        else:
            filename = f"bird_{timestamp}.jpg"

        # Extra security: ensure filename doesn't contain path separators
        # This is defense-in-depth in case sanitize_filename has a bug
        filename = os.path.basename(filename)

        filepath = os.path.join(self.snapshots_dir, filename)

        # Draw bounding boxes using the shared detector instance
        if self.detector:
            annotated = self.detector.get_annotated_image(image, detections)
        else:
            annotated = image

        # Create clean image without metadata
        clean_image = Image.new(annotated.mode, annotated.size)
        clean_image.putdata(list(annotated.getdata()))

        # Save with no EXIF metadata
        clean_image.save(filepath, quality=95, optimize=True, exif=b"")

        # Set secure file permissions (owner read/write only - 600)
        try:
            os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)
        except Exception as e:
            print(f"Warning: Could not set secure permissions on {filepath}: {e}")

        # Clean up old snapshots
        self._cleanup_old_snapshots()

        return filepath

    def _send_macos_notification(self, bird_count: int, snapshot_path: Optional[str] = None, species_names: List = None):
        """Send macOS notification"""
        if not PYNC_AVAILABLE:
            # Fallback to terminal notification
            species_str = f" ({', '.join(species_names)})" if species_names else ""
            print(f"\n🐦 BIRD DETECTED! Found {bird_count} bird(s){species_str}")
            return

        title = "🐦 Bird Detected!"

        # Include species names in message if available
        if species_names and len(species_names) > 0:
            species_str = ', '.join(species_names)
            if bird_count == 1:
                message = f"Found {species_str} on your camera"
            else:
                message = f"Found {bird_count} birds on your camera ({species_str})"
        else:
            message = f"Found {bird_count} bird{'s' if bird_count > 1 else ''} on your camera"

        try:
            pync.notify(
                message,
                title=title,
                sound="Glass",  # macOS system sound
                appIcon=snapshot_path  # Show snapshot as icon (if available)
            )
        except Exception as e:
            print(f"Warning: Could not send notification: {e}")
            print(f"🐦 BIRD DETECTED! {message}")

    def _log_detections(self, detections: List, snapshot_path: Optional[str] = None):
        """Log detections to CSV file"""
        timestamp = datetime.now().isoformat()

        try:
            with open(self.detection_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    writer.writerow([
                        timestamp,
                        f"{detection.confidence:.3f}",
                        x1, y1, x2, y2,
                        detection.species_name or "",
                        snapshot_path or ""
                    ])
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

    def _cleanup_old_snapshots(self):
        """Remove old snapshots if we exceed max_snapshots"""
        try:
            snapshots_real = os.path.realpath(self.snapshots_dir)
            files = []
            for filename in os.listdir(self.snapshots_dir):
                if filename.startswith("bird_") and filename.endswith(".jpg"):
                    filepath = os.path.join(self.snapshots_dir, filename)

                    # Skip symlinks and files outside the snapshots directory
                    if os.path.islink(filepath):
                        continue
                    real_path = os.path.realpath(filepath)
                    if not real_path.startswith(snapshots_real + os.sep):
                        continue

                    files.append((os.path.getmtime(filepath), filepath))

            # Sort by modification time (oldest first)
            files.sort()

            # Remove oldest files if we exceed the limit
            while len(files) > self.max_snapshots:
                _, filepath = files.pop(0)
                if os.path.isfile(filepath) and not os.path.islink(filepath):
                    os.remove(filepath)

        except Exception as e:
            print(f"Warning: Could not cleanup snapshots: {e}")



