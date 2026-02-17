#!/usr/bin/env python3
"""
Missed Bird Logger
Logs frames where birds were present but not detected by YOLO.
These can later be annotated and used to retrain the detection model.
"""
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Optional


class MissedBirdLogger:
    """Logs frames where birds were missed by the detector"""

    def __init__(self, log_file: str = "missed_birds.json",
                 images_dir: str = "missed_birds"):
        self.log_file = Path(log_file)
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        self.missed_birds = self._load_log()

    def _load_log(self) -> dict:
        """Load existing missed bird log"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_log(self):
        """Save missed bird log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.missed_birds, f, indent=2)

    def log_missed_bird(self, image: Image.Image, species: Optional[str] = None,
                       notes: str = "") -> str:
        """
        Log a frame where a bird was present but not detected.

        Args:
            image: PIL Image of the frame
            species: Optional species name if known
            notes: Optional notes about the bird

        Returns:
            Path to saved image
        """
        # Generate filename with timestamp
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        image_filename = f"missed_bird_{timestamp_str}.jpg"
        image_path = self.images_dir / image_filename

        # Save image
        image.save(image_path, quality=95)

        # Log entry
        entry = {
            'timestamp': timestamp.isoformat(),
            'image_path': str(image_path),
            'species': species or "Unknown",
            'notes': notes,
            'annotated': False,  # Flag for whether this has been annotated
            'status': 'pending'  # pending, annotated, or trained
        }

        self.missed_birds[str(image_path)] = entry
        self._save_log()

        print(f"✓ Logged missed bird: {image_path}")
        return str(image_path)

    def get_stats(self) -> dict:
        """Get statistics about missed birds"""
        total = len(self.missed_birds)
        annotated = sum(1 for e in self.missed_birds.values() if e.get('annotated', False))
        pending = total - annotated

        species_counts = {}
        for entry in self.missed_birds.values():
            species = entry.get('species', 'Unknown')
            species_counts[species] = species_counts.get(species, 0) + 1

        return {
            'total': total,
            'annotated': annotated,
            'pending': pending,
            'species_distribution': species_counts
        }

