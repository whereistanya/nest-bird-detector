#!/usr/bin/env python3
"""
Manages bird identification corrections for feedback loop.
Stores user corrections and validates species names.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Maximum length for species names
MAX_SPECIES_NAME_LENGTH = 100

# Characters that could trigger CSV formula injection when opened in spreadsheets
_CSV_INJECTION_CHARS = ('=', '+', '-', '@', '|', '%')

class CorrectionManager:
    """Manages user corrections to bird identifications"""

    def __init__(self, corrections_file: str = "bird_corrections.json"):
        self.corrections_file = Path(corrections_file)
        self.corrections = self._load_corrections()

        # Valid species names (add more as you identify them)
        self.valid_species = {
            "House Finch",
            "Mourning Dove",
            "Northern Flicker",
            "Common Poorwill",
            "American Robin",
            "Lesser Goldfinch",
            "White-winged Dove",
            "Black-chinned Hummingbird",
            "Pyrrhuloxia",
            "Unknown",  # For birds you can't identify
            "Not a Bird"  # For false positives
        }

    def _load_corrections(self) -> Dict:
        """Load existing corrections from file"""
        if self.corrections_file.exists():
            with open(self.corrections_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_corrections(self):
        """Save corrections to file"""
        with open(self.corrections_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)

    @staticmethod
    def _sanitize_species_input(species: str) -> tuple:
        """
        Validate and sanitize a species name input.

        Returns:
            (is_valid, error_or_cleaned_name)
        """
        if not species or not species.strip():
            return False, "Species name cannot be empty"

        species = species.strip()

        if len(species) > MAX_SPECIES_NAME_LENGTH:
            return False, f"Species name too long (max {MAX_SPECIES_NAME_LENGTH} characters)"

        # Reject control characters (including null bytes)
        if any(ord(c) < 32 for c in species):
            return False, "Species name contains invalid characters"

        # Reject CSV injection characters at start of string
        if species.startswith(_CSV_INJECTION_CHARS):
            return False, "Species name cannot start with special characters"

        # Allow only letters, numbers, spaces, hyphens, apostrophes, periods, parens
        if not re.match(r"^[A-Za-z0-9\s\-'.()]+$", species):
            return False, "Species name contains invalid characters"

        return True, species

    def add_correction(self, image_path: str, original_prediction: str,
                      corrected_species: str, confidence: float = 0.0) -> bool:
        """
        Add a correction for a misidentified bird.

        Args:
            image_path: Path to the image file
            original_prediction: What the model predicted
            corrected_species: The correct species name
            confidence: Original confidence score

        Returns:
            True if correction was added, False if validation failed
        """
        is_valid, result = self._sanitize_species_input(corrected_species)
        if not is_valid:
            return False

        corrected_species = result

        # Check for case-insensitive match with existing species
        normalized_species = corrected_species
        for valid_name in self.valid_species:
            if valid_name.lower() == corrected_species.lower():
                normalized_species = valid_name
                break
        else:
            # New species - add it to valid species list
            # Capitalize first letter of each word
            normalized_species = ' '.join(word.capitalize() for word in corrected_species.split())
            self.valid_species.add(normalized_species)

        # Store correction
        self.corrections[image_path] = {
            'original_prediction': original_prediction,
            'corrected_species': normalized_species,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'status': 'corrected'
        }

        self._save_corrections()
        return True

    def exclude_from_training(self, image_path: str, reason: str = "User excluded") -> bool:
        """
        Mark an image to be excluded from training.

        Args:
            image_path: Path to the image file
            reason: Reason for exclusion

        Returns:
            True if excluded successfully
        """
        self.corrections[image_path] = {
            'original_prediction': self.corrections.get(image_path, {}).get('original_prediction', 'Unknown'),
            'corrected_species': 'Excluded',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'status': 'excluded',
            'reason': reason
        }

        self._save_corrections()
        return True

    def get_correction(self, image_path: str) -> Optional[Dict]:
        """Get correction for an image if it exists"""
        return self.corrections.get(image_path)

    def get_corrections_for_training(self) -> List[Dict]:
        """
        Get corrections formatted for training pipeline.

        Returns:
            List of dicts with 'image_path' and 'species'
        """
        training_data = []
        for img_path, correction in self.corrections.items():
            # Skip excluded images
            if correction.get('status') == 'excluded':
                continue

            # Only include valid corrections (not "Not a Bird")
            if correction['corrected_species'] not in ['Not a Bird', 'Unknown', 'Excluded']:
                training_data.append({
                    'image_path': img_path,
                    'species': correction['corrected_species'],
                    'quality': 'good',
                    'source': 'user_correction'
                })

        return training_data

    def validate_species_name(self, species: str) -> tuple:
        """
        Validate a species name.

        Returns:
            (is_valid, normalized_name or error_message)
        """
        is_valid, result = self._sanitize_species_input(species)
        if not is_valid:
            return False, result

        species = result

        # Check if it matches existing species (case-insensitive)
        for valid_name in self.valid_species:
            if valid_name.lower() == species.lower():
                return True, valid_name

        # New species - normalize capitalization
        normalized = ' '.join(word.capitalize() for word in species.split())

        # Check for close matches to suggest
        suggestions = [s for s in self.valid_species
                      if species.lower() in s.lower() or s.lower() in species.lower()]

        if suggestions:
            return True, f"{normalized} (new species - similar to: {', '.join(suggestions[:2])})"
        else:
            return True, f"{normalized} (new species - will be added to list)"

    def get_valid_species(self) -> List[str]:
        """Get list of valid species names"""
        return sorted(self.valid_species)

    def get_stats(self) -> Dict:
        """Get statistics about corrections"""
        species_counts = {}
        for correction in self.corrections.values():
            species = correction['corrected_species']
            species_counts[species] = species_counts.get(species, 0) + 1

        return {
            'total_corrections': len(self.corrections),
            'species_distribution': species_counts,
            'valid_for_training': len(self.get_corrections_for_training())
        }

