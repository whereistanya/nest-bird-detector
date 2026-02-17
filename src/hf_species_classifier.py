"""
HuggingFace Bird Species Classifier using EfficientNetB2
Identifies 525 bird species offline using pre-trained model from HuggingFace
Model: dennisjooo/Birds-Classifier-EfficientNetB2
Accuracy: 99.1% test accuracy on 525-species dataset

Supports optional location-based filtering using eBird regional species lists.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, List, TYPE_CHECKING
from transformers import AutoImageProcessor, AutoModelForImageClassification

if TYPE_CHECKING:
    from ebird_client import EBirdClient


class HFSpeciesClassifier:
    """
    HuggingFace bird species classifier using EfficientNetB2.
    Identifies 525 bird species completely offline with high accuracy (99.1%).
    """

    MODEL_NAME = "dennisjooo/Birds-Classifier-EfficientNetB2"

    # Expected model characteristics for integrity verification
    EXPECTED_NUM_LABELS = 525
    EXPECTED_MIN_PARAMS = 7_000_000   # ~7M parameters for EfficientNetB2
    EXPECTED_MAX_PARAMS = 12_000_000  # ~12M max

    def __init__(self, ebird_client: Optional['EBirdClient'] = None):
        """
        Initialize the classifier and load the pre-trained model.

        Args:
            ebird_client: Optional eBird client for location-based species filtering.
                         If provided, predictions will be filtered to only species
                         that occur in the configured region.
        """
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ebird_client = ebird_client
        self._load_model()

        if self.ebird_client:
            print(f"✓ Location filtering enabled ({len(self.ebird_client.common_names)} regional species)")

    def _load_model(self):
        """Load the pre-trained EfficientNetB2 model from HuggingFace."""
        print("🔄 Loading HuggingFace species classifier...")
        print(f"   Model: {self.MODEL_NAME}")
        print("   Architecture: EfficientNetB2")
        print("   Species: 525 bird species")
        print("   Accuracy: 99.1% test accuracy")
        print("   (Model will download on first run - ~30MB)")

        try:
            # Load image processor (for preprocessing)
            # NOTE: HuggingFace uses torch.load() internally which has pickle deserialization risks
            # Only use models from trusted/verified sources
            print("   Loading image processor...")
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)

            # Load pre-trained model
            print("   Loading model weights...")
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Verify model integrity
            self._verify_model_integrity()

            print("✓ Species classifier loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load species classifier: {e}")

    def _verify_model_integrity(self):
        """
        Verify the loaded model has expected architecture and parameters.
        This helps detect if the model has been tampered with or is corrupted.
        """
        print("   Verifying model integrity...")

        # Check number of labels (skip if expected is None for custom models)
        num_labels = self.model.config.num_labels
        if self.EXPECTED_NUM_LABELS is not None and num_labels != self.EXPECTED_NUM_LABELS:
            raise RuntimeError(
                f"Model has {num_labels} labels, expected {self.EXPECTED_NUM_LABELS}. "
                f"Model may be corrupted or tampered with."
            )

        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        if not (self.EXPECTED_MIN_PARAMS <= param_count <= self.EXPECTED_MAX_PARAMS):
            raise RuntimeError(
                f"Model has {param_count:,} parameters, expected "
                f"{self.EXPECTED_MIN_PARAMS:,} to {self.EXPECTED_MAX_PARAMS:,}. "
                f"Model may be corrupted or tampered with."
            )

        # Test with dummy input
        try:
            dummy_image = Image.new('RGB', (224, 224), color='red')
            inputs = self.processor(images=dummy_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Verify output shape (skip if expected is None for custom models)
            if self.EXPECTED_NUM_LABELS is not None and logits.shape[1] != self.EXPECTED_NUM_LABELS:
                raise RuntimeError(
                    f"Model output has {logits.shape[1]} classes, "
                    f"expected {self.EXPECTED_NUM_LABELS}"
                )
        except Exception as e:
            if "Model may be corrupted" in str(e) or "RuntimeError" in type(e).__name__:
                raise
            else:
                raise RuntimeError(f"Model failed integrity test: {e}")

        print(f"   ✓ Model integrity verified ({param_count:,} parameters, {num_labels} species)")

    def _validate_bbox(self, bbox: tuple, image_width: int, image_height: int):
        """
        Validate bounding box coordinates.

        Args:
            bbox: Tuple of (x1, y1, x2, y2)
            image_width: Width of the image
            image_height: Height of the image

        Raises:
            ValueError: If bounding box is invalid
        """
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError("Bounding box must be a tuple/list of 4 coordinates (x1, y1, x2, y2)")

        x1, y1, x2, y2 = bbox

        # Validate all coordinates are numeric
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            raise ValueError("All bounding box coordinates must be numeric")

        # Validate coordinates are non-negative
        if any(coord < 0 for coord in bbox):
            raise ValueError("Bounding box coordinates must be non-negative")

        # Validate x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bounding box dimensions: x1 must be < x2 and y1 must be < y2")

        # Validate within image bounds
        if x2 > image_width or y2 > image_height:
            raise ValueError(
                f"Bounding box ({x1}, {y1}, {x2}, {y2}) exceeds "
                f"image dimensions ({image_width}x{image_height})"
            )

    def _preprocess_image(self, image: Image.Image, bbox: Optional[tuple] = None) -> Dict:
        """
        Preprocess image for the model.

        Args:
            image: PIL Image
            bbox: Optional bounding box (x1, y1, x2, y2) to crop to bird region

        Returns:
            Preprocessed inputs ready for model

        Raises:
            ValueError: If image or bounding box is invalid
        """
        # Validate image
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image).__name__}")

        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError(f"Image must be RGB or RGBA mode, got {image.mode}")

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Validate and crop to bounding box if provided
        if bbox:
            self._validate_bbox(bbox, image.width, image.height)
            x1, y1, x2, y2 = bbox
            image = image.crop((x1, y1, x2, y2))

        # Use HuggingFace processor for preprocessing
        # This handles resizing, normalization, etc. automatically
        inputs = self.processor(images=image, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _validate_model_output(self, logits: torch.Tensor, probabilities: torch.Tensor,
                              predicted_idx: torch.Tensor, confidence: torch.Tensor) -> None:
        """
        Validate model output tensors for anomalies.

        Args:
            logits: Raw model outputs
            probabilities: Softmax probabilities
            predicted_idx: Index of predicted class
            confidence: Confidence score

        Raises:
            ValueError: If model outputs are invalid
        """
        # Check for NaN or infinity in logits
        if torch.isnan(logits).any():
            raise ValueError("Model logits contain NaN values")
        if torch.isinf(logits).any():
            raise ValueError("Model logits contain infinity values")

        # Check for NaN or infinity in probabilities
        if torch.isnan(probabilities).any():
            raise ValueError("Probabilities contain NaN values")
        if torch.isinf(probabilities).any():
            raise ValueError("Probabilities contain infinity values")

        # Validate probabilities sum to approximately 1.0 (within tolerance)
        prob_sum = probabilities.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Probabilities sum to {prob_sum}, expected ~1.0")

        # Validate confidence is in valid range [0, 1]
        conf_value = confidence.item()
        if not (0.0 <= conf_value <= 1.0):
            raise ValueError(f"Confidence {conf_value} outside valid range [0, 1]")

        # Validate predicted index is in valid range
        pred_idx = predicted_idx.item()
        num_labels = self.model.config.num_labels
        if not (0 <= pred_idx < num_labels):
            raise ValueError(
                f"Predicted index {pred_idx} outside valid range "
                f"[0, {num_labels})"
            )

    def _get_top_predictions(self, image: Image.Image, bbox: Optional[tuple] = None,
                            top_k: int = 10) -> List[Dict]:
        """
        Get top K species predictions from an image.

        Args:
            image: PIL Image containing a bird
            bbox: Optional bounding box (x1, y1, x2, y2) to focus on bird region
            top_k: Number of top predictions to return

        Returns:
            List of dictionaries with 'common_name' and 'score', sorted by confidence
        """
        if self.model is None or self.processor is None:
            return []

        try:
            # Preprocess image
            inputs = self._preprocess_image(image, bbox)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get all probabilities
            probabilities = F.softmax(logits, dim=1)

            # Get top K predictions
            num_labels = self.model.config.num_labels
            top_probs, top_indices = torch.topk(probabilities[0], k=min(top_k, num_labels))

            # Convert to list of predictions
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                species_name = self.model.config.id2label[idx.item()]
                confidence_score = prob.item()
                predictions.append({
                    'common_name': species_name,
                    'score': confidence_score
                })

            return predictions

        except Exception as e:
            print(f"⚠️  Failed to get top predictions: {e}")
            return []

    def identify_species(self, image: Image.Image, bbox: Optional[tuple] = None) -> Optional[Dict]:
        """
        Identify bird species from an image.

        If eBird client is configured, filters predictions to only species that
        occur in the configured region, significantly improving accuracy.

        Args:
            image: PIL Image containing a bird
            bbox: Optional bounding box (x1, y1, x2, y2) to focus on bird region

        Returns:
            Dictionary with 'common_name' and 'score' (confidence), or None if failed
        """
        if self.model is None or self.processor is None:
            print("⚠️  Model not loaded")
            return None

        try:
            # If eBird filtering is enabled, get top predictions and filter
            # Check more candidates (30) to increase chances of finding a regional match
            if self.ebird_client:
                predictions = self._get_top_predictions(image, bbox, top_k=30)

                if not predictions:
                    return None

                # Filter to only species in the region
                regional_predictions = [
                    p for p in predictions
                    if self.ebird_client.is_species_in_region(p['common_name'])
                ]

                if regional_predictions:
                    # Return highest-confidence regional species
                    return regional_predictions[0]
                else:
                    # No regional matches found - DON'T return impossible species
                    print(f"⚠️  No regional matches in top 30 predictions!")
                    print(f"   Top prediction was: {predictions[0]['common_name']} ({predictions[0]['score']:.2%})")
                    print(f"   This species does not occur in {self.ebird_client.region_code}")
                    print(f"   Returning None - cannot identify bird with regional constraints")
                    return None  # Don't show impossible species!

            # No eBird filtering - use original single-prediction logic
            # Preprocess image
            inputs = self._preprocess_image(image, bbox)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get predictions
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            # Validate outputs
            self._validate_model_output(logits, probabilities, predicted_idx, confidence)

            # Get species name from model's label mapping
            species_id = predicted_idx.item()
            species_name = self.model.config.id2label[species_id]
            confidence_score = confidence.item()

            return {
                'common_name': species_name,
                'score': confidence_score
            }

        except Exception as e:
            print(f"⚠️  Species identification failed: {e}")
            return None


