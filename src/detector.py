"""
Bird Detection using YOLOv8
Detects birds in images using a pre-trained YOLO model.
"""
from PIL import Image
from typing import List, Optional
from ultralytics import YOLO


class Detection:
    """Represents a single bird detection"""

    def __init__(self, bbox: tuple, confidence: float, class_id: int, class_name: str,
                 species_name: Optional[str] = None):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.species_name = species_name  # Optional species identification from local classifier

    def __repr__(self):
        species_str = f", species={self.species_name}" if self.species_name else ""
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}{species_str}, bbox={self.bbox})"



class BirdDetector:
    """YOLOv8-based bird detector"""

    # COCO dataset class ID for birds
    BIRD_CLASS_ID = 14

    def __init__(self, model_name: str = "yolov8m.pt", confidence_threshold: float = 0.5):
        """
        Initialize the bird detector.

        Args:
            model_name: YOLOv8 model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
                       Default is yolov8m.pt (medium) for better accuracy on difficult detections
            confidence_threshold: Minimum confidence score (0.0 - 1.0)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model (downloads if necessary)"""
        print(f"🔄 Loading YOLOv8 model: {self.model_name}")
        print("   (This will download the model on first run - should only happen once)")

        try:
            # PyTorch 2.6+ changed default for weights_only to True
            # YOLO models require weights_only=False (safe for official Ultralytics models)
            import torch
            original_load = torch.load

            def patched_load(*args, **kwargs):
                # Force weights_only=False for YOLO model loading
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            # Temporarily patch torch.load
            torch.load = patched_load
            try:
                self.model = YOLO(self.model_name)
            finally:
                # Restore original torch.load
                torch.load = original_load

            print(f"✓ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect_birds(self, image: Image.Image) -> List[Detection]:
        """
        Detect birds in an image.

        Args:
            image: PIL Image to analyze

        Returns:
            List of Detection objects for birds found
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Run inference
        results = self.model(image, verbose=False)

        # Extract bird detections
        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Check if it's a bird and meets confidence threshold
                if class_id == self.BIRD_CLASS_ID and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Get class name
                    class_name = result.names[class_id]

                    detections.append(Detection(bbox, confidence, class_id, class_name))

        return detections

    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def get_annotated_image(self, image: Image.Image, detections: List[Detection]) -> Image.Image:
        """
        Draw bounding boxes on the image for visualization.

        Args:
            image: Original PIL Image
            detections: List of Detection objects

        Returns:
            New PIL Image with bounding boxes drawn
        """
        from PIL import ImageDraw, ImageFont

        # Create a copy to draw on
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Draw bounding box (green)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Draw label with confidence at top
            label = f"Bird: {detection.confidence:.2f}"

            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except (IOError, OSError):
                font = ImageFont.load_default()

            # Draw confidence label at top
            text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
            draw.rectangle(text_bbox, fill="green")
            draw.text((x1, y1 - 20), label, fill="white", font=font)

            # Draw species name at bottom-right if available
            if detection.species_name:
                species_label = detection.species_name

                # Calculate position at bottom-right, slightly inside the box to avoid obscuring bird
                species_bbox = draw.textbbox((0, 0), species_label, font=font)
                label_width = species_bbox[2] - species_bbox[0]
                label_height = species_bbox[3] - species_bbox[1]

                # Position: right-aligned at bottom, with small padding
                padding = 5
                species_x = x2 - label_width - padding
                species_y = y2 - label_height - padding

                # Ensure label stays within image bounds
                species_x = max(x1 + padding, species_x)
                species_y = max(y1 + padding, species_y)

                # Draw species text background
                species_text_bbox = draw.textbbox((species_x, species_y), species_label, font=font)
                draw.rectangle(species_text_bbox, fill="green")
                draw.text((species_x, species_y), species_label, fill="white", font=font)

        return annotated
