"""
Unit tests for bird detection functionality.
"""
import pytest
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from detector import BirdDetector, Detection


class TestBirdDetector:
    """Test BirdDetector functionality."""

    @pytest.fixture
    def mock_yolo(self):
        """Mock YOLO model to avoid loading weights in tests."""
        with patch('detector.YOLO') as mock:
            # Create a mock model instance
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    @pytest.fixture
    def detector(self, mock_yolo):
        """Create a BirdDetector instance with mocked YOLO."""
        return BirdDetector(confidence_threshold=0.5)

    def test_detector_initialization(self, mock_yolo):
        """Test that detector initializes with correct parameters."""
        detector = BirdDetector(confidence_threshold=0.7)
        assert detector.confidence_threshold == 0.7
        assert detector.model is not None
        # Verify YOLO was called with correct model name
        mock_yolo[0].assert_called_once_with("yolov8m.pt")

    def test_detector_default_confidence(self, mock_yolo):
        """Test that detector uses default confidence threshold."""
        detector = BirdDetector()
        assert detector.confidence_threshold == 0.5

    def test_detect_birds_returns_list(self, detector, sample_image, mock_yolo):
        """Test that detect_birds returns a list."""
        # Mock the YOLO results
        mock_result = Mock()
        mock_result.boxes = []  # Empty list of boxes
        mock_yolo[1].return_value = [mock_result]

        detections = detector.detect_birds(sample_image)
        assert isinstance(detections, list)

    def test_detect_birds_with_invalid_image(self, detector):
        """Test that detect_birds handles invalid input gracefully."""
        with pytest.raises((TypeError, AttributeError, RuntimeError)):
            detector.detect_birds(None)

        with pytest.raises((TypeError, AttributeError, RuntimeError)):
            detector.detect_birds("not an image")

    def test_detection_structure(self, detector, sample_image, mock_yolo):
        """Test that detections have the correct structure."""
        # Mock bird detection result
        mock_box = Mock()
        mock_box.cls = np.array([14])
        mock_box.conf = np.array([0.85])
        mock_box.xyxy = np.array([[100, 100, 200, 200]])

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_result.names = {14: "bird"}
        mock_yolo[1].return_value = [mock_result]

        detections = detector.detect_birds(sample_image)

        assert len(detections) > 0
        detection = detections[0]
        assert isinstance(detection, Detection)
        assert hasattr(detection, 'class_id')
        assert hasattr(detection, 'confidence')
        assert hasattr(detection, 'bbox')
        assert hasattr(detection, 'class_name')

    def test_confidence_threshold_filtering(self, sample_image, mock_yolo):
        """Test that confidence threshold filters detections."""
        # Mock detections with different confidences
        mock_box1 = Mock()
        mock_box1.cls = np.array([14])
        mock_box1.conf = np.array([0.95])
        mock_box1.xyxy = np.array([[100, 100, 200, 200]])

        mock_box2 = Mock()
        mock_box2.cls = np.array([14])
        mock_box2.conf = np.array([0.60])
        mock_box2.xyxy = np.array([[300, 100, 400, 200]])

        mock_box3 = Mock()
        mock_box3.cls = np.array([14])
        mock_box3.conf = np.array([0.25])
        mock_box3.xyxy = np.array([[500, 100, 600, 200]])

        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2, mock_box3]
        mock_result.names = {14: "bird"}
        mock_yolo[1].return_value = [mock_result]

        # High threshold should filter out more
        detector_high = BirdDetector(confidence_threshold=0.9)
        detections_high = detector_high.detect_birds(sample_image)
        assert len(detections_high) == 1  # Only 0.95

        # Low threshold should allow more
        detector_low = BirdDetector(confidence_threshold=0.3)
        detections_low = detector_low.detect_birds(sample_image)
        assert len(detections_low) == 2  # 0.95 and 0.60

    def test_get_annotated_image(self, detector, sample_image, mock_yolo):
        """Test that get_annotated_image returns valid image."""
        # Create sample detections
        detections = [
            Detection((100, 100, 200, 200), 0.85, 14, "bird"),
            Detection((300, 150, 400, 250), 0.72, 14, "bird")
        ]

        annotated = detector.get_annotated_image(sample_image, detections)

        assert isinstance(annotated, Image.Image)
        assert annotated.size == sample_image.size
        assert annotated.mode == sample_image.mode

    def test_get_annotated_image_no_detections(self, detector, sample_image):
        """Test that get_annotated_image works with empty detections."""
        annotated = detector.get_annotated_image(sample_image, [])

        assert isinstance(annotated, Image.Image)
        assert annotated.size == sample_image.size

    def test_bird_class_filtering(self, detector, sample_image, mock_yolo):
        """Test that only bird class (14) is detected."""
        # Mock mixed detections (bird and non-bird)
        mock_box1 = Mock()
        mock_box1.cls = np.array([14])
        mock_box1.conf = np.array([0.85])
        mock_box1.xyxy = np.array([[100, 100, 200, 200]])

        mock_box2 = Mock()
        mock_box2.cls = np.array([16])  # cat
        mock_box2.conf = np.array([0.90])
        mock_box2.xyxy = np.array([[300, 100, 400, 200]])

        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.names = {14: "bird", 16: "cat"}
        mock_yolo[1].return_value = [mock_result]

        detections = detector.detect_birds(sample_image)

        # Should only have one detection (the bird, not the cat)
        assert len(detections) == 1
        assert detections[0].class_id == 14
        assert detections[0].class_name == "bird"

    def test_bbox_coordinates_valid(self, detector, sample_image, mock_yolo):
        """Test that bounding box coordinates are valid."""
        width, height = sample_image.size

        # Mock detection within image bounds
        mock_box = Mock()
        mock_box.cls = np.array([14])
        mock_box.conf = np.array([0.85])
        mock_box.xyxy = np.array([[100, 100, 200, 200]])

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_result.names = {14: "bird"}
        mock_yolo[1].return_value = [mock_result]

        detections = detector.detect_birds(sample_image)

        for detection in detections:
            bbox = detection.bbox
            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox

            # Check coordinates are within image bounds
            assert 0 <= x1 < width
            assert 0 <= y1 < height
            assert 0 <= x2 <= width
            assert 0 <= y2 <= height

            # Check x2 > x1 and y2 > y1
            assert x2 > x1
            assert y2 > y1

    def test_confidence_values_valid(self, detector, sample_image, mock_yolo):
        """Test that confidence values are in valid range."""
        mock_box = Mock()
        mock_box.cls = np.array([14])
        mock_box.conf = np.array([0.85])
        mock_box.xyxy = np.array([[100, 100, 200, 200]])

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_result.names = {14: "bird"}
        mock_yolo[1].return_value = [mock_result]

        detections = detector.detect_birds(sample_image)

        for detection in detections:
            assert 0.0 <= detection.confidence <= 1.0
            assert detection.confidence >= detector.confidence_threshold


class TestDetectionDataClass:
    """Test Detection dataclass functionality."""

    def test_detection_creation(self):
        """Test creating a Detection object."""
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=14,
            class_name="bird"
        )

        assert detection.class_id == 14
        assert detection.confidence == 0.85
        assert detection.bbox == (100, 100, 200, 200)
        assert detection.class_name == "bird"



class TestDetectorImageSizes:
    """Test detector with various image sizes."""

    @pytest.fixture
    def mock_yolo(self):
        """Mock YOLO model."""
        with patch('detector.YOLO') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance

            # Setup default return value for inference
            mock_result = Mock()
            mock_result.boxes = []  # Empty list
            mock_instance.return_value = [mock_result]

            yield mock, mock_instance

    @pytest.fixture
    def detector(self, mock_yolo):
        return BirdDetector()

    def test_small_image(self, detector, mock_yolo):
        """Test detector with small image."""
        img = Image.new('RGB', (320, 240))
        detections = detector.detect_birds(img)
        assert isinstance(detections, list)

    def test_large_image(self, detector, mock_yolo):
        """Test detector with large image."""
        img = Image.new('RGB', (1920, 1080))
        detections = detector.detect_birds(img)
        assert isinstance(detections, list)

    def test_square_image(self, detector, mock_yolo):
        """Test detector with square image."""
        img = Image.new('RGB', (640, 640))
        detections = detector.detect_birds(img)
        assert isinstance(detections, list)

    def test_portrait_image(self, detector, mock_yolo):
        """Test detector with portrait orientation."""
        img = Image.new('RGB', (480, 640))
        detections = detector.detect_birds(img)
        assert isinstance(detections, list)
