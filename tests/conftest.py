"""
Pytest configuration and shared fixtures for Nest Bird Detector tests.
"""
import sys
from pathlib import Path

# Add src/ to path so tests can import the application modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import tempfile
import os
from PIL import Image
import numpy as np


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("OAUTH_CLIENT_ID=test_client_id\n")
        f.write("PROJECT_ID=test_project\n")
        f.write("DEVICE_ID=test_device\n")
        f.write("CHECK_INTERVAL=10\n")
        f.write("CONFIDENCE_THRESHOLD=0.6\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 640x480 RGB image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    return img


@pytest.fixture
def sample_image_with_bird():
    """Create a sample image that might contain a bird-like shape."""
    # Create image with a brownish ellipse (bird-like)
    img = Image.new('RGB', (640, 480), color=(135, 206, 235))  # Sky blue background
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw bird-like ellipse
    draw.ellipse([250, 200, 390, 280], fill=(139, 90, 43), outline=(0, 0, 0))  # Brown ellipse
    return img


@pytest.fixture
def mock_nest_response():
    """Mock response from Nest API."""
    return {
        "name": "enterprises/test/devices/test_device",
        "type": "sdm.devices.types.CAMERA",
        "traits": {
            "sdm.devices.traits.Info": {
                "customName": "Test Camera"
            },
            "sdm.devices.traits.CameraLiveStream": {
                "maxVideoResolution": {
                    "width": 1920,
                    "height": 1080
                }
            }
        }
    }


@pytest.fixture
def mock_token_response():
    """Mock OAuth token refresh response."""
    return {
        "access_token": "new_test_access_token",
        "expires_in": 3600,
        "scope": "https://www.googleapis.com/auth/sdm.service",
        "token_type": "Bearer"
    }


@pytest.fixture
def temp_snapshots_dir():
    """Create a temporary directory for snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_detections():
    """Sample bird detections for testing."""
    return [
        {
            'class': 14,  # Bird class in COCO dataset
            'confidence': 0.85,
            'bbox': [100, 100, 200, 200],
            'name': 'bird'
        },
        {
            'class': 14,
            'confidence': 0.72,
            'bbox': [300, 150, 400, 250],
            'name': 'bird'
        }
    ]


@pytest.fixture(autouse=True)
def reset_keyring(monkeypatch):
    """Mock keyring to avoid touching real system keychain in tests."""
    class MockKeyring:
        _storage = {}

        @classmethod
        def get_password(cls, service, key):
            return cls._storage.get(f"{service}:{key}")

        @classmethod
        def set_password(cls, service, key, value):
            cls._storage[f"{service}:{key}"] = value

        @classmethod
        def delete_password(cls, service, key):
            cls._storage.pop(f"{service}:{key}", None)

        @classmethod
        def reset(cls):
            cls._storage.clear()

    # Patch keyring module
    import sys
    if 'keyring' in sys.modules:
        import keyring as kr
        monkeypatch.setattr(kr, 'get_password', MockKeyring.get_password)
        monkeypatch.setattr(kr, 'set_password', MockKeyring.set_password)
        monkeypatch.setattr(kr, 'delete_password', MockKeyring.delete_password)

    MockKeyring.reset()
    yield MockKeyring
    MockKeyring.reset()
