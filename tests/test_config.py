"""
Unit tests for configuration management and validation.
"""
import pytest
import os
import tempfile
from config import Config


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_confidence_threshold_valid_values(self, temp_env_file):
        """Test that valid confidence_threshold values are accepted."""
        config = Config(temp_env_file)

        config.confidence_threshold = 0.0
        assert config.confidence_threshold == 0.0

        config.confidence_threshold = 0.5
        assert config.confidence_threshold == 0.5

        config.confidence_threshold = 1.0
        assert config.confidence_threshold == 1.0

    def test_confidence_threshold_out_of_range(self, temp_env_file):
        """Test that out-of-range confidence_threshold raises ValueError."""
        config = Config(temp_env_file)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            config.confidence_threshold = -0.1

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            config.confidence_threshold = 1.1



class TestConfigPersistence:
    """Test configuration persistence and file handling."""

    def test_config_read_write(self, temp_env_file):
        """Test that configuration values persist to file."""
        config = Config(temp_env_file)

        config.check_interval = 42
        config.confidence_threshold = 0.75

        # Create new config instance reading same file
        config2 = Config(temp_env_file)
        assert config2.check_interval == 42
        assert config2.confidence_threshold == 0.75

    def test_config_file_permissions(self, temp_env_file):
        """Test that config file has secure permissions."""
        config = Config(temp_env_file)
        config.check_interval = 10  # Trigger a write

        # Check file permissions (should be 600)
        stat_info = os.stat(temp_env_file)
        mode = stat_info.st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent.env")
            config = Config(config_path)

            # Should use defaults (or values from .env if it was loaded by dotenv)
            # The actual values might vary depending on test execution order
            assert isinstance(config.check_interval, int)
            assert config.check_interval > 0
            assert isinstance(config.confidence_threshold, float)
            assert 0.0 <= config.confidence_threshold <= 1.0


class TestConfigSensitiveData:
    """Test handling of sensitive configuration data."""

    def test_sensitive_data_in_keychain(self, temp_env_file, reset_keyring):
        """Test that sensitive data is stored in keychain, not .env."""
        config = Config(temp_env_file)

        # Set sensitive values
        config.oauth_client_secret = "test_secret"
        config.access_token = "test_access_token"
        config.refresh_token = "test_refresh_token"

        # Check they're in keychain
        assert reset_keyring.get_password("nest-bird-detector", "OAUTH_CLIENT_SECRET") == "test_secret"
        assert reset_keyring.get_password("nest-bird-detector", "ACCESS_TOKEN") == "test_access_token"
        assert reset_keyring.get_password("nest-bird-detector", "REFRESH_TOKEN") == "test_refresh_token"

        # Check they're NOT in .env file
        with open(temp_env_file, 'r') as f:
            env_content = f.read()
            assert "test_secret" not in env_content
            assert "test_access_token" not in env_content
            assert "test_refresh_token" not in env_content

    def test_sensitive_data_retrieval(self, temp_env_file, reset_keyring):
        """Test that sensitive data can be retrieved from keychain."""
        config = Config(temp_env_file)

        # Set and retrieve
        config.oauth_client_secret = "my_secret"
        assert config.oauth_client_secret == "my_secret"

        config.access_token = "my_token"
        assert config.access_token == "my_token"


class TestConfigStatus:
    """Test configuration status methods."""

    def test_is_configured_complete(self, temp_env_file, reset_keyring):
        """Test is_configured returns True when all required fields present."""
        config = Config(temp_env_file)

        config.oauth_client_id = "test_client_id"
        config.oauth_client_secret = "test_secret"
        config.project_id = "test_project"

        assert config.is_configured() is True

    def test_is_configured_incomplete(self, temp_env_file):
        """Test is_configured returns False when required fields missing."""
        config = Config(temp_env_file)

        # Missing oauth_client_secret
        assert config.is_configured() is False

    def test_has_tokens_complete(self, temp_env_file, reset_keyring):
        """Test has_tokens returns True when tokens present."""
        config = Config(temp_env_file)

        config.access_token = "test_access"
        config.refresh_token = "test_refresh"

        assert config.has_tokens() is True

    def test_has_tokens_incomplete(self, temp_env_file):
        """Test has_tokens returns False when tokens missing."""
        config = Config(temp_env_file)
        assert config.has_tokens() is False
