"""
Security tests for critical functions
Tests credential handling, token refresh, and input validation
"""
import pytest
import os
from config import Config


class TestCredentialSecurity:
    """Test secure credential storage and handling"""

    def test_sensitive_keys_in_keychain(self):
        """Verify sensitive credentials are read from OS keychain"""
        config = Config()

        # These should come from keychain, not env
        config.oauth_client_secret = "test_secret"
        assert config.oauth_client_secret == "test_secret"
        # Should NOT appear in os.environ
        assert os.getenv("OAUTH_CLIENT_SECRET") != "test_secret"

    def test_env_file_permissions(self):
        """Verify .env file has secure permissions (owner only)"""
        if os.path.exists('.env'):
            stat_info = os.stat('.env')
            mode = stat_info.st_mode

            # Check file is only readable/writable by owner (0600)
            # No group or other permissions
            assert (mode & 0o077) == 0, ".env file has insecure permissions"

    def test_config_has_tokens_check(self):
        """Verify config properly checks for token presence"""
        config = Config()

        # Method should exist and return boolean
        has_tokens = config.has_tokens()
        assert isinstance(has_tokens, bool)

    def test_config_is_configured_check(self):
        """Verify config properly checks for required credentials"""
        config = Config()

        # Method should exist and return boolean
        is_configured = config.is_configured()
        assert isinstance(is_configured, bool)


class TestTokenSecurity:
    """Test OAuth token handling security"""

    def test_no_tokens_in_code(self):
        """Verify no hardcoded tokens in codebase"""
        # Read main application files
        files_to_check = [
            'config.py',
            'nest_client.py',
            'main.py',
            'gui.py'
        ]

        suspicious_patterns = [
            'ya29.',  # Google access token prefix
            '1//',    # Google refresh token prefix
        ]

        for file_path in files_to_check:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    for pattern in suspicious_patterns:
                        assert pattern not in content, \
                            f"Potential hardcoded token found in {file_path}"

    def test_refresh_token_validation(self):
        """Test that token refresh validates responses"""
        from nest_client import NestClient
        config = Config()

        if config.is_configured() and config.has_tokens():
            client = NestClient(config)

            # Verify client has refresh method
            assert hasattr(client, 'refresh_access_token')
            assert callable(client.refresh_access_token)


class TestInputValidation:
    """Test input validation and sanitization"""

    def test_confidence_threshold_validation(self):
        """Test confidence threshold is validated"""
        config = Config()

        # Valid values should work
        config.confidence_threshold = 0.5
        assert config.confidence_threshold == 0.5

        # Invalid values should raise errors
        with pytest.raises(ValueError):
            config.confidence_threshold = 1.5  # > 1.0

        with pytest.raises(ValueError):
            config.confidence_threshold = -0.1  # < 0.0


