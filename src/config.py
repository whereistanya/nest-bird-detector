"""
Configuration Management
Loads settings from .env file and provides configuration access.
Sensitive credentials (tokens, secrets) stored securely in OS keychain.
"""
import os
import stat
from dotenv import load_dotenv
from typing import Optional
import keyring


class Config:
    """Application configuration with secure credential storage"""

    KEYRING_SERVICE = "nest-bird-detector"

    # Keys that belong in keychain but may exist in old .env files
    _KEYCHAIN_KEYS = ["OAUTH_CLIENT_SECRET", "ACCESS_TOKEN", "REFRESH_TOKEN", "DEVICE_ID", "EBIRD_API_KEY"]

    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        if os.path.exists(env_file):
            load_dotenv(env_file)
            try:
                os.chmod(env_file, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            self._migrate_to_keychain()

    def _get_env(self, key: str, default=None) -> Optional[str]:
        return os.getenv(key, default)

    def _set_env(self, key: str, value: str):
        """Set a value in .env file"""
        os.environ[key] = value
        lines = []
        updated = False

        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break

        if not updated:
            lines.append(f"{key}={value}\n")

        with open(self.env_file, 'w') as f:
            f.writelines(lines)
        os.chmod(self.env_file, stat.S_IRUSR | stat.S_IWUSR)

    def _migrate_to_keychain(self):
        """Move any sensitive values from .env to keychain (one-time migration)."""
        migrated = []
        for key in self._KEYCHAIN_KEYS:
            value = os.getenv(key)
            if value and value.strip() and not self._get_keychain(key):
                try:
                    self._set_keychain(key, value)
                    migrated.append(key)
                except Exception:
                    pass
        if migrated:
            # Remove migrated keys from .env
            if os.path.exists(self.env_file):
                with open(self.env_file, 'r') as f:
                    lines = [l for l in f.readlines()
                             if not any(l.startswith(f"{k}=") for k in migrated)]
                with open(self.env_file, 'w') as f:
                    f.writelines(lines)
                os.chmod(self.env_file, stat.S_IRUSR | stat.S_IWUSR)
            print(f"✓ Migrated {len(migrated)} credential(s) to keychain")

    def _get_keychain(self, key: str) -> Optional[str]:
        try:
            return keyring.get_password(self.KEYRING_SERVICE, key)
        except Exception:
            return None

    def _set_keychain(self, key: str, value: str):
        keyring.set_password(self.KEYRING_SERVICE, key, value)

    # --- Credentials (keychain) ---

    @property
    def oauth_client_id(self) -> Optional[str]:
        return self._get_env("OAUTH_CLIENT_ID")

    @oauth_client_id.setter
    def oauth_client_id(self, value: str):
        self._set_env("OAUTH_CLIENT_ID", value)

    @property
    def oauth_client_secret(self) -> Optional[str]:
        return self._get_keychain("OAUTH_CLIENT_SECRET")

    @oauth_client_secret.setter
    def oauth_client_secret(self, value: str):
        self._set_keychain("OAUTH_CLIENT_SECRET", value)

    @property
    def project_id(self) -> Optional[str]:
        return self._get_env("PROJECT_ID")

    @project_id.setter
    def project_id(self, value: str):
        self._set_env("PROJECT_ID", value)

    @property
    def access_token(self) -> Optional[str]:
        return self._get_keychain("ACCESS_TOKEN")

    @access_token.setter
    def access_token(self, value: str):
        self._set_keychain("ACCESS_TOKEN", value)

    @property
    def refresh_token(self) -> Optional[str]:
        return self._get_keychain("REFRESH_TOKEN")

    @refresh_token.setter
    def refresh_token(self, value: str):
        self._set_keychain("REFRESH_TOKEN", value)

    @property
    def device_id(self) -> Optional[str]:
        return self._get_keychain("DEVICE_ID")

    @device_id.setter
    def device_id(self, value: str):
        self._set_keychain("DEVICE_ID", value)

    # --- Application settings (.env) ---

    @property
    def check_interval(self) -> int:
        try:
            return int(self._get_env("CHECK_INTERVAL", "5"))
        except ValueError:
            return 5

    @check_interval.setter
    def check_interval(self, value: int):
        self._set_env("CHECK_INTERVAL", str(value))

    @property
    def confidence_threshold(self) -> float:
        try:
            return float(self._get_env("CONFIDENCE_THRESHOLD", "0.5"))
        except ValueError:
            return 0.5

    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {value}")
        self._set_env("CONFIDENCE_THRESHOLD", str(value))

    # --- eBird (optional) ---

    @property
    def ebird_api_key(self) -> Optional[str]:
        key = self._get_keychain("EBIRD_API_KEY")
        if key and key != "YOUR_EBIRD_API_KEY_HERE":
            return key
        return None

    @ebird_api_key.setter
    def ebird_api_key(self, value: str):
        self._set_keychain("EBIRD_API_KEY", value)

    @property
    def ebird_region_code(self) -> Optional[str]:
        return self._get_env("EBIRD_REGION_CODE")

    # --- Status checks ---

    def is_configured(self) -> bool:
        return all([self.oauth_client_id, self.oauth_client_secret, self.project_id])

    def has_tokens(self) -> bool:
        return all([self.access_token, self.refresh_token])
