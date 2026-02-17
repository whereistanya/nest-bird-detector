"""
Google Nest API Client
Handles OAuth authentication and WebRTC streaming.
"""
from datetime import datetime, timedelta
from typing import Optional
import requests

from config import Config
from sdp_validator import validate_offer_sdp


class NestAPIError(Exception):
    """Raised when Nest API returns an error"""
    pass


class NestClient:
    """Client for interacting with Google Nest API"""

    OAUTH_TOKEN_URL = "https://www.googleapis.com/oauth2/v4/token"

    @staticmethod
    def exchange_authorization_code(config: Config, auth_code: str) -> bool:
        """
        Exchange an authorization code for access and refresh tokens.
        Saves tokens to the config's keychain on success.

        Returns True on success, False on failure.
        """
        try:
            response = requests.post(
                NestClient.OAUTH_TOKEN_URL,
                data={
                    "client_id": config.oauth_client_id,
                    "client_secret": config.oauth_client_secret,
                    "code": auth_code,
                    "grant_type": "authorization_code",
                    "redirect_uri": "https://www.google.com"
                }
            )

            if response.status_code == 200:
                tokens = response.json()
                access_token = tokens.get("access_token")
                refresh_token = tokens.get("refresh_token")

                if access_token and refresh_token:
                    config.access_token = access_token
                    config.refresh_token = refresh_token
                    return True

            print(f"Token exchange failed (HTTP {response.status_code})")
            return False

        except Exception as e:
            print(f"Error exchanging code: {e}")
            return False

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.token_expires_at: Optional[datetime] = None

        # Check if we have tokens
        if config.access_token:
            # Conservative estimate: assume 50 minutes remaining for existing tokens
            # (will be updated to actual value on next refresh)
            self.token_expires_at = datetime.now() + timedelta(minutes=50)

    def _ensure_valid_token(self):
        """Ensure we have a valid access token, refresh if needed"""
        if not self.config.access_token:
            raise NestAPIError("No access token available. Run setup_credentials.py first.")

        # Check if we don't know when token expires (safer to refresh)
        if not self.token_expires_at:
            print("🔄 Token expiration unknown, refreshing for safety...")
            self.refresh_access_token()
            return

        # Check if token is about to expire (within 5 minutes) or already expired
        time_until_expiry = self.token_expires_at - datetime.now()
        if time_until_expiry.total_seconds() < 300:  # Less than 5 minutes
            if time_until_expiry.total_seconds() <= 0:
                print("🔄 Access token expired, refreshing...")
            else:
                print(f"🔄 Access token expiring soon ({int(time_until_expiry.total_seconds())}s remaining), refreshing...")
            self.refresh_access_token()

    def refresh_access_token(self):
        """Refresh the access token using the refresh token"""
        if not self.config.refresh_token:
            raise NestAPIError("No refresh token available. Please re-authorize.")

        if not self.config.oauth_client_id or not self.config.oauth_client_secret:
            raise NestAPIError("OAuth credentials not configured.")

        try:
            print("🔄 Requesting new access token from Google OAuth...")
            response = self.session.post(
                "https://www.googleapis.com/oauth2/v4/token",
                data={
                    "client_id": self.config.oauth_client_id,
                    "client_secret": self.config.oauth_client_secret,
                    "refresh_token": self.config.refresh_token,
                    "grant_type": "refresh_token"
                }
            )

            if response.status_code == 200:
                tokens = response.json()
                new_access_token = tokens.get("access_token")
                expires_in = tokens.get("expires_in", 3600)  # Default to 1 hour if not provided

                if new_access_token:
                    self.config.access_token = new_access_token
                    # Set expiry based on actual expires_in from response
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    print(f"✓ Access token refreshed successfully (expires in {expires_in}s)")
                else:
                    raise NestAPIError("Invalid response from token refresh endpoint")
            else:
                # Extract only the error type, not the full response (which may contain tokens)
                error_type = "unknown"
                if response.headers.get('content-type', '').startswith('application/json'):
                    try:
                        error_type = response.json().get("error", "unknown")
                    except ValueError:
                        pass
                print(f"❌ Token refresh failed with status {response.status_code}")
                raise NestAPIError(f"Token refresh failed: HTTP {response.status_code} ({error_type})")

        except NestAPIError:
            raise
        except Exception as e:
            print(f"❌ Unexpected error during token refresh: {type(e).__name__}")
            raise NestAPIError(f"Failed to refresh token: {type(e).__name__}")

    def generate_webrtc_stream(self, device_id: Optional[str] = None, offer_sdp: Optional[str] = None) -> tuple:
        """
        Generate a WebRTC stream URL and SDP offer/answer.
        Returns (answer_sdp, media_session_id)

        Args:
            device_id: Nest device ID (optional, uses configured device if not provided)
            offer_sdp: SDP offer MUST be provided from WebRTC library (aiortc).
                      NEVER use hardcoded/default SDP - must have proper ICE credentials.

        Raises:
            NestAPIError: If no offer_sdp provided or if SDP validation fails

        Security:
            - Validates offer SDP for proper ICE credentials and DTLS fingerprints
            - Rejects weak/hardcoded credentials
            - Ensures encryption is properly specified
        """
        if device_id is None:
            device_id = self.config.device_id

        if not device_id:
            raise NestAPIError("No device ID specified")

        self._ensure_valid_token()

        # SECURITY: Require proper SDP from WebRTC library
        # Never use hardcoded/default SDP with weak credentials
        if offer_sdp is None:
            raise NestAPIError(
                "No SDP offer provided. Must be generated by WebRTC library with "
                "proper ICE credentials and DTLS fingerprints. Never use hardcoded SDP."
            )

        # Validate offer SDP has proper security parameters
        is_valid, error = validate_offer_sdp(offer_sdp)
        if not is_valid:
            raise NestAPIError(f"Invalid SDP offer: {error}")

        try:
            for attempt in range(2):
                response = self.session.post(
                    f"https://smartdevicemanagement.googleapis.com/v1/{device_id}:executeCommand",
                    headers={
                        "Authorization": f"Bearer {self.config.access_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "command": "sdm.devices.commands.CameraLiveStream.GenerateWebRtcStream",
                        "params": {
                            "offerSdp": offer_sdp
                        }
                    }
                )

                if response.status_code == 200:
                    result = response.json().get("results", {})
                    answer_sdp = result.get("answerSdp")
                    media_session_id = result.get("mediaSessionId")

                    if not answer_sdp:
                        raise NestAPIError("No answer SDP in response")

                    return (answer_sdp, media_session_id)

                elif response.status_code == 401 and attempt == 0:
                    print("🔄 Token expired, refreshing and retrying...")
                    self.refresh_access_token()
                    continue
                else:
                    break

            raise NestAPIError(f"Failed to generate WebRTC stream: HTTP {response.status_code}")

        except requests.RequestException as e:
            raise NestAPIError(f"Request failed: {e}")


