#!/usr/bin/env python3
"""
Nest Bird Detector
Main entry point for the application.
"""
import sys
import os
import webbrowser

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from nest_client import NestClient


def reauthorize(config):
    """Interactive OAuth re-authorization when tokens are missing or expired.

    Opens the user's browser to the Google authorization page, then
    exchanges the resulting authorization code for new tokens.
    Returns True on success, False on failure or cancellation.
    """
    auth_url = (
        f"https://nestservices.google.com/partnerconnections/{config.project_id}/auth?"
        f"redirect_uri=https://www.google.com&"
        f"access_type=offline&"
        f"prompt=consent&"
        f"client_id={config.oauth_client_id}&"
        f"response_type=code&"
        f"scope=https://www.googleapis.com/auth/sdm.service"
    )

    print("Opening authorization page in your browser...")
    print()
    print("If the browser doesn't open, copy this URL:")
    print(f"  {auth_url}")
    print()
    webbrowser.open(auth_url)

    print("After authorizing:")
    print("  1. You'll be redirected to google.com")
    print("  2. Copy everything after 'code=' in the URL bar")
    print("     (the code ends at '&scope' or the end of the URL)")
    print()

    auth_code = input("Paste the authorization code here (or press Enter to cancel): ").strip()
    if not auth_code:
        return False

    print()
    print("Exchanging authorization code for tokens...")

    if NestClient.exchange_authorization_code(config, auth_code):
        print("New tokens saved to keychain.")
        return True
    return False


def check_connection_health():
    """Verify Nest API connection before starting GUI.

    If tokens are missing or expired, offers interactive re-authorization.
    Returns True if the connection is healthy and ready to use.
    """
    print("Checking Nest API connection...")
    print()

    config = Config()

    # Check basic configuration
    if not config.is_configured():
        print("Nest API credentials not configured!")
        print()
        print("Please run the setup first:")
        print("  python3 tools/setup_credentials.py")
        print()
        return False

    # If no tokens, offer to authorize
    if not config.has_tokens():
        print("OAuth tokens not found. Authorization required.")
        print()
        if not reauthorize(config):
            return False
        print()

    # Test token refresh (this validates the refresh token)
    try:
        client = NestClient(config)
        print("  Testing token refresh...")
        client.refresh_access_token()
        print("  Nest API connection verified.")
        print()
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        print()
        print("Your refresh token may have expired. Re-authorizing...")
        print()
        if not reauthorize(config):
            return False

        # Verify the new tokens work
        try:
            client = NestClient(config)
            client.refresh_access_token()
            print("  Nest API connection verified.")
            print()
            return True
        except Exception as e2:
            print(f"Still failing after re-authorization: {e2}")
            return False


def main():
    """Main entry point"""
    print("=" * 70)
    print("  Nest Bird Detector")
    print("=" * 70)
    print()

    # Check if configuration exists
    if not os.path.exists(".env"):
        print("Configuration file not found!")
        print()
        print("Please run the setup first:")
        print("  1. cp .env.example .env")
        print("  2. python3 tools/setup_credentials.py")
        print()
        sys.exit(1)

    # Health check: verify connection before starting GUI
    if not check_connection_health():
        sys.exit(1)

    # Launch GUI
    print("Starting GUI application...")
    print()

    import logging
    from PIL import Image
    from PyQt5.QtWidgets import QApplication
    from logging_config import setup_logging
    from gui import MainWindow

    # Protect against decompression bombs (limit to ~8K resolution with margin)
    Image.MAX_IMAGE_PIXELS = 178_956_970

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Nest Bird Detector application")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    logger.info("Application window displayed, entering event loop")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
