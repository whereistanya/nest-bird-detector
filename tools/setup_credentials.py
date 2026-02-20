"""
Interactive Nest API Credentials Setup Guide
Helps users configure Google Nest API access step-by-step.
"""
import sys
import os
import webbrowser
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import Config
from nest_client import NestClient


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(number, title):
    """Print a step header"""
    print(f"\n📋 Step {number}: {title}")
    print("-" * 70)


def wait_for_user():
    """Wait for user to press Enter"""
    input("\nPress Enter to continue...")


def main():
    print_header("🐦 Nest Bird Detector - API Setup Guide")

    config = Config()

    print("This guide will help you set up Google Nest API credentials.")
    print("The process takes about 20-30 minutes and costs $5 (one-time fee).")
    print("\nYou will need:")
    print("  • A Google account")
    print("  • A Nest camera")
    print("  • $5 for the Device Access registration")
    print("  • A credit card for Google Cloud Platform")

    response = input("\nReady to start? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return

    # Step 1: Register for Device Access
    print_step(1, "Register for Google Device Access")
    print("You need to register for the Device Access program ($5 fee).")
    print("\nThis will open: https://console.nest.google.com/device-access/")
    print("\nInstructions:")
    print("  1. Click 'Go to Device Access Console'")
    print("  2. Accept the Terms of Service")
    print("  3. Pay the $5 registration fee")
    print("  4. Click 'Continue'")

    response = input("\nOpen this page now? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open("https://console.nest.google.com/device-access/")

    wait_for_user()

    # Step 2: Create Google Cloud Project
    print_step(2, "Create Google Cloud Project")
    print("You need a Google Cloud project with the Smart Device Management API enabled.")
    print("\nThis will open: https://console.cloud.google.com/")
    print("\nInstructions:")
    print("  1. Create a new project (or select existing)")
    print("  2. Note your Project ID")
    print("  3. Enable 'Smart Device Management API'")
    print("  4. Go to: APIs & Services > Library")
    print("  5. Search for 'Smart Device Management API'")
    print("  6. Click 'Enable'")

    response = input("\nOpen Google Cloud Console? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open("https://console.cloud.google.com/")

    wait_for_user()

    # Step 3: Create OAuth 2.0 Credentials
    print_step(3, "Create OAuth 2.0 Credentials")
    print("Create OAuth credentials for your app.")
    print("\nThis will open: https://console.cloud.google.com/apis/credentials")
    print("\nInstructions:")
    print("  1. Click 'Create Credentials' > 'OAuth 2.0 Client ID'")
    print("  2. Configure OAuth consent screen if prompted:")
    print("     - User type: External")
    print("     - App name: Nest Bird Detector")
    print("     - User support email: (your email)")
    print("     - Developer contact: (your email)")
    print("  3. Application type: Web application")
    print("  4. Name: Nest Bird Detector")
    print("  5. Authorized redirect URIs: https://www.google.com")
    print("  6. Click 'Create'")
    print("  7. Copy the Client ID and Client Secret")

    response = input("\nOpen credentials page? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open("https://console.cloud.google.com/apis/credentials")

    print("\n")
    client_id = input("Enter your OAuth Client ID: ").strip()
    client_secret = input("Enter your OAuth Client Secret: ").strip()

    if client_id and client_secret:
        config.oauth_client_id = client_id
        config.oauth_client_secret = client_secret
        print("✓ OAuth credentials saved (secret stored in keychain)")

    wait_for_user()

    # Step 4: Create Device Access Project
    print_step(4, "Create Device Access Project")
    print("Create a Device Access project to link your OAuth credentials.")
    print("\nThis will open: https://console.nest.google.com/device-access/project-list")
    print("\nInstructions:")
    print("  1. Click 'Create project'")
    print("  2. Enter project name: Nest Bird Detector")
    print(f"  3. Paste your OAuth 2.0 Client ID: {client_id}")
    print("  4. Click 'Next' and 'Create project'")
    print("  5. Copy the Project ID (looks like a UUID)")

    response = input("\nOpen Device Access Console? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open("https://console.nest.google.com/device-access/project-list")

    print("\n")
    project_id = input("Enter your Device Access Project ID: ").strip()

    if project_id:
        config.project_id = project_id
        print("✓ Project ID saved")

    wait_for_user()

    # Step 5: Authorize Access
    print_step(5, "Authorize App to Access Your Nest Devices")
    print("Generate an authorization code to get access tokens.")

    if not (client_id and project_id):
        print("\n❌ Error: Missing credentials. Please restart the setup.")
        return

    # Generate authorization URL
    auth_url = (
        f"https://nestservices.google.com/partnerconnections/{project_id}/auth?"
        f"redirect_uri=https://www.google.com&"
        f"access_type=offline&"
        f"prompt=consent&"
        f"client_id={client_id}&"
        f"response_type=code&"
        f"scope=https://www.googleapis.com/auth/sdm.service"
    )

    print("\nThis will open the authorization URL.")
    print("\nInstructions:")
    print("  1. Sign in with your Google account")
    print("  2. Select the home and devices to authorize")
    print("  3. Click 'Allow'")
    print("  4. You'll be redirected to https://www.google.com/?code=...")
    print("  5. Copy the 'code' parameter from the URL")
    print("     (it's the long string after 'code=')")

    response = input("\nOpen authorization URL? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open(auth_url)
    else:
        print(f"\nManually open this URL:\n{auth_url}")

    print("\n")
    auth_code = input("Enter the authorization code from the URL: ").strip()

    if not auth_code:
        print("\n❌ No authorization code provided.")
        print("You can run this script again or manually exchange the code.")
        print("\nSee README.md for manual token exchange instructions.")
        return

    # Exchange code for tokens
    print("\n🔄 Exchanging authorization code for tokens...")

    if NestClient.exchange_authorization_code(config, auth_code):
        access_token = config.access_token
        print("✓ Tokens obtained and saved to keychain!")
    else:
        print("❌ Error: Token exchange failed")
        return

    wait_for_user()

    # Step 6: List Devices
    print_step(6, "List Your Nest Devices")
    print("Finding your cameras...")

    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"https://smartdevicemanagement.googleapis.com/v1/enterprises/{project_id}/devices",
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            devices = data.get("devices", [])

            if not devices:
                print("\n❌ No devices found. Make sure you:")
                print("  • Authorized the correct home")
                print("  • Have a Nest camera")
                print("  • The camera is online")
                return

            print(f"\n✓ Found {len(devices)} device(s):\n")

            for i, device in enumerate(devices, 1):
                device_id = device.get("name")
                device_type = device.get("type")
                traits = device.get("traits", {})
                info = traits.get("sdm.devices.traits.Info", {})
                device_name = info.get("customName", "Unknown")

                print(f"  {i}. {device_name}")
                print(f"     Type: {device_type}")
                print(f"     ID: {device_id}")
                print()

            if len(devices) == 1:
                # Auto-select if only one device
                device_id = devices[0].get("name")
                config.device_id = device_id
                print(f"✓ Automatically selected device: {device_id}")
            else:
                # Let user choose
                choice = input(f"Select a device (1-{len(devices)}): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(devices):
                        device_id = devices[idx].get("name")
                        config.device_id = device_id
                        print(f"✓ Selected device: {device_id}")
                    else:
                        print("❌ Invalid selection")
                        return
                except ValueError:
                    print("❌ Invalid input")
                    return

        else:
            print(f"❌ Error listing devices (status {response.status_code})")
            print(f"Response: {response.text}")
            return

    except Exception as e:
        print(f"❌ Error listing devices: {e}")
        return

    # Success!
    print_header("✅ Setup Complete!")
    print("Your Nest Bird Detector is now configured!")
    print("\nConfiguration saved to: .env")
    print("\nNext steps:")
    print("  1. Run: python3 main.py")
    print("  2. The app will start monitoring for birds!")
    print("\nNote: Access tokens expire after 1 hour, but will be automatically refreshed.")
    print("      Refresh tokens expire after 7 days (you'll need to re-authorize).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
