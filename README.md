# 🐦 Nest Bird Detector

A macOS desktop application that monitors your Nest camera and alerts you when it detects birds using AI-powered computer vision.

Built by Claude. Very much an educational toy project. Models don't work
great. Code has not been reviewed by a human. Works on my machine, etc.


## Features

- ✅ **Real-time Bird Detection**: Uses YOLOv8 AI model to detect birds in a live WebRTC video stream
- ✅ **Species Identification**: EfficientNet classifier identifies 525+ bird species
- ✅ **Notifications**: macOS system notifications, saved snapshots, and detailed logs
- ✅ **Configurable**: Adjust detection confidence and check intervals
- ✅ **Full GUI**: PyQt5-based interface with live video display and controls
- ✅ **Feedback Loop**: Review detections, confirm or correct species, and improve over time

## Architecture

```
┌─────────────────┐
│   PyQt5 GUI     │  ← User Interface
└────────┬────────┘
         │
    ┌────┴─────┬────────────┬──────────────┐
    │          │            │              │
┌───▼───┐ ┌───▼────┐ ┌─────▼─────┐ ┌──────▼──────┐
│ Nest  │ │ YOLO   │ │ Species   │ │ Notification│
│Client │ │Detector│ │Classifier │ │  Manager    │
└───┬───┘ └────────┘ └───────────┘ └─────────────┘
    │
┌───▼─────────┐
│   WebRTC    │
│   Manager   │
└───┬─────────┘
    │
┌───▼─────┐
│ Nest    │
│ Camera  │
└─────────┘
```

## Requirements

- macOS (tested on macOS 14+)
- Python 3.11 or higher
- Google Nest camera
- Google Cloud Platform account ($5 Device Access fee)

AI models (YOLOv8 and EfficientNet) are downloaded automatically on first run.

## Installation

### 1. Clone or Download

```bash
cd ~/nest-bird-detector
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Nest API

```bash
cp .env.example .env
python3 tools/setup_credentials.py
```

Follow the interactive guide to:
- Register for Google Device Access ($5 one-time fee)
- Create OAuth 2.0 credentials
- Authorize access to your Nest camera
- Test the connection

The setup process takes about 20-30 minutes.

## Usage

### Start the Application

```bash
source venv/bin/activate
python3 src/main.py
```

Or make it easier with an alias:

```bash
alias bird-detector='cd ~/nest-bird-detector && source venv/bin/activate && python3 src/main.py'
```

Then just run: `bird-detector`

### Application Controls

- **Start Monitoring**: Begin watching the live camera stream for birds
- **Confidence Threshold**: Minimum detection confidence (0.3-0.9)
- **Review Detections**: See recent detections and correct species labels

### What Happens When a Bird is Detected

1. **macOS Notification**: System alert with bird count and sound
2. **Snapshot Saved**: Image with bounding boxes saved to `snapshots/`
3. **Detection Logged**: Details written to `logs/detections.csv`

Notifications have a 30-second cooldown to prevent spam.

## Configuration

Edit `.env` file to customize:

```bash
# Check interval (seconds)
CHECK_INTERVAL=5

# Minimum confidence (0.0 - 1.0)
CONFIDENCE_THRESHOLD=0.5
```

Both settings can also be adjusted in the GUI while the app is running.

## File Structure

```
nest-bird-detector/
├── src/                       # Runtime application code
│   ├── main.py               # Application entry point
│   ├── gui.py                # PyQt5 GUI
│   ├── nest_client.py        # Nest API client (OAuth + WebRTC)
│   ├── webrtc_manager.py     # WebRTC stream management
│   ├── detector.py           # YOLOv8 bird detection
│   ├── hf_species_classifier.py  # Bird species identification
│   ├── ebird_client.py       # eBird API for regional species
│   ├── notifications.py      # Alert system
│   ├── feedback_widget.py    # Detection review UI
│   ├── correction_manager.py # Species correction tracking
│   ├── missed_bird_logger.py # Missed detection logging
│   ├── missed_bird_dialog.py # Missed bird report UI
│   ├── sdp_validator.py      # WebRTC SDP security validation
│   ├── logging_config.py     # Logging setup
│   └── config.py             # Configuration management
├── tools/                     # Setup and training scripts
│   ├── setup_credentials.py  # OAuth setup guide
│   ├── train_custom_model.py # Fine-tune species classifier
│   ├── prepare_training_data.py  # Prepare training dataset
│   └── label_images.py       # Interactive image labeler
├── tests/                     # Test suite
├── requirements.txt           # Python dependencies
└── .env                       # Configuration (gitignored)
```

## Setting Up on a New Machine

Credentials (OAuth secrets, tokens, device ID) are stored in the macOS keychain, not in files. When moving to a new machine:

1. Clone the repo and install dependencies as above
2. Run `python3 tools/setup_credentials.py` to re-authorize

## Troubleshooting

### "No access token available"

Run the setup again:
```bash
python3 tools/setup_credentials.py
```

### "Token refresh failed"

Tokens expire after 7 days in Testing mode. The app will prompt you to
re-authorize automatically on next startup.

### "No devices found"

Check that:
- Camera is online in the Google Home app
- You selected the correct home during authorization
- Camera is powered on

### "pync not available"

macOS notifications require pync. Install it:
```bash
pip install pync
```

## Advanced Usage

### Adjusting Detection

For more sensitive detection:
- Lower confidence threshold (0.3-0.4)
- May increase false positives

For fewer false positives:
- Higher confidence threshold (0.6-0.8)
- May miss some birds

### Running Tests

```bash
python3 -m pytest tests/
```

## API Limitations

- **Stream Duration**: WebRTC streams last 5 minutes, then auto-renew
- **Token Expiry**: Access tokens expire hourly (auto-refreshed)
- **Refresh Tokens**: Expire after 7 days in Testing mode
- **Cost**: $5 one-time Device Access fee

## Privacy & Security

### Data Storage

- **Local Only**: All images and logs stored locally on your Mac
- **No Cloud Upload**: Nothing sent to external servers (except Google Nest API)
- **Credentials**: Sensitive tokens and secrets stored in OS keychain (not in files)

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv/

# Remove all data (optional)
rm -rf snapshots/ logs/

# Remove configuration (optional)
rm .env

# Remove stored credentials from keychain (optional)
python3 -c "import keyring; [keyring.delete_password('nest-bird-detector', k) for k in ['OAUTH_CLIENT_SECRET','ACCESS_TOKEN','REFRESH_TOKEN','DEVICE_ID','EBIRD_API_KEY']]"
```

## Development

### Project Structure

- `src/main.py`: Entry point and connection health check
- `src/gui.py`: PyQt5 interface
- `src/nest_client.py`: Nest API integration and OAuth
- `src/webrtc_manager.py`: WebRTC stream lifecycle
- `src/sdp_validator.py`: WebRTC SDP security validation
- `src/detector.py`: YOLOv8 bird detection
- `src/hf_species_classifier.py`: Species identification (525+ species)
- `src/ebird_client.py`: eBird API for regional filtering
- `src/notifications.py`: Alert system (macOS notifications and snapshots)
- `src/feedback_widget.py`: Detection review and correction UI
- `src/correction_manager.py`: Species correction persistence
- `src/missed_bird_logger.py`: Missed detection logging
- `src/missed_bird_dialog.py`: Missed bird report UI
- `src/logging_config.py`: Logging setup
- `src/config.py`: Configuration and keychain management

## FAQ

**Q: Does this work with Nest Aware recordings?**
A: No, the API provides live WebRTC streams only, not recorded video.

**Q: Can I detect specific bird species?**
A: Yes! After YOLOv8 detects a bird, an EfficientNet classifier identifies the species from 525+ possibilities. You can also connect to the eBird API to filter by species seen in your region.

**Q: How much bandwidth does this use?**
A: The WebRTC video stream uses roughly 1-2 Mbps depending on resolution.

**Q: Can I run this 24/7?**
A: Yes, but remember to refresh tokens every 7 days.

**Q: Does this work with multiple cameras?**
A: Currently monitors one camera. You can modify the code to support multiple devices.

**Q: Is the bird detection accurate?**
A: YOLOv8 is quite accurate (80-90%) but can have false positives. Adjust the confidence threshold to tune it.

## License

This project is for personal use. Nest API usage subject to Google's terms of service.

## Support

For issues:
1. Check `logs/detections.csv` for detection history
2. Run `python3 -m pytest tests/` to check for issues
3. Review Nest API documentation: https://developers.google.com/nest/device-access

## Credits

- **YOLOv8**: Ultralytics (https://github.com/ultralytics/ultralytics)
- **EfficientNet**: via Hugging Face Transformers
- **Google Nest API**: https://developers.google.com/nest/device-access
- **eBird API**: https://ebird.org/home
- **PyQt5**: Riverbank Computing

---

Made with ❤️ for bird watching 🐦
