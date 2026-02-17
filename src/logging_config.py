"""
Logging configuration for the Nest Bird Detector application.
"""
import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Configure application-wide logging with console and rotating file output."""
    log_path = Path(log_dir)
    log_path.mkdir(mode=0o700, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers.clear()

    simple_fmt = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(simple_fmt)
    root_logger.addHandler(console)

    # Rotating file (10MB, 5 backups)
    log_file = log_path / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_485_760, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

    if log_file.exists():
        os.chmod(log_file, 0o600)

    # Suppress noisy libraries
    for lib in ['PIL', 'ultralytics', 'aiortc', 'libav', 'libav.h264']:
        logging.getLogger(lib).setLevel(logging.CRITICAL)
