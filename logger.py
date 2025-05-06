"""Logging configuration for the application."""
import logging
from config import Config

def setup_logger():
    """Configure application logging."""
    logger = logging.getLogger('cloudybot')
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    # File handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()
