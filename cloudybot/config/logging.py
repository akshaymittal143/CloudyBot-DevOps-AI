"""
Logging configuration for CloudyBot.

This module provides comprehensive logging setup with file rotation,
console output, and proper formatting for the CloudyBot application.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from cloudybot.config.settings import get_settings, LoggingSettings


def setup_logging(
    settings: Optional[LoggingSettings] = None,
    force_reload: bool = False
) -> logging.Logger:
    """
    Set up application logging with file and console handlers.
    
    Args:
        settings: Logging settings to use. If None, loads from global settings.
        force_reload: Whether to force reload of logging configuration
        
    Returns:
        Configured logger instance
        
    Raises:
        Exception: If logging setup fails
    """
    if settings is None:
        app_settings = get_settings()
        settings = app_settings.logging
    
    # Get or create the main logger
    logger = logging.getLogger('cloudybot')
    
    # Don't reconfigure if already configured (unless forced)
    if logger.handlers and not force_reload:
        return logger
    
    # Clear existing handlers if force reload
    if force_reload:
        logger.handlers.clear()
    
    # Set log level
    logger.setLevel(getattr(logging, settings.level.value))
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter(settings.format)
    
    # Set up console handler if enabled
    if settings.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, settings.level.value))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Set up file handler if log file is specified
    if settings.log_file:
        try:
            # Create log directory if it doesn't exist
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=settings.log_file,
                maxBytes=settings.max_file_size,
                backupCount=settings.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # Always debug level for file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # If file handler fails, log to console
            logger.warning(f"Failed to set up file logging: {e}")
    
    # Log initial setup message
    logger.info(f"Logging configured - Level: {settings.level.value}, File: {settings.log_file}")
    
    return logger


def get_logger(name: str = 'cloudybot') -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name. Defaults to 'cloudybot'
        
    Returns:
        Logger instance
    """
    # Ensure parent logger is configured
    parent_logger = logging.getLogger('cloudybot')
    if not parent_logger.handlers:
        setup_logging()
    
    # Return child logger if name is different
    if name != 'cloudybot':
        return logging.getLogger(f'cloudybot.{name}')
    
    return parent_logger


def configure_third_party_loggers() -> None:
    """
    Configure third-party library loggers to reduce noise.
    """
    # Reduce noise from HTTP libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Reduce noise from Streamlit
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    
    # Reduce noise from transformers library
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    
    # Reduce noise from torch
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Reduce noise from openai
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this class."""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__.lower()
            self._logger = get_logger(class_name)
        return self._logger


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger('function_calls')
        
        # Log function entry
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_errors(func):
    """
    Decorator to log errors that occur in functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger('errors')
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {e}",
                exc_info=True,
                extra={
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
            )
            raise
    
    return wrapper 