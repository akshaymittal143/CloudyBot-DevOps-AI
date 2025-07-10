"""
Configuration management for CloudyBot.

This package handles all configuration-related functionality including
settings management, environment variables, and logging configuration.
"""

from cloudybot.config.settings import Settings, get_settings
from cloudybot.config.logging import setup_logging, get_logger

__all__ = [
    "Settings",
    "get_settings", 
    "setup_logging",
    "get_logger",
] 