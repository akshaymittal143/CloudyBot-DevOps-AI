"""
Core CloudyBot functionality.

This package contains the main bot logic and core components.
"""

from cloudybot.core.bot import CloudyBot
from cloudybot.core.exceptions import CloudyBotError, ClientError, ConfigurationError

__all__ = [
    "CloudyBot",
    "CloudyBotError",
    "ClientError", 
    "ConfigurationError",
] 