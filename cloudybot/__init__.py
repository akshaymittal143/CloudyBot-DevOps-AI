"""
CloudyBot: AI-Powered DevOps Assistant

A comprehensive DevOps assistant powered by AI, helping with cloud infrastructure,
Kubernetes, Docker, CI/CD, and more.
"""

__version__ = "0.2.0"
__author__ = "Akshay Mittal"
__email__ = "akshaycanodia@gmail.com"
__license__ = "Apache-2.0"

from cloudybot.core.bot import CloudyBot
from cloudybot.core.exceptions import CloudyBotError, ClientError, ConfigurationError

__all__ = [
    "CloudyBot",
    "CloudyBotError", 
    "ClientError",
    "ConfigurationError",
    "__version__",
] 