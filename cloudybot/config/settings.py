"""
Settings and configuration management for CloudyBot.

This module provides centralized configuration management with environment variable
support, type validation, and default values.
"""

import os
from enum import Enum
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

from cloudybot.core.exceptions import ConfigurationError


class ModelProvider(str, Enum):
    """Supported AI model providers."""
    OPENAI = "OPENAI"
    HUGGINGFACE = "HUGGINGFACE"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class OpenAISettings:
    """OpenAI-specific configuration settings."""
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3


@dataclass
class HuggingFaceSettings:
    """Hugging Face-specific configuration settings."""
    model: str = "google/flan-t5-base"
    api_token: Optional[str] = None
    temperature: float = 0.7
    max_length: int = 512
    device: str = "auto"  # "auto", "cpu", "cuda"
    cache_dir: Optional[str] = None


@dataclass
class UISettings:
    """User interface configuration settings."""
    page_title: str = "CloudyBot - DevOps Assistant"
    page_icon: str = "☁️"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    max_width: int = 1200
    theme_color: str = "#007fff"


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "logs/cloudybot.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class Settings:
    """Main configuration settings for CloudyBot."""
    
    # Core settings
    model_provider: ModelProvider = ModelProvider.OPENAI
    debug: bool = False
    
    # Provider-specific settings
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    huggingface: HuggingFaceSettings = field(default_factory=HuggingFaceSettings)
    
    # UI settings
    ui: UISettings = field(default_factory=UISettings)
    
    # Logging settings
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    
    # Chat settings
    max_chat_history: int = 10
    default_examples: list = field(default_factory=lambda: [
        "How do I restart a Kubernetes pod?",
        "Explain blue-green deployment.",
        "How to debug Docker container failures?",
        "What's the difference between Docker and Kubernetes?",
        "How can I automate AWS infrastructure deployment?"
    ])

    def validate(self) -> None:
        """Validate configuration settings."""
        # Validate OpenAI settings
        if self.model_provider == ModelProvider.OPENAI:
            if not self.openai.api_key:
                raise ConfigurationError(
                    "OpenAI API key is required when using OpenAI provider",
                    config_key="OPENAI_API_KEY"
                )
            
            if self.openai.temperature < 0 or self.openai.temperature > 2:
                raise ConfigurationError(
                    "OpenAI temperature must be between 0 and 2",
                    config_key="openai.temperature"
                )
        
        # Validate Hugging Face settings
        if self.model_provider == ModelProvider.HUGGINGFACE:
            if self.huggingface.temperature < 0 or self.huggingface.temperature > 2:
                raise ConfigurationError(
                    "Hugging Face temperature must be between 0 and 2",
                    config_key="huggingface.temperature"
                )
        
        # Validate UI settings
        if self.ui.max_width < 800:
            raise ConfigurationError(
                "UI max width must be at least 800 pixels",
                config_key="ui.max_width"
            )
        
        # Validate chat settings
        if self.max_chat_history < 1:
            raise ConfigurationError(
                "Max chat history must be at least 1",
                config_key="max_chat_history"
            )


def load_from_env() -> Settings:
    """
    Load settings from environment variables.
    
    Returns:
        Settings object populated from environment variables
        
    Raises:
        ConfigurationError: If required settings are missing or invalid
    """
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Core settings
        model_provider = ModelProvider(
            os.getenv("MODEL_PROVIDER", ModelProvider.OPENAI.value).upper()
        )
        debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        
        # OpenAI settings
        openai_settings = OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        )
        
        # Hugging Face settings
        huggingface_settings = HuggingFaceSettings(
            model=os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base"),
            api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            temperature=float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7")),
            max_length=int(os.getenv("HUGGINGFACE_MAX_LENGTH", "512")),
            device=os.getenv("HUGGINGFACE_DEVICE", "auto"),
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR")
        )
        
        # UI settings
        ui_settings = UISettings(
            page_title=os.getenv("UI_PAGE_TITLE", "CloudyBot - DevOps Assistant"),
            page_icon=os.getenv("UI_PAGE_ICON", "☁️"),
            layout=os.getenv("UI_LAYOUT", "wide"),
            sidebar_state=os.getenv("UI_SIDEBAR_STATE", "expanded"),
            max_width=int(os.getenv("UI_MAX_WIDTH", "1200")),
            theme_color=os.getenv("UI_THEME_COLOR", "#007fff")
        )
        
        # Logging settings
        logging_settings = LoggingSettings(
            level=LogLevel(os.getenv("LOG_LEVEL", LogLevel.INFO.value).upper()),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_file=os.getenv("LOG_FILE", "logs/cloudybot.log"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            console_output=os.getenv("LOG_CONSOLE_OUTPUT", "true").lower() in ("true", "1", "yes")
        )
        
        # Chat settings
        max_chat_history = int(os.getenv("MAX_CHAT_HISTORY", "10"))
        
        settings = Settings(
            model_provider=model_provider,
            debug=debug,
            openai=openai_settings,
            huggingface=huggingface_settings,
            ui=ui_settings,
            logging=logging_settings,
            max_chat_history=max_chat_history
        )
        
        # Validate settings
        settings.validate()
        
        return settings
        
    except ValueError as e:
        raise ConfigurationError(f"Invalid configuration value: {e}")
    except Exception as e:
        import traceback
        print("DEBUG: load_from_env called")
        print("Settings load error:", e)
        traceback.print_exc()
        raise ConfigurationError(f"Error loading configuration: {e}")


def load_from_secrets(secrets: Dict[str, Any]) -> Settings:
    """
    Load settings from Streamlit secrets or similar dictionary.
    
    Args:
        secrets: Dictionary containing configuration values
        
    Returns:
        Settings object populated from secrets
        
    Raises:
        ConfigurationError: If required settings are missing or invalid
    """
    try:
        # Core settings
        model_provider = ModelProvider(
            secrets.get("MODEL_PROVIDER", ModelProvider.OPENAI.value).upper()
        )
        debug = str(secrets.get("DEBUG", "false")).lower() in ("true", "1", "yes")
        
        # OpenAI settings
        openai_settings = OpenAISettings(
            api_key=secrets.get("OPENAI_API_KEY"),
            model=secrets.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(secrets.get("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(secrets.get("OPENAI_MAX_TOKENS", "1000")),
            timeout=int(secrets.get("OPENAI_TIMEOUT", "30")),
            max_retries=int(secrets.get("OPENAI_MAX_RETRIES", "3"))
        )
        
        # Hugging Face settings
        huggingface_settings = HuggingFaceSettings(
            model=secrets.get("HUGGINGFACE_MODEL", "google/flan-t5-base"),
            api_token=secrets.get("HUGGINGFACE_API_TOKEN"),
            temperature=float(secrets.get("HUGGINGFACE_TEMPERATURE", "0.7")),
            max_length=int(secrets.get("HUGGINGFACE_MAX_LENGTH", "512")),
            device=secrets.get("HUGGINGFACE_DEVICE", "auto"),
            cache_dir=secrets.get("HUGGINGFACE_CACHE_DIR")
        )
        
        # UI settings
        ui_settings = UISettings(
            page_title=secrets.get("UI_PAGE_TITLE", "CloudyBot - DevOps Assistant"),
            page_icon=secrets.get("UI_PAGE_ICON", "☁️"),
            layout=secrets.get("UI_LAYOUT", "wide"),
            sidebar_state=secrets.get("UI_SIDEBAR_STATE", "expanded"),
            max_width=int(secrets.get("UI_MAX_WIDTH", "1200")),
            theme_color=secrets.get("UI_THEME_COLOR", "#007fff")
        )
        
        # Logging settings
        logging_settings = LoggingSettings(
            level=LogLevel(secrets.get("LOG_LEVEL", LogLevel.INFO.value).upper()),
            format=secrets.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_file=secrets.get("LOG_FILE", "logs/cloudybot.log"),
            max_file_size=int(secrets.get("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(secrets.get("LOG_BACKUP_COUNT", "5")),
            console_output=str(secrets.get("LOG_CONSOLE_OUTPUT", "true")).lower() in ("true", "1", "yes")
        )
        
        # Chat settings
        max_chat_history = int(secrets.get("MAX_CHAT_HISTORY", "10"))
        
        settings = Settings(
            model_provider=model_provider,
            debug=debug,
            openai=openai_settings,
            huggingface=huggingface_settings,
            ui=ui_settings,
            logging=logging_settings,
            max_chat_history=max_chat_history
        )
        
        # Validate settings
        settings.validate()
        
        return settings
        
    except ValueError as e:
        raise ConfigurationError(f"Invalid configuration value: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration from secrets: {e}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(
    reload: bool = False,
    secrets: Optional[Dict[str, Any]] = None
) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        reload: Whether to reload settings from environment
        secrets: Optional secrets dictionary (e.g., from Streamlit)
        
    Returns:
        Settings instance
        
    Raises:
        ConfigurationError: If settings cannot be loaded
    """
    global _settings
    
    if _settings is None or reload:
        if secrets is not None:
            _settings = load_from_secrets(secrets)
        else:
            _settings = load_from_env()
    
    return _settings


def update_settings(**kwargs) -> None:
    """
    Update specific settings values.
    
    Args:
        **kwargs: Settings to update
        
    Raises:
        ConfigurationError: If settings are invalid
    """
    global _settings
    
    if _settings is None:
        _settings = load_from_env()
    
    # Update settings
    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)
        else:
            raise ConfigurationError(f"Unknown setting: {key}")
    
    # Validate updated settings
    _settings.validate() 