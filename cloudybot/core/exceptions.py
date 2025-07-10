"""
Custom exceptions for CloudyBot.

This module defines custom exception classes used throughout the CloudyBot application
for better error handling and debugging.
"""

from typing import Optional


class CloudyBotError(Exception):
    """Base exception class for CloudyBot-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        """
        Initialize CloudyBot error.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class ClientError(CloudyBotError):
    """Exception raised for AI client-related errors."""

    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Initialize client error.

        Args:
            message: Error message
            provider: AI provider name (e.g., 'OPENAI', 'HUGGINGFACE')
            model: Model name that caused the error
            details: Optional additional error details
        """
        super().__init__(message, details)
        self.provider = provider
        self.model = model

    def __str__(self) -> str:
        """Return string representation of the client error."""
        error_parts = [self.message]
        
        if self.provider:
            error_parts.append(f"Provider: {self.provider}")
        
        if self.model:
            error_parts.append(f"Model: {self.model}")
            
        if self.details:
            error_parts.append(f"Details: {self.details}")
            
        return ". ".join(error_parts)


class ConfigurationError(CloudyBotError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            details: Optional additional error details
        """
        super().__init__(message, details)
        self.config_key = config_key

    def __str__(self) -> str:
        """Return string representation of the configuration error."""
        error_parts = [self.message]
        
        if self.config_key:
            error_parts.append(f"Configuration key: {self.config_key}")
            
        if self.details:
            error_parts.append(f"Details: {self.details}")
            
        return ". ".join(error_parts)


class ModelLoadError(ClientError):
    """Exception raised when a model fails to load."""

    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        provider: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Initialize model load error.

        Args:
            message: Error message
            model_path: Path or name of the model that failed to load
            provider: AI provider name
            details: Optional additional error details
        """
        super().__init__(message, provider, model_path, details)
        self.model_path = model_path


class APIError(ClientError):
    """Exception raised for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[dict] = None,
        provider: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
            response_data: API response data
            provider: AI provider name
            details: Optional additional error details
        """
        super().__init__(message, provider, None, details)
        self.status_code = status_code
        self.response_data = response_data

    def __str__(self) -> str:
        """Return string representation of the API error."""
        error_parts = [self.message]
        
        if self.provider:
            error_parts.append(f"Provider: {self.provider}")
            
        if self.status_code:
            error_parts.append(f"Status Code: {self.status_code}")
            
        if self.response_data:
            error_parts.append(f"Response: {self.response_data}")
            
        if self.details:
            error_parts.append(f"Details: {self.details}")
            
        return ". ".join(error_parts) 