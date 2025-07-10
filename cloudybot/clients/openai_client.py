"""
OpenAI client implementation for CloudyBot.

This module provides OpenAI API integration with comprehensive error handling,
retry logic, and proper async support.
"""

import asyncio
from typing import Optional, Dict, Any, List
from functools import wraps

import openai
from openai import OpenAI, AsyncOpenAI

from cloudybot.clients.base import AIClient, ChatHistory, Message
from cloudybot.core.exceptions import ClientError, APIError, ConfigurationError
from cloudybot.config.settings import OpenAISettings


def async_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for async retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (openai.RateLimitError, openai.APITimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(wait_time)
                        continue
                    break
                except openai.APIError as e:
                    # Don't retry for other API errors
                    raise APIError(
                        f"OpenAI API error: {e}",
                        status_code=getattr(e, 'status_code', None),
                        provider="OPENAI"
                    )
            
            # If we've exhausted retries, raise the last exception
            if last_exception:
                raise APIError(
                    f"OpenAI API failed after {max_retries} retries: {last_exception}",
                    provider="OPENAI"
                )
                
        return wrapper
    return decorator


class OpenAIClient(AIClient):
    """
    OpenAI client implementation with async support and comprehensive error handling.
    """
    
    def __init__(
        self,
        settings: OpenAISettings,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            settings: OpenAI configuration settings
            **kwargs: Additional configuration options
        """
        super().__init__(settings.model, **kwargs)
        self.settings = settings
        self._client: Optional[AsyncOpenAI] = None
        self._sync_client: Optional[OpenAI] = None
        
        # Validate API key
        if not settings.api_key:
            raise ConfigurationError(
                "OpenAI API key is required",
                config_key="OPENAI_API_KEY"
            )
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            self.logger.info(f"Initializing OpenAI client with model: {self.model}")
            
            # Initialize async client
            self._client = AsyncOpenAI(
                api_key=self.settings.api_key,
                timeout=self.settings.timeout,
                max_retries=0  # We handle retries manually
            )
            
            # Initialize sync client for non-async usage
            self._sync_client = OpenAI(
                api_key=self.settings.api_key,
                timeout=self.settings.timeout,
                max_retries=0
            )
            
            # Test the connection
            await self._test_connection()
            
            self._is_initialized = True
            self.logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ClientError(
                f"Failed to initialize OpenAI client: {e}",
                provider="OPENAI",
                model=self.model
            )
    
    async def _test_connection(self) -> None:
        """Test the OpenAI API connection."""
        try:
            # Make a simple API call to test connectivity
            response = await self._client.models.list()
            self.logger.debug(f"OpenAI connection test successful. Available models: {len(response.data)}")
        except Exception as e:
            raise APIError(
                f"OpenAI connection test failed: {e}",
                provider="OPENAI"
            )
    
    def is_available(self) -> bool:
        """Check if the OpenAI client is available."""
        return (
            self._is_initialized 
            and self._client is not None 
            and bool(self.settings.api_key)
        )
    
    @async_retry(max_retries=3)
    async def generate_response(
        self,
        question: str,
        chat_history: Optional[ChatHistory] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            question: User's question
            chat_history: Optional chat history for context
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            ClientError: If client is not available or initialized
            APIError: If API call fails
        """
        if not self.is_available():
            raise ClientError(
                "OpenAI client is not available. Please initialize first.",
                provider="OPENAI",
                model=self.model
            )
        
        try:
            # Prepare messages
            messages = self._prepare_messages(question, chat_history)
            
            # Extract generation parameters
            temperature = kwargs.get('temperature', self.settings.temperature)
            max_tokens = kwargs.get('max_tokens', self.settings.max_tokens)
            
            self.logger.debug(f"Generating response with {len(messages)} messages")
            
            # Make API call
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            
            # Extract response content
            if not response.choices:
                raise APIError(
                    "No response choices returned from OpenAI API",
                    provider="OPENAI"
                )
            
            content = response.choices[0].message.content
            if not content:
                raise APIError(
                    "Empty response content from OpenAI API",
                    provider="OPENAI"
                )
            
            self.logger.debug(f"Generated response of length: {len(content)}")
            return content.strip()
            
        except openai.AuthenticationError as e:
            raise APIError(
                f"OpenAI authentication failed: {e}",
                status_code=401,
                provider="OPENAI"
            )
        except openai.PermissionDeniedError as e:
            raise APIError(
                f"OpenAI permission denied: {e}",
                status_code=403,
                provider="OPENAI"
            )
        except openai.NotFoundError as e:
            raise APIError(
                f"OpenAI model not found: {e}",
                status_code=404,
                provider="OPENAI",
                details={"model": self.model}
            )
        except openai.RateLimitError as e:
            # This should be handled by the retry decorator
            raise APIError(
                f"OpenAI rate limit exceeded: {e}",
                status_code=429,
                provider="OPENAI"
            )
        except openai.APIError as e:
            raise APIError(
                f"OpenAI API error: {e}",
                status_code=getattr(e, 'status_code', None),
                provider="OPENAI"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI response generation: {e}")
            raise ClientError(
                f"Unexpected error generating response: {e}",
                provider="OPENAI",
                model=self.model
            )
    
    def _prepare_messages(
        self,
        question: str,
        chat_history: Optional[ChatHistory] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI API.
        
        Args:
            question: User's question
            chat_history: Optional chat history
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message
        system_message = (
            "You are CloudyBot, an AI-powered DevOps assistant. "
            "Help users with cloud infrastructure, Kubernetes, Docker, CI/CD, "
            "and other DevOps practices. Provide clear, accurate, and practical advice."
        )
        messages.append({"role": "system", "content": system_message})
        
        # Add chat history if provided
        if chat_history and len(chat_history) > 0:
            # Limit history to avoid token limits
            recent_messages = chat_history.get_messages(limit=5)
            for msg in recent_messages:
                messages.append(msg.to_dict())
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        return messages
    
    def generate_response_sync(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous version of generate_response for backward compatibility.
        
        Args:
            question: User's question
            chat_history: Optional chat history as list of dicts
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        if not self.is_available():
            raise ClientError(
                "OpenAI client is not available. Please initialize first.",
                provider="OPENAI",
                model=self.model
            )
        
        try:
            # Convert dict history to ChatHistory if provided
            history = None
            if chat_history:
                history = ChatHistory.from_dict_list(chat_history)
            
            # Prepare messages
            messages = self._prepare_messages(question, history)
            
            # Extract generation parameters
            temperature = kwargs.get('temperature', self.settings.temperature)
            max_tokens = kwargs.get('max_tokens', self.settings.max_tokens)
            
            # Make API call using sync client
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            
            # Extract response content
            if not response.choices:
                raise APIError(
                    "No response choices returned from OpenAI API",
                    provider="OPENAI"
                )
            
            content = response.choices[0].message.content
            if not content:
                raise APIError(
                    "Empty response content from OpenAI API",
                    provider="OPENAI"
                )
            
            return content.strip()
            
        except Exception as e:
            if isinstance(e, (ClientError, APIError)):
                raise
            
            self.logger.error(f"Error in sync response generation: {e}")
            raise ClientError(
                f"Error generating response: {e}",
                provider="OPENAI",
                model=self.model
            )
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models.
        
        Returns:
            List of available model names
        """
        if not self.is_available():
            raise ClientError(
                "OpenAI client is not available",
                provider="OPENAI"
            )
        
        try:
            response = await self._client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            raise APIError(
                f"Failed to get available models: {e}",
                provider="OPENAI"
            )
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current client settings.
        
        Returns:
            Dictionary of current settings
        """
        return {
            "model": self.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.timeout,
            "max_retries": self.settings.max_retries,
            "provider": "OPENAI"
        } 