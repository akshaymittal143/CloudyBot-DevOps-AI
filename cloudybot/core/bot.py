"""
Main CloudyBot class implementation.

This module provides the primary CloudyBot interface that orchestrates
AI clients and manages conversations.
"""

import asyncio
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from cloudybot.clients.base import AIClient, ChatHistory
from cloudybot.clients.openai_client import OpenAIClient
from cloudybot.clients.hf_client import HuggingFaceClient
from cloudybot.config.settings import get_settings, ModelProvider, Settings
from cloudybot.config.logging import LoggerMixin, get_logger
from cloudybot.core.exceptions import CloudyBotError, ClientError, ConfigurationError


class BotStatus(str, Enum):
    """Bot status enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


class CloudyBot(LoggerMixin):
    """
    Main CloudyBot class for DevOps AI assistance.
    
    This class provides a high-level interface for interacting with AI models
    and managing conversations about DevOps topics.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        auto_initialize: bool = True
    ):
        """
        Initialize CloudyBot.
        
        Args:
            settings: Optional settings. If None, loads from environment.
            auto_initialize: Whether to automatically initialize the bot.
        """
        self.settings = settings or get_settings()
        self._clients: Dict[str, AIClient] = {}
        self._current_provider: Optional[str] = None
        self._chat_history = ChatHistory(max_size=self.settings.max_chat_history)
        self._status = BotStatus.IDLE
        self._initialization_error: Optional[Exception] = None
        
        if auto_initialize:
            asyncio.create_task(self.initialize())
    
    async def initialize(self) -> None:
        """
        Initialize the bot and its AI clients.
        
        Raises:
            CloudyBotError: If initialization fails
        """
        try:
            self._status = BotStatus.INITIALIZING
            self.logger.info("Initializing CloudyBot...")
            
            # Initialize clients based on configuration
            await self._initialize_clients()
            
            # Set current provider
            self._current_provider = self.settings.model_provider.value
            
            self._status = BotStatus.READY
            self.logger.info("CloudyBot initialized successfully")
            
        except Exception as e:
            self._status = BotStatus.ERROR
            self._initialization_error = e
            self.logger.error(f"Failed to initialize CloudyBot: {e}")
            raise CloudyBotError(f"Failed to initialize CloudyBot: {e}")
    
    async def _initialize_clients(self) -> None:
        """Initialize AI clients based on configuration."""
        # Initialize OpenAI client if configured
        if (self.settings.model_provider == ModelProvider.OPENAI or
            self.settings.openai.api_key):
            try:
                openai_client = OpenAIClient(self.settings.openai)
                await openai_client.initialize()
                self._clients["OPENAI"] = openai_client
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
                if self.settings.model_provider == ModelProvider.OPENAI:
                    raise
        
        # Initialize Hugging Face client
        try:
            hf_client = HuggingFaceClient(self.settings.huggingface)
            await hf_client.initialize()
            self._clients["HUGGINGFACE"] = hf_client
            self.logger.info("Hugging Face client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Hugging Face client: {e}")
            if self.settings.model_provider == ModelProvider.HUGGINGFACE:
                raise
        
        # Ensure at least one client is available
        if not self._clients:
            raise ConfigurationError("No AI clients could be initialized")
    
    async def ask(
        self,
        question: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_history: bool = True,
        **kwargs
    ) -> str:
        """
        Ask a question and get a response.
        
        Args:
            question: The question to ask
            provider: Optional specific provider to use
            model: Optional specific model to use
            use_history: Whether to include chat history
            **kwargs: Additional generation parameters
            
        Returns:
            The AI's response
            
        Raises:
            CloudyBotError: If the bot is not ready or request fails
        """
        if self._status != BotStatus.READY:
            if self._status == BotStatus.ERROR and self._initialization_error:
                raise CloudyBotError(
                    f"Bot is in error state: {self._initialization_error}"
                )
            else:
                raise CloudyBotError(f"Bot is not ready (status: {self._status})")
        
        try:
            self._status = BotStatus.PROCESSING
            self.logger.info(f"Processing question: {question[:100]}...")
            
            # Determine which client to use
            client = self._get_client(provider)
            
            # Update model if specified
            if model and model != client.model:
                client.model = model
            
            # Prepare chat history
            history = self._chat_history if use_history else None
            
            # Generate response
            response = await client.generate_response(
                question=question,
                chat_history=history,
                **kwargs
            )
            
            # Update chat history
            if use_history:
                self._chat_history.add_message("user", question)
                self._chat_history.add_message("assistant", response)
            
            self.logger.info("Question processed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            if isinstance(e, (CloudyBotError, ClientError)):
                raise
            raise CloudyBotError(f"Error processing question: {e}")
        finally:
            self._status = BotStatus.READY
    
    def ask_sync(
        self,
        question: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_history: bool = True,
        **kwargs
    ) -> str:
        """
        Synchronous version of ask for backward compatibility.
        
        Args:
            question: The question to ask
            provider: Optional specific provider to use
            model: Optional specific model to use
            use_history: Whether to include chat history
            **kwargs: Additional generation parameters
            
        Returns:
            The AI's response
        """
        # Convert chat history to dict format for backward compatibility
        chat_history = None
        if use_history and len(self._chat_history) > 0:
            chat_history = self._chat_history.to_dict_list()
        
        # Get the appropriate client
        client = self._get_client(provider)
        
        # Use sync methods for backward compatibility
        if hasattr(client, 'generate_response_sync'):
            response = client.generate_response_sync(
                question=question,
                chat_history=chat_history,
                **kwargs
            )
        else:
            # Fallback to async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    client.generate_response(
                        question=question,
                        chat_history=ChatHistory.from_dict_list(chat_history) if chat_history else None,
                        **kwargs
                    )
                )
            finally:
                loop.close()
        
        # Update chat history
        if use_history:
            self._chat_history.add_message("user", question)
            self._chat_history.add_message("assistant", response)
        
        return response
    
    def _get_client(self, provider: Optional[str] = None) -> AIClient:
        """
        Get the appropriate AI client.
        
        Args:
            provider: Optional specific provider
            
        Returns:
            AI client instance
            
        Raises:
            CloudyBotError: If client is not available
        """
        target_provider = provider or self._current_provider
        
        if not target_provider:
            raise CloudyBotError("No provider specified and no default provider set")
        
        target_provider = target_provider.upper()
        
        if target_provider not in self._clients:
            available = list(self._clients.keys())
            raise CloudyBotError(
                f"Provider '{target_provider}' not available. Available: {available}"
            )
        
        client = self._clients[target_provider]
        if not client.is_available():
            raise CloudyBotError(f"Provider '{target_provider}' is not available")
        
        return client
    
    def switch_provider(self, provider: str) -> None:
        """
        Switch to a different AI provider.
        
        Args:
            provider: Provider name to switch to
            
        Raises:
            CloudyBotError: If provider is not available
        """
        provider = provider.upper()
        
        if provider not in self._clients:
            available = list(self._clients.keys())
            raise CloudyBotError(
                f"Provider '{provider}' not available. Available: {available}"
            )
        
        client = self._clients[provider]
        if not client.is_available():
            raise CloudyBotError(f"Provider '{provider}' is not available")
        
        self._current_provider = provider
        self.logger.info(f"Switched to provider: {provider}")
    
    def clear_history(self) -> None:
        """Clear the chat history."""
        self._chat_history.clear()
        self.logger.info("Chat history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current chat history.
        
        Returns:
            List of message dictionaries
        """
        return self._chat_history.to_dict_list()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current bot status and information.
        
        Returns:
            Dictionary containing status information
        """
        status_info = {
            "status": self._status.value,
            "current_provider": self._current_provider,
            "available_providers": list(self._clients.keys()),
            "chat_history_length": len(self._chat_history),
            "settings": {
                "model_provider": self.settings.model_provider.value,
                "max_chat_history": self.settings.max_chat_history
            }
        }
        
        if self._initialization_error:
            status_info["initialization_error"] = str(self._initialization_error)
        
        # Add client information
        client_info = {}
        for provider, client in self._clients.items():
            client_info[provider] = client.get_model_info()
        status_info["clients"] = client_info
        
        return status_info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check.
        
        Returns:
            Dictionary containing health status
        """
        health_status = {
            "overall_status": "healthy",
            "bot_status": self._status.value,
            "timestamp": asyncio.get_event_loop().time(),
            "clients": {}
        }
        
        # Check each client
        for provider, client in self._clients.items():
            try:
                client_health = await client.health_check()
                health_status["clients"][provider] = client_health
                
                if client_health.get("status") != "healthy":
                    health_status["overall_status"] = "degraded"
                    
            except Exception as e:
                health_status["clients"][provider] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    def get_examples(self) -> List[str]:
        """
        Get example questions for users.
        
        Returns:
            List of example questions
        """
        return self.settings.default_examples.copy()
    
    async def reload_settings(self, new_settings: Optional[Settings] = None) -> None:
        """
        Reload settings and reinitialize if necessary.
        
        Args:
            new_settings: Optional new settings to use
        """
        self.logger.info("Reloading settings...")
        
        old_provider = self.settings.model_provider
        self.settings = new_settings or get_settings(reload=True)
        
        # If provider changed, reinitialize
        if old_provider != self.settings.model_provider:
            self.logger.info("Provider changed, reinitializing...")
            await self.initialize()
        
        # Update chat history size
        self._chat_history.max_size = self.settings.max_chat_history
    
    def __repr__(self) -> str:
        """String representation of the bot."""
        return (
            f"CloudyBot(status={self._status.value}, "
            f"provider={self._current_provider}, "
            f"clients={list(self._clients.keys())})"
        ) 