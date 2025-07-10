"""
Abstract base class for AI clients.

This module defines the interface that all AI clients must implement
to ensure consistent behavior across different providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from cloudybot.config.logging import LoggerMixin


class Message:
    """Represents a chat message."""
    
    def __init__(self, role: str, content: str):
        """
        Initialize a message.
        
        Args:
            role: Message role (e.g., 'user', 'assistant', 'system')
            content: Message content
        """
        self.role = role
        self.content = content
    
    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create message from dictionary format."""
        return cls(role=data["role"], content=data["content"])
    
    def __repr__(self) -> str:
        """String representation of message."""
        return f"Message(role='{self.role}', content='{self.content[:50]}...')"


class ChatHistory:
    """Manages chat history with size limits."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize chat history.
        
        Args:
            max_size: Maximum number of messages to keep
        """
        self.max_size = max_size
        self.messages: List[Message] = []
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to history.
        
        Args:
            role: Message role
            content: Message content
        """
        message = Message(role, content)
        self.messages.append(message)
        
        # Keep only the latest messages within size limit
        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """
        Get messages from history.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:] if limit > 0 else []
    
    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()
    
    def to_dict_list(self) -> List[Dict[str, str]]:
        """Convert history to list of dictionaries."""
        return [msg.to_dict() for msg in self.messages]
    
    @classmethod
    def from_dict_list(cls, data: List[Dict[str, str]], max_size: int = 10) -> "ChatHistory":
        """
        Create chat history from list of dictionaries.
        
        Args:
            data: List of message dictionaries
            max_size: Maximum number of messages to keep
            
        Returns:
            ChatHistory instance
        """
        history = cls(max_size)
        for msg_data in data:
            history.add_message(msg_data["role"], msg_data["content"])
        return history
    
    def __len__(self) -> int:
        """Get number of messages in history."""
        return len(self.messages)
    
    def __bool__(self) -> bool:
        """Check if history has any messages."""
        return len(self.messages) > 0


class AIClient(ABC, LoggerMixin):
    """
    Abstract base class for AI clients.
    
    All AI clients must inherit from this class and implement the required methods.
    """
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the AI client.
        
        Args:
            model: Model name or identifier
            **kwargs: Additional client-specific configuration
        """
        self.model = model
        self.config = kwargs
        self._is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the client (load models, authenticate, etc.).
        
        This method should be called before using the client.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        question: str, 
        chat_history: Optional[ChatHistory] = None,
        **kwargs
    ) -> str:
        """
        Generate a response to a question.
        
        Args:
            question: User's question
            chat_history: Optional chat history for context
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the client is available and ready to use.
        
        Returns:
            True if client is available, False otherwise
            
        Must be implemented by subclasses.
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if client has been initialized."""
        return self._is_initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model": self.model,
            "provider": self.__class__.__name__,
            "initialized": self.is_initialized,
            "available": self.is_available()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the client.
        
        Returns:
            Dictionary containing health status
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            available = self.is_available()
            status = "healthy" if available else "unhealthy"
            
            return {
                "status": status,
                "model": self.model,
                "provider": self.__class__.__name__,
                "initialized": self.is_initialized,
                "available": available
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "model": self.model,
                "provider": self.__class__.__name__,
                "error": str(e)
            }
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return f"{self.__class__.__name__}(model='{self.model}', initialized={self.is_initialized})" 