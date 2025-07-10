"""
Hugging Face client implementation for CloudyBot.

This module provides Hugging Face model integration with proper async support,
comprehensive error handling, and efficient model management.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import threading

# Set environment variables before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

from cloudybot.clients.base import AIClient, ChatHistory, Message
from cloudybot.core.exceptions import ClientError, ModelLoadError, ConfigurationError
from cloudybot.config.settings import HuggingFaceSettings


class HuggingFaceClient(AIClient):
    """
    Hugging Face client implementation with async support and comprehensive error handling.
    """
    
    def __init__(
        self,
        settings: HuggingFaceSettings,
        **kwargs
    ):
        """
        Initialize Hugging Face client.
        
        Args:
            settings: Hugging Face configuration settings
            **kwargs: Additional configuration options
        """
        super().__init__(settings.model, **kwargs)
        self.settings = settings
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._device: str = "cpu"
        self._model_lock = threading.Lock()
        
        # Determine device
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Set up the compute device for the model."""
        if self.settings.device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
                self.logger.info("CUDA available, using GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
                self.logger.info("MPS available, using Apple Silicon GPU")
            else:
                self._device = "cpu"
                self.logger.info("Using CPU for inference")
        else:
            self._device = self.settings.device
            self.logger.info(f"Using specified device: {self._device}")
    
    async def initialize(self) -> None:
        """Initialize the Hugging Face client and load the model."""
        try:
            self.logger.info(f"Initializing Hugging Face client with model: {self.model}")
            
            # Load model in a separate thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync
            )
            
            self._is_initialized = True
            self.logger.info("Hugging Face client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Hugging Face client: {e}")
            raise ClientError(
                f"Failed to initialize Hugging Face client: {e}",
                provider="HUGGINGFACE",
                model=self.model
            )
    
    def _load_model_sync(self) -> None:
        """Load the model and tokenizer synchronously."""
        with self._model_lock:
            if self._model is not None and self._tokenizer is not None:
                self.logger.debug("Model already loaded, skipping")
                return
            
            try:
                self.logger.info(f"Loading Hugging Face model: {self.model}")
                
                # Set up cache directory if specified
                cache_dir = self.settings.cache_dir
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                
                # Load tokenizer
                self.logger.debug("Loading tokenizer...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model,
                    cache_dir=cache_dir,
                    token=self.settings.api_token
                )
                
                # Try to determine the model type and load appropriate model class
                self.logger.debug("Loading model...")
                try:
                    # Try seq2seq model first (like T5, BART)
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model,
                        cache_dir=cache_dir,
                        token=self.settings.api_token,
                        torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.logger.debug("Loaded as seq2seq model")
                except Exception:
                    # Fallback to causal LM (like GPT)
                    self.logger.debug("Trying as causal LM model...")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model,
                        cache_dir=cache_dir,
                        token=self.settings.api_token,
                        torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.logger.debug("Loaded as causal LM model")
                
                # Move model to device
                if self._device != "cpu":
                    self._model = self._model.to(self._device)
                
                # Set model to evaluation mode
                self._model.eval()
                
                self.logger.info(f"Model loaded successfully on {self._device}")
                
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                raise ModelLoadError(
                    f"Failed to load model '{self.model}': {e}",
                    model_path=self.model,
                    provider="HUGGINGFACE"
                )
    
    def is_available(self) -> bool:
        """Check if the Hugging Face client is available."""
        return (
            self._is_initialized 
            and self._model is not None 
            and self._tokenizer is not None
        )
    
    async def generate_response(
        self,
        question: str,
        chat_history: Optional[ChatHistory] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using Hugging Face model.
        
        Args:
            question: User's question
            chat_history: Optional chat history for context
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            ClientError: If client is not available or initialized
        """
        if not self.is_available():
            raise ClientError(
                "Hugging Face client is not available. Please initialize first.",
                provider="HUGGINGFACE",
                model=self.model
            )
        
        try:
            # Run generation in executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_response_sync, question, chat_history, kwargs
            )
            return response
            
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            
            self.logger.error(f"Error generating response: {e}")
            raise ClientError(
                f"Error generating response: {e}",
                provider="HUGGINGFACE",
                model=self.model
            )
    
    def _generate_response_sync(
        self,
        question: str,
        chat_history: Optional[ChatHistory] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response synchronously.
        
        Args:
            question: User's question
            chat_history: Optional chat history
            generation_kwargs: Generation parameters
            
        Returns:
            Generated response
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        
        try:
            with self._model_lock:
                # Prepare input text
                input_text = self._prepare_input(question, chat_history)
                
                # Tokenize input
                inputs = self._tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                
                # Move inputs to device
                if self._device != "cpu":
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                # Extract generation parameters
                temperature = generation_kwargs.get('temperature', self.settings.temperature)
                max_length = generation_kwargs.get('max_length', self.settings.max_length)
                
                # Generate response
                with torch.no_grad():
                    if hasattr(self._model, 'generate'):
                        outputs = self._model.generate(
                            **inputs,
                            max_length=max_length,
                            do_sample=True,
                            temperature=max(temperature, 0.1),  # Prevent temperature=0
                            top_p=0.9,
                            num_return_sequences=1,
                            pad_token_id=self._tokenizer.eos_token_id,
                            early_stopping=True
                        )
                    else:
                        # For models without generate method
                        outputs = self._model(**inputs)
                        outputs = outputs.logits.argmax(dim=-1)
                
                # Decode response
                if isinstance(outputs, torch.Tensor):
                    if len(outputs.shape) > 1:
                        output_ids = outputs[0]
                    else:
                        output_ids = outputs
                else:
                    output_ids = outputs[0]
                
                response = self._tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Clean up response
                response = self._clean_response(response, input_text)
                
                self.logger.debug(f"Generated response of length: {len(response)}")
                return response
                
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("CUDA out of memory error")
            raise ClientError(
                "GPU out of memory. Try using a smaller model or CPU mode.",
                provider="HUGGINGFACE",
                model=self.model,
                details={"error_type": "out_of_memory"}
            )
        except Exception as e:
            self.logger.error(f"Error in sync response generation: {e}")
            raise ClientError(
                f"Error generating response: {e}",
                provider="HUGGINGFACE",
                model=self.model
            )
    
    def _prepare_input(
        self,
        question: str,
        chat_history: Optional[ChatHistory] = None
    ) -> str:
        """
        Prepare input text for the model.
        
        Args:
            question: User's question
            chat_history: Optional chat history
            
        Returns:
            Formatted input text
        """
        if not chat_history or len(chat_history) == 0:
            return f"Question: {question}\nAnswer:"
        
        # Build context from chat history
        context_parts = []
        recent_messages = chat_history.get_messages(limit=3)  # Limit to avoid token limits
        
        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"Question: {msg.content}")
            elif msg.role == "assistant":
                context_parts.append(f"Answer: {msg.content}")
        
        # Add current question
        context_parts.append(f"Question: {question}")
        context_parts.append("Answer:")
        
        return "\n".join(context_parts)
    
    def _clean_response(self, response: str, input_text: str) -> str:
        """
        Clean and format the model response.
        
        Args:
            response: Raw model response
            input_text: Original input text
            
        Returns:
            Cleaned response
        """
        # Remove input text from response if present
        if input_text in response:
            response = response.replace(input_text, "").strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Answer:", "Response:", "Output:", "A:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove excessive whitespace and newlines
        response = " ".join(response.split())
        
        # Ensure response is not empty
        if not response:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return response
    
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
            # Try to initialize if not already done
            if not self._is_initialized:
                self._load_model_sync()
                self._is_initialized = True
        
        # Convert dict history to ChatHistory if provided
        history = None
        if chat_history:
            history = ChatHistory.from_dict_list(chat_history)
        
        return self._generate_response_sync(question, history, kwargs)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        if self._model is not None:
            try:
                # Get model parameters count
                num_params = sum(p.numel() for p in self._model.parameters())
                
                # Get model size in MB
                model_size_mb = sum(
                    p.numel() * p.element_size() for p in self._model.parameters()
                ) / (1024 * 1024)
                
                base_info.update({
                    "device": self._device,
                    "num_parameters": num_params,
                    "model_size_mb": round(model_size_mb, 2),
                    "model_type": type(self._model).__name__,
                    "tokenizer_type": type(self._tokenizer).__name__ if self._tokenizer else None
                })
            except Exception as e:
                self.logger.warning(f"Could not get detailed model info: {e}")
        
        return base_info
    
    def clear_cache(self) -> None:
        """Clear GPU cache if using CUDA."""
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cleared CUDA cache")
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                self.logger.info("Model unloaded from memory")
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
                self.logger.info("Tokenizer unloaded from memory")
            
            self.clear_cache()
            self._is_initialized = False
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current client settings.
        
        Returns:
            Dictionary of current settings
        """
        return {
            "model": self.model,
            "device": self._device,
            "temperature": self.settings.temperature,
            "max_length": self.settings.max_length,
            "cache_dir": self.settings.cache_dir,
            "provider": "HUGGINGFACE"
        }
    
    def __del__(self):
        """Cleanup when the client is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass  # Ignore errors during cleanup 