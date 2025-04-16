
import os
"""
This module provides functionality to interact with AI models from different providers.

Functions:
    ask_bot(question, chat_history=None, provider="OPENAI", model=None):
        Routes a user's question to the appropriate AI provider and returns the response.

Dependencies:
    - os: For interacting with the operating system.
    - dotenv.load_dotenv: For loading environment variables from a .env file.
    - get_openai_response: A function from the openai_client module to interact with OpenAI models.
    - get_hf_response: A function from the hf_client module to interact with Hugging Face models.
"""
from dotenv import load_dotenv
from openai_client import get_openai_response
from hf_client import get_hf_response

# Load environment variables
load_dotenv()

def ask_bot(question, chat_history=None, provider="OPENAI", model=None):
    """Route the question to appropriate AI provider and return response."""
    try:
        if provider == "OPENAI":
            return get_openai_response(question, chat_history, model)
        elif provider == "HUGGINGFACE":
            return get_hf_response(question, chat_history)
        else:
            return f"Error: Unknown provider {provider}"
    except Exception as e:
        return f"Error: {str(e)}"