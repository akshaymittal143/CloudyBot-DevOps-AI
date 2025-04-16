import os
from dotenv import load_dotenv
from openai_client import get_openai_response
from hf_client import get_hf_response

# Load environment variables
load_dotenv()

# Get model provider from environment
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "OPENAI").upper()

def ask_bot(prompt, conversation_history=None, provider=None, model=None):
    """
    Get a response from the bot based on the configured provider.
    
    Args:
        prompt (str): The user query
        conversation_history (list, optional): Previous conversation messages
        provider (str, optional): The model provider to use (OPENAI or HUGGINGFACE)
        model (str, optional): Specific model to use
        
    Returns:
        str: The AI response
    """
    # Use default provider if not specified
    if provider is None:
        provider = DEFAULT_PROVIDER
    
    provider = provider.upper()
    
    # Route to appropriate provider
    if provider == "OPENAI":
        return get_openai_response(prompt, conversation_history, model)
    
    elif provider == "HUGGINGFACE":
        return get_hf_response(prompt, conversation_history, model)
    
    else:
        return f"Error: Unknown provider '{provider}'. Please use 'OPENAI' or 'HUGGINGFACE'."