import os
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