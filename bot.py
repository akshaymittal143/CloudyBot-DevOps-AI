# bot.py
import os
from openai_client import ask_openai
from hf_client import ask_hf

# Read setting from environment to decide which backend to use
use_openai = os.getenv("USE_OPENAI", "true").lower() == "true"

def get_bot_response(user_query: str) -> str:
    """
    Determine which backend to use and get the response for the user's query.
    """
    if use_openai:
        # Use OpenAI API for response
        return ask_openai(user_query)
    else:
        # Use Hugging Face local model for response
        return ask_hf(user_query)

