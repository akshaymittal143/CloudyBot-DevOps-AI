import os
"""
This module provides functionality to interact with the OpenAI API for generating
responses based on user input and optional chat history.
Functions:
    get_openai_response(question, chat_history=None, model="gpt-3.5-turbo"):
        Generates a response using the OpenAI API based on the provided question
        and optional chat history.
Dependencies:
    - os: For accessing environment variables.
    - openai.OpenAI: For interacting with the OpenAI API.
    - dotenv.load_dotenv: For loading environment variables from a .env file.
Environment Variables:
    - OPENAI_API_KEY: The API key required to authenticate with the OpenAI API.
"""
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_response(question, chat_history=None, model="gpt-3.5-turbo"):
    """Generate response using OpenAI API."""
    try:
        # Prepare messages
        messages = []
        
        # Add chat history if provided
        if chat_history:
            messages.extend([
                {"role": msg["role"], "content": msg["content"]}
                for msg in chat_history[-5:]  # Include last 5 messages for context
            ])
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            n=1,
            stop=None,
        )
        
        # Extract and return the response text
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"