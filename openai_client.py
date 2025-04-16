# openai_client.py
import os
import openai

# Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")


# This module provides a function to interact with OpenAI's Chat Completion API.
# It requires the OpenAI API key to be set in the environment variable "OPENAI_API_KEY".
def ask_openai(question: str) -> str:
    """
    Send a question to the OpenAI Chat Completion API and return the response text.
    """
    try:
        # Call the ChatCompletion endpoint with a user prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another model name
            messages=[{"role": "user", "content": question}]
        )
        # Extract the assistant's answer from the response
        answer = response["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        # Basic error handling: return an error message as answer
        return f"Error: {e}"
