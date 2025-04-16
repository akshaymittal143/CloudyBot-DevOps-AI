import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def get_openai_response(prompt, conversation_history=None, model=None):
    """
    Get a response from OpenAI's API.
    
    Args:
        prompt (str): The user query
        conversation_history (list): Previous conversation messages
        model (str, optional): The OpenAI model to use
        
    Returns:
        str: The AI response
    """
    if not openai.api_key:
        return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
    
    if model is None:
        model = DEFAULT_MODEL
    
    # System message for DevOps focus
    system_message = {
        "role": "system", 
        "content": (
            "You are CloudyBot, an AI assistant specializing in DevOps, cloud infrastructure, "
            "and software development practices. Provide helpful, accurate, and concise responses "
            "to technical questions. Include code examples when relevant. If you're unsure about "
            "something, acknowledge it rather than providing potentially incorrect information."
        )
    }
    
    # Prepare the messages array
    messages = [system_message]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Call the API
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and return the response text
        return response.choices[0].message["content"].strip()
    
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"