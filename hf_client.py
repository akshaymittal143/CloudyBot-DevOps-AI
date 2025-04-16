"""
hf_client.py
This module provides functionality to load a Hugging Face model and tokenizer, 
and to generate responses using the loaded model. It supports environment 
variable configuration for the model path and Hugging Face API token.
Functions:
----------
- load_model(model_path=None):
    Loads the Hugging Face model and tokenizer. If the model and tokenizer 
    are already loaded, it returns the existing instances. The model path 
    can be specified as an argument or defaults to the value of the 
    `HUGGINGFACE_MODEL` environment variable.
    Parameters:
    - model_path (str, optional): Path or name of the Hugging Face model to load.
    Returns:
    - tuple: A tuple containing the loaded model and tokenizer, or (None, None) 
      if an error occurs during loading.
- get_hf_response(question, chat_history=None, temperature=0.7):
    Generates a response to a given question using the loaded Hugging Face model. 
    Optionally includes chat history in the input for context.
    Parameters:
    - question (str): The input question or prompt for the model.
    - chat_history (list of dict, optional): A list of dictionaries representing 
      the chat history. Each dictionary should have keys 'role' and 'content'.
    - temperature (float, optional): Sampling temperature for response generation. 
      Higher values result in more diverse outputs. Default is 0.7.
    Returns:
    - str: The generated response from the model, or an error message if the 
      model is not loaded or an error occurs during generation.
"""


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this before other imports

import numpy as np  # Import numpy before torch
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define model path from environment or use default
DEFAULT_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model(model_path=None):
    """Load the Hugging Face model and tokenizer."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        if model_path is None:
            model_path = DEFAULT_MODEL
        
        try:
            print(f"Loading model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            print(f"Model loaded successfully: {model_path}")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None
    
    return model, tokenizer

def get_hf_response(question, chat_history=None, temperature=0.7):
    """Generate response using Hugging Face model."""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly"
    
    try:
        input_text = question
        if chat_history:
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
            input_text = f"{history_text}\nuser: {question}"
        
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"