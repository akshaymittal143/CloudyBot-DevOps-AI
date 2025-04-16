import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define model path from environment or use default
DEFAULT_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Optional for some models

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model(model_path=None):
    """
    Load the Hugging Face model and tokenizer.
    
    Args:
        model_path (str, optional): Path to the model on Hugging Face Hub
        
    Returns:
        tuple: (model, tokenizer)
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        if model_path is None:
            model_path = DEFAULT_MODEL
        
        try:
            print(f"Loading model: {model_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                token=HF_TOKEN if HF_TOKEN else None
            )
            
            # Load model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                token=HF_TOKEN if HF_TOKEN else None,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print(f"Model loaded successfully: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None
    
    return model, tokenizer

def get_hf_response(prompt, conversation_history=None, model_path=None):
    """
    Get a response from a Hugging Face model.
    
    Args:
        prompt (str): The user query
        conversation_history (list, optional): Previous conversation messages
        model_path (str, optional): Path to the model on Hugging Face Hub
        
    Returns:
        str: The AI response
    """
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)
    
    if model is None or tokenizer is None:
        return "Error: Failed to load the Hugging Face model."
    
    # Prepare context with history if provided
    context = ""
    if conversation_history:
        # Format conversation history
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                context += f"User: {content}\n"
            elif role == "assistant":
                context += f"CloudyBot: {content}\n"
    
    # Prepare full prompt including DevOps context
    full_prompt = (
        "You are CloudyBot, a helpful DevOps assistant. "
        f"{context}"
        f"User: {prompt}\n"
        "CloudyBot:"
    )
    
    try:
        # Tokenize the input
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate response
        outputs = model.generate(
            **inputs, 
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "CloudyBot:" in response:
            response = response.split("CloudyBot:")[-1].strip()
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"