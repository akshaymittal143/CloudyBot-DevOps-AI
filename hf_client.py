# hf_client.py
from transformers import pipeline

# Initialize a text-generation pipeline with a small model
# This will download the model weights on first run if not already available.
generator = pipeline("text-generation", model="gpt2")  # You can replace "gpt2" with another model


def ask_hf(question: str) -> str:
    """
    Generate a response to the question using a local Hugging Face model.
    """
    try:
        # Use the pipeline to generate text. We limit max_length to avoid very long responses.
        results = generator(question, max_length=100, num_return_sequences=1)
        # Extract the generated text from results
        answer = results[0]["generated_text"]
        return answer
    except Exception as e:
        return f"Error: {e}"
