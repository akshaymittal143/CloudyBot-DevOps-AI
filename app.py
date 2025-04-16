import os
import streamlit as st
from dotenv import load_dotenv
from bot import ask_bot

# Load environment variables
load_dotenv()

# Get default provider and models
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "OPENAI").upper()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")

# Page configuration
st.set_page_config(
    page_title="CloudyBot - DevOps Assistant",
    page_icon="☁️",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    # Title and description
    st.title("☁️ CloudyBot: DevOps Assistant")
    st.markdown(
        """
        CloudyBot is an AI-powered assistant for DevOps and cloud-related questions.
        Ask about Kubernetes, Docker, CI/CD, cloud providers, or any DevOps practices!
        """
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Model provider selection
        provider = st.radio(
            "Select AI Provider:",
            options=["OpenAI", "Hugging Face"],
            index=0 if DEFAULT_PROVIDER == "OPENAI" else 1
        )
        
        # Model selection based on provider
        if provider == "OpenAI":
            model = st.text_input("OpenAI Model:", value=OPENAI_MODEL)
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ OpenAI API key not set. Add it to your .env file.")
        else:  # Hugging Face
            model = st.text_input("Hugging Face Model:", value=HUGGINGFACE_MODEL)
            st.info("ℹ️ First response may be slow while loading the model.")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()  # Updated from experimental_rerun
    
        # Show example queries
        st.header("Example Queries")
        examples = [
            "How do I restart a Kubernetes pod?",
            "Explain blue-green deployment.",
            "How to debug Docker container failures?",
            "What's the difference between Docker and Kubernetes?",
            "How can I automate AWS infrastructure deployment?"
        ]
        
        # Handle example queries
        for example in examples:
            if st.button(example):
                st.session_state.messages.append({"role": "user", "content": example})
                with st.spinner("CloudyBot is thinking..."):
                    response = ask_bot(
                        example, 
                        st.session_state.messages[:-1], 
                        provider.upper().replace(" ", ""),
                        model
                    )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()  # Updated from experimental_rerun
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with modern API
    if prompt := st.chat_input("Ask CloudyBot a DevOps question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("CloudyBot is thinking..."):
                response = ask_bot(
                    prompt, 
                    st.session_state.messages[:-1], 
                    provider.upper().replace(" ", ""),
                    model
                )
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()  # Refresh the chat display

if __name__ == "__main__":
    main()