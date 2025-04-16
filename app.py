"""
A Streamlit-based web application for a DevOps AI assistant called CloudyBot.
This application provides an interactive chat interface where users can ask questions
about DevOps, cloud computing, and related technologies. It supports multiple AI providers
(OpenAI and Hugging Face) and allows users to customize model settings.
Environment Variables:
    MODEL_PROVIDER (str): Default AI provider ('OPENAI' or 'HUGGINGFACE')
    OPENAI_MODEL (str): Default OpenAI model name
    HUGGINGFACE_MODEL (str): Default Hugging Face model name
    OPENAI_API_KEY (str): API key for OpenAI services
Features:
    - Interactive chat interface
    - Multiple AI provider support (OpenAI and Hugging Face)
    - Customizable model selection
    - Chat history management
    - Example queries for quick testing
    - Clear chat functionality
    - Real-time response generation
    - Responsive layout with sidebar settings
Dependencies:
    - streamlit
    - python-dotenv
    - Custom bot module for AI interaction
Usage:
    Run the application using:
    `streamlit run app.py`
Note:
    Requires appropriate API keys and environment variables to be set in a .env file
    for full functionality.
"""
# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from bot import ask_bot

# Page configuration with custom styling
st.set_page_config(
    page_title="CloudyBot - DevOps Assistant",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Helper function to get configuration value
def get_config(key, default_value):
    """Get configuration from Streamlit secrets or environment variables"""
    try:
        return st.secrets.get(key, os.getenv(key, default_value))
    except:
        return os.getenv(key, default_value)

# Get environment variables with fallbacks
DEFAULT_PROVIDER = st.secrets.get("MODEL_PROVIDER", os.getenv("MODEL_PROVIDER", "OPENAI")).upper()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
HUGGINGFACE_MODEL = st.secrets.get("HUGGINGFACE_MODEL", os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base"))

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        background-color: #007fff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.title("☁️ CloudyBot: DevOps Assistant")
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
        CloudyBot is an AI-powered assistant for DevOps and cloud-related questions.
        Ask about Kubernetes, Docker, CI/CD, cloud providers, or any DevOps practices!
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("""
            <div style='background-color: #e9ecef; padding: 1rem; border-radius: 5px;'>
            <h2 style='color: #007fff; margin-bottom: 1rem;'>Settings</h2>
            </div>
        """, unsafe_allow_html=True)
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