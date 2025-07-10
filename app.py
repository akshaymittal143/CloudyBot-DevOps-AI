"""
CloudyBot Streamlit Application.

A modern, professional Streamlit interface for the CloudyBot AI-powered DevOps assistant.
This application provides an interactive chat interface with multiple AI providers,
comprehensive error handling, and professional UI design.
"""

import asyncio
import streamlit as st
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import os

# Add the cloudybot package to the path
sys.path.insert(0, str(Path(__file__).parent))

from cloudybot import CloudyBot, __version__
from cloudybot.config.settings import get_settings, ModelProvider
from cloudybot.config.logging import setup_logging, configure_third_party_loggers
from cloudybot.core.exceptions import CloudyBotError, ClientError, ConfigurationError


# Configure logging and reduce third-party noise
configure_third_party_loggers()


class StreamlitCloudyBotApp:
    """Main Streamlit application class for CloudyBot."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_logging()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        # Load settings for UI configuration
        try:
            settings = get_settings(secrets=dict(st.secrets) if hasattr(st, 'secrets') else None)
            ui_settings = settings.ui
        except Exception:
            # Fallback configuration if settings fail to load
            ui_settings = type('UI', (), {
                'page_title': 'CloudyBot - DevOps Assistant',
                'page_icon': '☁️',
                'layout': 'wide',
                'sidebar_state': 'expanded'
            })
        
        st.set_page_config(
            page_title=ui_settings.page_title,
            page_icon=ui_settings.page_icon,
            layout=ui_settings.layout,
            initial_sidebar_state=ui_settings.sidebar_state
        )
    
    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        # Initialize bot instance
        if "bot" not in st.session_state:
            st.session_state.bot = None
            st.session_state.bot_status = "not_initialized"
            st.session_state.initialization_error = None
        
        # Initialize chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize settings
        if "settings" not in st.session_state:
            try:
                st.session_state.settings = get_settings(
                    secrets=dict(st.secrets) if hasattr(st, 'secrets') else None
                )
            except Exception as e:
                st.session_state.settings = None
                st.session_state.settings_error = str(e)
        
        # UI state
        if "current_provider" not in st.session_state:
            st.session_state.current_provider = None
        
        if "show_settings" not in st.session_state:
            st.session_state.show_settings = False
    
    def setup_logging(self) -> None:
        """Set up application logging."""
        try:
            if st.session_state.settings:
                setup_logging(st.session_state.settings.logging)
        except Exception:
            # Continue without logging if setup fails
            pass
    
    async def initialize_bot(self) -> None:
        """Initialize the CloudyBot instance."""
        try:
            if st.session_state.settings is None:
                raise ConfigurationError("Settings could not be loaded")
            
            st.session_state.bot_status = "initializing"
            
            # Create bot instance
            bot = CloudyBot(settings=st.session_state.settings, auto_initialize=False)
            await bot.initialize()
            
            st.session_state.bot = bot
            st.session_state.bot_status = "ready"
            st.session_state.current_provider = bot._current_provider
            st.session_state.initialization_error = None
            
        except Exception as e:
            st.session_state.bot_status = "error"
            st.session_state.initialization_error = str(e)
            st.session_state.bot = None
    
    def render_header(self) -> None:
        """Render the application header."""
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem 0;'>
                <h1 style='color: #007fff; margin-bottom: 0.5rem;'>
                    ☁️ CloudyBot: DevOps Assistant
                </h1>
                <p style='color: #666; font-size: 1.1rem; margin-bottom: 1rem;'>
                    AI-powered assistance for DevOps, cloud infrastructure, and automation
                </p>
                <p style='color: #888; font-size: 0.9rem;'>
                    Version {__version__} | Status: {st.session_state.bot_status.replace('_', ' ').title()}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show initialization status
        if st.session_state.bot_status == "initializing":
            st.info("🔄 Initializing CloudyBot... Please wait.")
        elif st.session_state.bot_status == "error":
            st.error(f"❌ Initialization failed: {st.session_state.initialization_error}")
            if st.button("🔄 Retry Initialization"):
                asyncio.run(self.initialize_bot())
                st.rerun()
        elif st.session_state.bot_status == "ready":
            st.success("✅ CloudyBot is ready to assist you!")
    
    def render_sidebar(self) -> None:
        """Render the sidebar with settings and controls."""
        with st.sidebar:
            st.markdown("## ⚙️ Settings")
            
            # Bot status
            if st.session_state.bot:
                status_info = st.session_state.bot.get_status()
                
                st.markdown("### 🤖 Bot Status")
                st.json({
                    "Status": status_info["status"],
                    "Provider": status_info["current_provider"],
                    "Available": status_info["available_providers"],
                    "Chat History": status_info["chat_history_length"]
                })
                
                # Provider switching
                if len(status_info["available_providers"]) > 1:
                    st.markdown("### 🔄 Switch Provider")
                    current_provider = status_info["current_provider"]
                    available_providers = status_info["available_providers"]
                    
                    new_provider = st.selectbox(
                        "Select AI Provider:",
                        options=available_providers,
                        index=available_providers.index(current_provider) if current_provider in available_providers else 0,
                        key="provider_selector"
                    )
                    
                    if new_provider != current_provider:
                        try:
                            st.session_state.bot.switch_provider(new_provider)
                            st.session_state.current_provider = new_provider
                            st.success(f"Switched to {new_provider}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to switch provider: {e}")
            
            # Chat controls
            st.markdown("### 💬 Chat Controls")
            
            if st.button("🗑️ Clear Chat History", key="clear_chat"):
                st.session_state.messages = []
                if st.session_state.bot:
                    st.session_state.bot.clear_history()
                st.success("Chat history cleared!")
                st.rerun()
            
            # Example queries
            st.markdown("### 💡 Example Questions")
            
            if st.session_state.bot:
                examples = st.session_state.bot.get_examples()
            else:
                examples = [
                    "How do I restart a Kubernetes pod?",
                    "Explain blue-green deployment.",
                    "How to debug Docker container failures?",
                    "What's the difference between Docker and Kubernetes?",
                    "How can I automate AWS infrastructure deployment?"
                ]
            
            for i, example in enumerate(examples):
                if st.button(example, key=f"example_{i}"):
                    self.add_message("user", example)
                    self.process_user_input(example)
                    st.rerun()
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                st.markdown("#### Debug Information")
                if st.session_state.bot:
                    if st.button("🏥 Health Check"):
                        try:
                            health = asyncio.run(st.session_state.bot.health_check())
                            st.json(health)
                        except Exception as e:
                            st.error(f"Health check failed: {e}")
                
                if st.button("📊 Show Full Status"):
                    if st.session_state.bot:
                        status = st.session_state.bot.get_status()
                        st.json(status)
                    else:
                        st.warning("Bot not initialized")
    
    def render_chat_interface(self) -> None:
        """Render the main chat interface."""
        st.markdown("## 💬 Chat with CloudyBot")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask CloudyBot about DevOps, cloud, or automation..."):
            self.add_message("user", prompt)
            self.process_user_input(prompt)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        st.session_state.messages.append({"role": role, "content": content})
    
    def process_user_input(self, user_input: str) -> None:
        """Process user input and generate a response."""
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("CloudyBot is thinking..."):
                try:
                    if st.session_state.bot_status != "ready":
                        response = "I'm not ready yet. Please wait for initialization to complete."
                    else:
                        # Use the refactored bot
                        response = st.session_state.bot.ask_sync(user_input)
                    
                    st.markdown(response)
                    self.add_message("assistant", response)
                    
                except CloudyBotError as e:
                    error_msg = f"CloudyBot Error: {e}"
                    st.error(error_msg)
                    self.add_message("assistant", error_msg)
                
                except ClientError as e:
                    error_msg = f"AI Client Error: {e}"
                    st.error(error_msg)
                    self.add_message("assistant", error_msg)
                
                except Exception as e:
                    error_msg = f"Unexpected error: {e}"
                    st.error(error_msg)
                    self.add_message("assistant", error_msg)
    
    def render_footer(self) -> None:
        """Render the application footer."""
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #666; padding: 1rem;'>
                <p>
                    Made with ❤️ by <a href="https://github.com/akshaymittal143" target="_blank">Akshay Mittal</a> | 
                    <a href="https://github.com/akshaymittal143/CloudyBot-DevOps-AI" target="_blank">GitHub</a> | 
                    Licensed under Apache 2.0
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    def apply_custom_css(self) -> None:
        """Apply custom CSS styling."""
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
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: background-color 0.3s ease;
            }
            
            .stButton button:hover {
                background-color: #0056b3;
            }
            
            .stSelectbox > div > div {
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            
            .stTextInput > div > div > input {
                border-radius: 5px;
                border: 2px solid #e9ecef;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #007fff;
                box-shadow: 0 0 0 0.2rem rgba(0, 127, 255, 0.25);
            }
            
            .stMarkdown h1 {
                color: #007fff;
            }
            
            .stMarkdown h2 {
                color: #495057;
                border-bottom: 2px solid #007fff;
                padding-bottom: 0.5rem;
            }
            
            .stMarkdown h3 {
                color: #6c757d;
            }
            
            .stSuccess {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
            }
            
            .stError {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
            }
            
            .stInfo {
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def run(self) -> None:
        """Run the Streamlit application."""
        # Apply custom styling
        self.apply_custom_css()
        
        # Initialize bot if not already done
        if st.session_state.bot_status == "not_initialized":
            asyncio.run(self.initialize_bot())
            st.rerun()
        
        # Render components
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()
        self.render_footer()


# Compatibility function for the old interface
def ask_bot(question: str, chat_history=None, provider="OPENAI", model=None) -> str:
    """
    Backward compatibility function for the old ask_bot interface.
    
    Args:
        question: User's question
        chat_history: Optional chat history
        provider: AI provider to use
        model: Model name
        
    Returns:
        AI response
    """
    try:
        # Initialize bot if needed
        if st.session_state.bot is None:
            settings = get_settings(secrets=dict(st.secrets) if hasattr(st, 'secrets') else None)
            bot = CloudyBot(settings=settings, auto_initialize=False)
            asyncio.run(bot.initialize())
            st.session_state.bot = bot
        
        # Use the new interface
        return st.session_state.bot.ask_sync(
            question=question,
            provider=provider,
            model=model,
            use_history=bool(chat_history)
        )
    
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main entry point for the Streamlit application."""
    app = StreamlitCloudyBotApp()
    app.run()


if __name__ == "__main__":
    print("ENV DUMP (from app.py):")
    for k, v in os.environ.items():
        if "HUGGINGFACE" in k or "MODEL_PROVIDER" in k or "MAX_CHAT_HISTORY" in k or "LOG_LEVEL" in k or "DEBUG" in k:
            print(f"{k}={v}")
    main()