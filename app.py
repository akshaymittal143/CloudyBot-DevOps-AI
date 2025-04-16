# app.py
import streamlit as st
from bot import get_bot_response

# Set page title and layout
st.set_page_config(page_title="CloudyBot - DevOps Chatbot", layout="centered")

st.title("🤖 CloudyBot - AI-powered DevOps Chatbot")

# Instructions or description
st.write("Ask any DevOps or cloud-related question, and the AI-powered CloudyBot will assist you.")

# Maintain chat history in Streamlit session state
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of (role, message) tuples

# Create a text input widget for the user's query
user_input = st.text_input("Your Question:", "", placeholder="e.g., How do I set up CI/CD on AWS?")

# When the user submits a question (press Enter), handle it
if user_input:
    # Display the user's question in the chat history
    st.session_state["history"].append(("user", user_input))
    # Get the AI bot response
    answer = get_bot_response(user_input)
    # Add the bot's answer to the chat history
    st.session_state["history"].append(("bot", answer))
    # Clear the input box for next question
    st.rerun()

# Display the conversation history
for role, msg in st.session_state["history"]:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**CloudyBot:** {msg}")
