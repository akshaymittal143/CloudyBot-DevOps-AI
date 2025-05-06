"""Test cases for bot functionality."""
import pytest
from bot import ask_bot

def test_ask_bot_with_invalid_provider():
    """Test ask_bot with invalid provider."""
    response = ask_bot("test question", provider="INVALID")
    assert "Error: Unknown provider" in response

def test_ask_bot_with_empty_question():
    """Test ask_bot with empty question."""
    response = ask_bot("")
    assert "Error" in response
