"""pytest configuration and fixtures."""
import pytest
from config import Config

@pytest.fixture
def mock_config():
    """Provide test configuration."""
    Config.OPENAI_API_KEY = "test_key"
    Config.MODEL_PROVIDER = "OPENAI"
    Config.MAX_TOKENS = 100
    return Config
