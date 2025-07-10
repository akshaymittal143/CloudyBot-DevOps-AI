import pytest
from cloudybot.config.settings import get_settings, ConfigurationError

def test_settings_load():
    print("\n[TEST] Attempting to load settings...")
    try:
        settings = get_settings(reload=True)
        print("[TEST] Settings loaded successfully:")
        print(settings)
    except ConfigurationError as e:
        print("[TEST] ConfigurationError:", e)
        assert False, f"ConfigurationError: {e}"
    except Exception as e:
        print("[TEST] Unexpected error:", e)
        assert False, f"Unexpected error: {e}"

if __name__ == "__main__":
    test_settings_load() 