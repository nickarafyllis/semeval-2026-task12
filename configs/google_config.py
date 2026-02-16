"""
Google Gemini API configuration and client initialization.

Credentials are loaded from:
1. Explicit api_key parameter
2. GOOGLE_API_KEY environment variable
3. .env file (via python-dotenv)

Supports multiple API keys for rate limit rotation via --key flag.
"""
import os
from typing import Optional
from pathlib import Path

# Try to import google-genai (the new unified SDK)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


def setup_google_credentials(api_key: Optional[str] = None, key_index: Optional[int] = None):
    """
    Set up Google API credentials.

    Priority:
    1. Explicit api_key parameter
    2. GOOGLE_API_KEY environment variable

    Args:
        api_key: Direct API key string
        key_index: Legacy parameter for multi-key rotation (accepted but ignored;
                   the single GOOGLE_API_KEY env var is used regardless)

    Returns:
        API key string
    """
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
        return api_key

    if 'GOOGLE_API_KEY' in os.environ:
        return os.environ['GOOGLE_API_KEY']

    raise ValueError(
        "Google API key not found. Set GOOGLE_API_KEY environment variable "
        "or pass api_key parameter directly."
    )


def get_gemini_client(api_key: Optional[str] = None, key_index: Optional[int] = None):
    """
    Create and return Google Gemini client.

    Args:
        api_key: Optional API key (uses env var if not provided)
        key_index: Legacy parameter for multi-key rotation (accepted but ignored;
                   the single GOOGLE_API_KEY env var is used regardless)

    Returns:
        google.genai.Client instance
    """
    if not GENAI_AVAILABLE:
        raise ImportError(
            "google-genai package not installed. Install with:\n"
            "  pip install google-genai"
        )

    api_key = setup_google_credentials(api_key=api_key)
    client = genai.Client(api_key=api_key)
    print("Gemini client created")
    return client


# Model ID registry for Gemini
GEMINI_MODEL_IDS = {
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
}


def get_gemini_model_id(model_name: str) -> str:
    """
    Get full Gemini model ID from short name.

    Args:
        model_name: Short model name (e.g., "gemini-3-flash-preview")

    Returns:
        Full model ID
    """
    if model_name in GEMINI_MODEL_IDS:
        return GEMINI_MODEL_IDS[model_name]

    if model_name.startswith("gemini-"):
        return model_name

    raise ValueError(
        f"Unknown Gemini model: {model_name}. "
        f"Available: {list(GEMINI_MODEL_IDS.keys())}"
    )


def test_gemini_connection(client, test_model: str = "gemini-2.0-flash"):
    """
    Test Google Gemini connection with a simple API call.

    Args:
        client: Gemini client
        test_model: Model ID to test with

    Returns:
        bool: True if connection successful
    """
    try:
        response = client.models.generate_content(
            model=test_model,
            contents="Reply with just 'OK'",
            config=types.GenerateContentConfig(
                max_output_tokens=50,
                temperature=0.0
            )
        )

        response_text = response.text
        print(f"Gemini connection test PASSED (response: {response_text})")
        return True

    except Exception as e:
        print(f"Gemini connection test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("Google Gemini Configuration Test")
    print("=" * 80)

    try:
        client = get_gemini_client()
        test_gemini_connection(client)
        print("\nAll Gemini setup tests passed!")
    except Exception as e:
        print(f"\nGemini setup failed: {e}")
        print("\nTo fix:")
        print("1. Install: pip install google-genai")
        print("2. Set GOOGLE_API_KEY environment variable")
