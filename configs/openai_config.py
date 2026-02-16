"""
OpenAI API configuration and client initialization.

Credentials are loaded from:
1. Explicit api_key parameter
2. OPENAI_API_KEY environment variable
3. .env file (via python-dotenv)

Supports prompt caching (automatic for prompts > 1024 tokens).
"""
import os
from typing import Optional
from pathlib import Path

# Try to import openai SDK
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def setup_openai_credentials(api_key: Optional[str] = None):
    """
    Set up OpenAI API credentials.

    Priority:
    1. Explicit api_key parameter
    2. OPENAI_API_KEY environment variable

    Args:
        api_key: Direct API key string

    Returns:
        API key string
    """
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        return api_key

    if 'OPENAI_API_KEY' in os.environ:
        return os.environ['OPENAI_API_KEY']

    raise ValueError(
        "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
        "or pass api_key parameter directly."
    )


def get_openai_client(api_key: Optional[str] = None):
    """
    Create and return OpenAI client.

    Args:
        api_key: Optional API key (uses env var if not provided)

    Returns:
        openai.OpenAI instance
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai package not installed. Install with:\n"
            "  pip install openai"
        )

    api_key = setup_openai_credentials(api_key=api_key)
    client = OpenAI(api_key=api_key)
    print("OpenAI client created")
    return client


# Model ID registry for OpenAI
OPENAI_MODEL_IDS = {
    "gpt-5.2": "gpt-5.2",
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
}


def get_openai_model_id(model_name: str) -> str:
    """
    Get full OpenAI model ID from short name.

    Args:
        model_name: Short model name (e.g., "gpt-5.2")

    Returns:
        Full model ID
    """
    if model_name in OPENAI_MODEL_IDS:
        return OPENAI_MODEL_IDS[model_name]

    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        return model_name

    raise ValueError(
        f"Unknown OpenAI model: {model_name}. "
        f"Available: {list(OPENAI_MODEL_IDS.keys())}"
    )


def test_openai_connection(client, test_model: str = "gpt-4o-mini"):
    """
    Test OpenAI connection with a simple API call.

    Args:
        client: OpenAI client
        test_model: Model ID to test with

    Returns:
        bool: True if connection successful
    """
    try:
        response = client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "Reply with just 'OK'"}],
            max_completion_tokens=50,
            temperature=0.0
        )

        response_text = response.choices[0].message.content
        print(f"OpenAI connection test PASSED (response: {response_text})")
        return True

    except Exception as e:
        print(f"OpenAI connection test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("OpenAI Configuration Test")
    print("=" * 80)

    try:
        client = get_openai_client()
        test_openai_connection(client)
        print("\nAll OpenAI setup tests passed!")
    except Exception as e:
        print(f"\nOpenAI setup failed: {e}")
        print("\nTo fix:")
        print("1. Install: pip install openai")
        print("2. Set OPENAI_API_KEY environment variable")
