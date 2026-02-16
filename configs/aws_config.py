"""
AWS Bedrock configuration and client initialization.

Credentials are loaded from:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. Standard AWS credentials file (~/.aws/credentials)
3. AWS CLI configuration (aws configure)
"""
import os
import json
from pathlib import Path
from typing import Optional
import boto3
from botocore.config import Config


def get_bedrock_client(region: str = 'us-east-1'):
    """
    Create and return AWS Bedrock runtime client.

    Args:
        region: AWS region (default: us-east-1)

    Returns:
        boto3 Bedrock runtime client
    """
    try:
        retry_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        client = boto3.client('bedrock-runtime', region_name=region, config=retry_config)
        print(f"AWS Bedrock client created (region: {region})")
        return client
    except Exception as e:
        print(f"Failed to create Bedrock client: {e}")
        print("\nTroubleshooting:")
        print("1. Run: aws configure")
        print("2. Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        raise


def test_bedrock_connection(client, test_model: str = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'):
    """
    Test AWS Bedrock connection with a simple API call.

    Args:
        client: Bedrock runtime client
        test_model: Model ID to test with (default: Claude 3.5 Haiku)

    Returns:
        bool: True if connection successful
    """
    try:
        response = client.invoke_model(
            modelId=test_model,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Reply with just 'OK'"}]
            })
        )

        result = json.loads(response['body'].read())
        response_text = result['content'][0]['text']

        print(f"Bedrock connection test PASSED (response: {response_text})")
        return True

    except Exception as e:
        print(f"Bedrock connection test FAILED: {e}")
        return False


def get_bedrock_agent_runtime_client(region: str = 'us-east-1'):
    """
    Create and return AWS Bedrock Agent Runtime client (for Rerank API).

    Args:
        region: AWS region (default: us-east-1)

    Returns:
        boto3 Bedrock Agent Runtime client
    """
    try:
        client = boto3.client('bedrock-agent-runtime', region_name=region)
        print(f"AWS Bedrock Agent Runtime client created (region: {region})")
        return client
    except Exception as e:
        print(f"Failed to create Bedrock Agent Runtime client: {e}")
        raise


# Model ID registry
MODEL_IDS = {
    # Claude models
    "claude-sonnet-4.0": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4.5-1m": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4.0": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-opus-4.1": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-haiku-3.5": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",

    # Llama models
    "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",

    # DeepSeek models
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "deepseek-v3.1": "deepseek.v3-v1:0",

    # Kimi models
    "kimi-k2-thinking": "moonshot.kimi-k2-thinking",

    # Embedding models
    "cohere.embed-v4": "cohere.embed-v4:0",
}


def get_model_id(model_name: str) -> str:
    """
    Get full model ID from short name.

    Args:
        model_name: Short model name (e.g., "claude-sonnet-4.5")

    Returns:
        Full model ID

    Raises:
        ValueError: If model name not found
    """
    if model_name in MODEL_IDS:
        return MODEL_IDS[model_name]

    # If it's already a full ID, return as-is
    if ':' in model_name:
        return model_name

    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_IDS.keys())}")


if __name__ == "__main__":
    print("AWS Bedrock Configuration Test")
    print("=" * 80)

    try:
        client = get_bedrock_client()
        test_bedrock_connection(client)
        print("\nAll AWS setup tests passed!")
    except Exception as e:
        print(f"\nAWS setup failed: {e}")
