"""
DSPy LM wrapper for Google Gemini 3 Flash Preview.
Supports context caching for cost optimization.
"""

import json
import time
import random
import threading
from types import SimpleNamespace
import dspy

from .llm_clients import ChatGemini, ChatGeminiCached


class TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens per second (e.g., 0.5 for 30 req/min)
            capacity: Max burst size
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.lock = threading.Lock()
        self.last_refill = time.time()

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Try to consume tokens. Blocks until available if blocking=True.
        """
        while True:
            with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not blocking:
                    return False

            # Wait before retry
            time.sleep(0.1)


class GeminiLM(dspy.LM):
    """
    DSPy LM interface for Google Gemini 3 Flash Preview.

    Features:
    - Built-in rate limiting
    - Exponential backoff retry
    - Context caching support (optional)
    - Token usage tracking
    """

    # Shared rate limiter across all instances
    _rate_limiter = None
    _limiter_lock = threading.Lock()
    _rate_limiter_rpm = None  # Track current RPM to detect changes

    def __init__(self, gemini_client, model_id="gemini-3-flash-preview",
                 model_name="gemini.3-flash", requests_per_minute=50,
                 use_cache=True, enable_context_caching=True,
                 temperature=1.0, thinking_level="high"):
        """
        Args:
            gemini_client: Google genai.Client instance
            model_id: Gemini model ID (default: gemini-3-flash-preview)
            model_name: DSPy model name
            requests_per_minute: Max requests per minute
            use_cache: Whether to enable caching (legacy parameter for compatibility)
            enable_context_caching: Whether to use ChatGeminiCached vs ChatGemini
            temperature: Sampling temperature (0.0-2.0, default: 1.0)
            thinking_level: Thinking level for Gemini 3 models (low/medium/high/off, default: high)
        """
        super().__init__(model_name)

        # Handle "off" option for thinking level
        actual_thinking_level = None if thinking_level == "off" else thinking_level

        # Initialize Gemini chat client
        if enable_context_caching:
            self.chat = ChatGeminiCached(model_id, gemini_client, "",
                                        temperature=temperature,
                                        thinking_level=actual_thinking_level)
            self.caching_enabled = True
        else:
            self.chat = ChatGemini(model_id, gemini_client, "", use_caching=False,
                                  temperature=temperature,
                                  thinking_level=actual_thinking_level)
            self.caching_enabled = False

        self.model_id = model_id
        self.last_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        }

        # Initialize shared rate limiter (one per process)
        with self._limiter_lock:
            if (GeminiLM._rate_limiter is None or
                GeminiLM._rate_limiter_rpm != requests_per_minute):

                req_per_sec = requests_per_minute / 60.0
                GeminiLM._rate_limiter = TokenBucket(
                    rate=req_per_sec,
                    capacity=max(int(req_per_sec * 2), 2)  # Allow small bursts
                )
                GeminiLM._rate_limiter_rpm = requests_per_minute

                cache_status = "with caching" if enable_context_caching else "no caching"
                print(f"   🚦 Gemini rate limiter: {requests_per_minute} RPM ({req_per_sec:.2f} req/sec) [{cache_status}]")

    def forward(self, prompt=None, messages=None, **kwargs):
        """DSPy BaseLM interface — returns OpenAI-compatible response object."""
        if messages:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"[SYSTEM]\n{content}")
                elif role == "assistant":
                    parts.append(f"[ASSISTANT]\n{content}")
                else:
                    parts.append(content)
            prompt_text = "\n\n".join(parts)
        elif prompt:
            prompt_text = prompt
        else:
            raise ValueError("No prompt or messages provided to GeminiLM")

        text = self._invoke(prompt_text, **kwargs)

        message = SimpleNamespace(content=text, tool_calls=None)
        choice = SimpleNamespace(message=message, logprobs=None)
        usage = self.last_usage.copy()  # dict, so dict(response.usage) works
        response = SimpleNamespace(choices=[choice], usage=usage, model=self.model_id)
        return response

    def _invoke(self, prompt: str, **kwargs) -> str:
        """
        Internal invoke with RATE LIMITING and RETRY.
        """
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # ============================================
        # RATE LIMITING: Wait for available capacity
        # ============================================
        self._rate_limiter.acquire(tokens=1, blocking=True)

        # ============================================
        # RETRY with exponential backoff
        # ============================================
        max_retries = 8
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                # Call Gemini via ChatGemini/ChatGeminiCached client
                result = self.chat.generate_isolated(prompt, use_cache=self.caching_enabled)

                # Extract response
                if isinstance(result, dict):
                    raw_response = result.get("raw_response", "")

                    # Handle None response
                    if raw_response is None:
                        raw_response = ""

                    # Check for errors
                    if raw_response and raw_response.startswith("ERROR::"):
                        raise Exception(f"Gemini API error: {raw_response}")

                    # Update usage tracking
                    if hasattr(self.chat, 'last_usage') and self.chat.last_usage:
                        self.last_usage = self.chat.last_usage.copy()

                    return raw_response
                else:
                    # String result (fallback)
                    return str(result)

            except Exception as e:
                error_str = str(e)

                # Check if throttling/quota error
                is_throttled = (
                    "quota" in error_str.lower() or
                    "rate" in error_str.lower() or
                    "429" in error_str or
                    "resource_exhausted" in error_str.lower()
                )

                if is_throttled and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"      ⚠️  Throttled (attempt {attempt+1}/{max_retries}), waiting {delay:.1f}s...")
                    print(f"         Error: {error_str}")
                    # Estimate token count for this failed call (rough approximation: 1 token ≈ 4 chars)
                    estimated_input_tokens = len(prompt) // 4
                    print(f"         Estimated tokens in failed call: {estimated_input_tokens:,} (prompt length: {len(prompt):,} chars)")
                    time.sleep(delay)
                    continue

                # Re-raise if not throttling or max retries reached
                if attempt == max_retries - 1:
                    print(f"      ❌ Max retries ({max_retries}) exceeded")
                raise

        raise Exception(f"Max retries ({max_retries}) exceeded due to throttling")
