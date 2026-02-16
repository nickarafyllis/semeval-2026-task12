"""
DSPy LM wrapper for Kimi K2 Thinking (Moonshot AI) via AWS Bedrock.
Supports thinking mode for enhanced reasoning.
"""

import json
import time
import random
import threading
import dspy

from .llm_clients import ChatKimi


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


class KimiLM(dspy.LM):
    """
    DSPy LM interface for Kimi K2 Thinking via AWS Bedrock.

    Features:
    - Built-in rate limiting
    - Exponential backoff retry
    - Extended thinking support
    - Thinking token extraction
    """

    # Shared rate limiter across all instances
    _rate_limiter = None
    _limiter_lock = threading.Lock()

    def __init__(self, bedrock_client, model_id="moonshot.kimi-k2-thinking",
                 model_name="kimi.k2", requests_per_minute=20):
        """
        Args:
            bedrock_client: boto3 bedrock-runtime client
            model_id: Kimi model ID (default: moonshot.kimi-k2-thinking)
            model_name: DSPy model name
            requests_per_minute: Max requests per minute
        """
        super().__init__(model_name)

        # Initialize Kimi chat client (correct parameter order)
        self.chat = ChatKimi(model_id, bedrock_client)
        self.model_id = model_id
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

        # Initialize shared rate limiter (one per process)
        with self._limiter_lock:
            if KimiLM._rate_limiter is None:
                req_per_sec = requests_per_minute / 60.0
                KimiLM._rate_limiter = TokenBucket(
                    rate=req_per_sec,
                    capacity=max(int(req_per_sec * 2), 2)  # Allow small bursts
                )
                print(f"   🚦 Kimi rate limiter: {requests_per_minute} RPM ({req_per_sec:.2f} req/sec)")

    def __call__(self, *args, **kwargs):
        """Flexible call that handles various invocation styles."""
        prompt = None

        if args:
            prompt = args[0]
        elif 'prompt' in kwargs:
            prompt = kwargs.pop('prompt')
        elif len(kwargs) > 0:
            prompt = str(kwargs)

        if prompt is None:
            raise ValueError("No prompt provided to KimiLM")

        return self.request(prompt, **kwargs)

    def request(self, prompt: str, **kwargs):
        """Core DSPy interface."""
        text = self._invoke(prompt, **kwargs)
        return [text]

    def complete(self, prompt: str, **kwargs):
        """Alternative interface."""
        return self._invoke(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs):
        """OpenAI-style interface."""
        text = self._invoke(prompt, **kwargs)
        return [text]

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
                # Call Kimi via ChatKimi client
                result = self.chat.generate_isolated(prompt)

                # Extract thinking and answer
                if isinstance(result, dict):
                    thinking = result.get("thinking", "")
                    answer = result.get("answer", "")

                    # Combine for DSPy (thinking gets passed through)
                    if thinking:
                        output = f"<reasoning>{thinking}</reasoning>\n{answer}"
                    else:
                        output = answer

                    # Store thinking in history for potential retrieval
                    if hasattr(self, '_last_thinking'):
                        self._last_thinking = thinking

                    return output
                else:
                    # String result (fallback)
                    return str(result)

            except Exception as e:
                error_str = str(e)

                # Check if throttling error
                is_throttled = (
                    "ThrottlingException" in error_str or
                    "throttling" in error_str.lower() or
                    "rate" in error_str.lower() or
                    "429" in error_str
                )

                if is_throttled and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"      ⚠️  Throttled (attempt {attempt+1}/{max_retries}), waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                # Re-raise if not throttling or max retries reached
                if attempt == max_retries - 1:
                    print(f"      ❌ Max retries ({max_retries}) exceeded")
                raise

        raise Exception(f"Max retries ({max_retries}) exceeded due to throttling")
