"""
Robust DSPy LM wrapper for AWS Bedrock Claude.
WITH BUILT-IN RATE LIMITING to prevent throttling.

--- VERSION 2.2 (DUAL CACHE BREAKPOINTS) ---
- CORRECT context detection via DSPy prompt structure
- Automatic ephemeral caching for long contexts
- Solves "Request is too long" during GEPA reflection
- Backward compatible with non-DSPy usage
- NEW: Dual cache breakpoints for context + events
  - Breakpoint 1: [context] - shared across ALL stages
  - Breakpoint 2: [context + events] - shared across expert/judge/discussion stages
  - Maximizes cache hits when events are reused across multiple expert calls
"""

import os
import json
import time
import random
import threading
import re
import dspy


import threading

# Thread-safe output function container
class _OutputFuncHolder:
    """Thread-safe container for the output function.
    
    Uses a lock to ensure thread-safe access to the output function.
    This is necessary because worker threads need to see the updated
    output function set by the main thread.
    """
    _lock = threading.Lock()
    _func = print
    
    @classmethod
    def set(cls, func):
        with cls._lock:
            cls._func = func
    
    @classmethod
    def get(cls):
        with cls._lock:
            return cls._func
    
    @classmethod
    def write(cls, msg):
        """Write a message using the current output function."""
        with cls._lock:
            func = cls._func
        # Call outside lock to avoid deadlocks
        func(msg)

def set_output_func(func):
    """Set the output function for throttling messages.
    
    Use this to redirect throttling messages through tqdm.write() during parallel processing:
        from src.models.dspy_bedrock_lm import set_output_func
        set_output_func(pbar.write)
    
    Reset to default after processing:
        set_output_func(print)
    """
    _OutputFuncHolder.set(func)

def get_output_func():
    """Get the current output function."""
    return _OutputFuncHolder.get()

def _output(msg):
    """Thread-safe output function."""
    _OutputFuncHolder.write(msg)


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


class BedrockClaudeLM(dspy.LM):
    """
    DSPy LM interface for AWS Bedrock Claude.
    
    Features:
    - Built-in rate limiting (prevents throttling)
    - Exponential backoff retry
    - Prompt caching support (system prompt)
    - **AUTOMATIC ephemeral context caching**
    - Extended thinking support
    
    Context Caching Strategy:
    - Detects "Context:" prefix in DSPy prompts
    - Applies cache_control to context portion
    - 90% latency reduction, 85% cost savings
    """
    
    # Shared rate limiter across all instances
    _rate_limiter = None
    _limiter_lock = threading.Lock()
    
    def __init__(self, bedrock_client, model_id, system_prompt=None,
                 model_name="bedrock.claude", use_cache=False,
                 enable_thinking=False, thinking_budget=4000,
                 requests_per_minute=20,
                 enable_context_caching=True,
                 enable_1m_context=False,
                 max_tokens=12000):  
        """
        Args:
            bedrock_client: boto3 bedrock-runtime client
            model_id: Model ID (e.g., "us.anthropic.claude-sonnet-4.5")
            system_prompt: Optional system prompt
            use_cache: Enable caching for system prompt
            enable_thinking: Enable extended thinking
            thinking_budget: Max thinking tokens
            requests_per_minute: Max requests per minute
            enable_context_caching: Auto-cache long contexts (default: True)
            enable_1m_context: Enable 1M token context window (beta)
            max_tokens: Maximum output tokens (default: 12000, increased from 10000)
        """
        super().__init__(model_name)
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.use_cache = use_cache  # For system prompt only
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.enable_context_caching = enable_context_caching
        self.enable_1m_context = enable_1m_context
        self.max_tokens = max_tokens  # Store configurable max_tokens
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}
        
        # Initialize shared rate limiter (one per process)
        with self._limiter_lock:
            if BedrockClaudeLM._rate_limiter is None:
                req_per_sec = requests_per_minute / 60.0
                # Allow bursts of 8 requests (extraction + 4 experts + judge + answer)
                # This prevents sequential waiting when firing parallel experts
                burst_capacity = max(int(req_per_sec * 16), 8)
                BedrockClaudeLM._rate_limiter = TokenBucket(
                    rate=req_per_sec,
                    capacity=burst_capacity
                )
                print(f"   🚦 Rate limiter: {requests_per_minute} RPM (burst: {burst_capacity})")
        
        # Disable litellm
        os.environ["LITELLM_DISABLED"] = "true"
        os.environ["DSPY_USE_LITELLM"] = "0"
    
    def __call__(self, *args, **kwargs):
        """Flexible call that handles various invocation styles."""
        prompt = None
        
        if args:
            prompt = args[0]
        elif 'prompt' in kwargs:
            prompt = kwargs.pop('prompt')
        elif len(kwargs) > 0:
            # Format kwargs into a standard "Field: Value" prompt string.
            prompt = "\n\n".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in kwargs.items())
        
        if prompt is None:
            raise ValueError("No prompt provided to BedrockClaudeLM")
        
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
    
    def _extract_context_for_system_cache(self, prompt: str) -> tuple:
        """
        Extract context from prompt to be cached in system message.

        Returns:
            (context_content, prompt_without_context) or (None, prompt) if no context found
        """
        dspy_context_match = re.search(
            r'\[\[\s*##\s*context\s*##\s*\]\]\s*\n(.+?)(?=\n\s*\[\[\s*##|\Z)',
            prompt,
            re.DOTALL | re.IGNORECASE
        )

        if dspy_context_match:
            context_content = dspy_context_match.group(1).strip()
            context_tokens = len(context_content) // 4

            # Only extract if substantial (>1000 tokens)
            if context_tokens > 1000:
                # Replace context content with reference to system message
                before_match = prompt[:dspy_context_match.start()]
                after_match = prompt[dspy_context_match.end():]
                prompt_without_context = before_match + "[[ ## context ## ]]\n[See system context above]\n" + after_match
                return context_content, prompt_without_context

        return None, prompt

    def _extract_events_for_system_cache(self, prompt: str) -> tuple:
        """
        Extract events from prompt to be cached in system message (second cache breakpoint).
        
        This enables caching of context + events together, so expert stages can
        benefit from cache hits on both the context AND the extracted events.

        Returns:
            (events_content, prompt_without_events) or (None, prompt) if no events found
        """
        dspy_events_match = re.search(
            r'\[\[\s*##\s*events\s*##\s*\]\]\s*\n(.+?)(?=\n\s*\[\[\s*##|\Z)',
            prompt,
            re.DOTALL | re.IGNORECASE
        )

        if dspy_events_match:
            events_content = dspy_events_match.group(1).strip()
            events_tokens = len(events_content) // 4

            # Only extract if meaningful (>100 tokens) - events are typically smaller than context
            if events_tokens > 100:
                # Replace events content with reference to system message
                before_match = prompt[:dspy_events_match.start()]
                after_match = prompt[dspy_events_match.end():]
                prompt_without_events = before_match + "[[ ## events ## ]]\n[See extracted events above]\n" + after_match
                return events_content, prompt_without_events

        return None, prompt

    def _split_prompt_for_caching(self, prompt: str) -> list:
        """
        Split prompt into content blocks (no caching in user message).
        Context caching now happens in system message via _extract_context_for_system_cache.

        Returns:
            List of content blocks
        """
        # Simple: just return the prompt as a single block
        # Context caching is handled in system message now
        return [{"type": "text", "text": prompt}]

    
    def _invoke(self, prompt: str, **kwargs) -> str:
        """
        Internal invoke with RATE LIMITING, RETRY, and AUTO-CACHING.
        """
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # ============================================
        # DETECT & EXTRACT DSPY MESSAGE FORMAT
        # ============================================
        # DSPy might pass a messages structure as string: "{'messages': [...]}"
        # or "Messages: [...]". We need to extract the actual user content.
        system_from_dspy = None
        user_content = prompt  # Default: use prompt as-is

        # Try to parse if prompt looks like a dict/messages structure
        if prompt.strip().startswith('{') or prompt.strip().startswith('Messages:'):
            try:
                # Try to parse as JSON-like structure
                import ast
                clean_prompt = prompt.strip()
                if clean_prompt.startswith('Messages:'):
                    clean_prompt = clean_prompt.replace('Messages:', '{"messages":', 1) + '}'

                parsed = ast.literal_eval(clean_prompt) if clean_prompt.startswith('{') else None

                if parsed and 'messages' in parsed:
                    messages_list = parsed['messages']
                    for msg in messages_list:
                        if msg.get('role') == 'system':
                            system_from_dspy = msg.get('content', '')
                        elif msg.get('role') == 'user':
                            content = msg.get('content', '')
                            if isinstance(content, str):
                                user_content = content
                            elif isinstance(content, list):
                                # Handle list of content blocks
                                user_content = ' '.join(
                                    c.get('text', '') if isinstance(c, dict) else str(c)
                                    for c in content
                                )
            except (ValueError, SyntaxError):
                # If parsing fails, use prompt as-is
                pass

        # ============================================
        # RATE LIMITING: Wait for available capacity
        # ============================================
        self._rate_limiter.acquire(tokens=1, blocking=True)

        # ============================================
        # DUAL CACHE BREAKPOINT STRATEGY
        # ============================================
        # Anthropic supports up to 4 cache breakpoints. We use 2:
        # 
        # Breakpoint 1: [context] - shared across ALL stages (extraction, experts, judge, etc.)
        # Breakpoint 2: [context + events] - shared across expert/judge/discussion stages
        #
        # Cache key = prefix up to each cached block:
        # - Extraction stage: writes breakpoint 1 (context only)
        # - Expert stage 1: reads breakpoint 1, writes breakpoint 2 (context + events)
        # - Expert stages 2-4: reads breakpoint 2 (context + events)
        # - Judge/Discussion: reads breakpoint 2 (context + events)
        #
        # This maximizes cache hits since context+events are identical across expert calls.
        system_blocks = []
        extracted_context = None
        extracted_events = None

        # Step 1: Extract context from user prompt (if substantial)
        if self.enable_context_caching:
            extracted_context, user_content = self._extract_context_for_system_cache(user_content)
            if extracted_context:
                # Context goes FIRST - this is cache breakpoint 1
                system_blocks.append({
                    "type": "text",
                    "text": f"[DOCUMENT CONTEXT]\n{extracted_context}\n[/DOCUMENT CONTEXT]",
                    "cache_control": {"type": "ephemeral"}
                })

        # Step 2: Extract events from user prompt (second cache breakpoint)
        if self.enable_context_caching:
            extracted_events, user_content = self._extract_events_for_system_cache(user_content)
            if extracted_events:
                # Events go AFTER context - this is cache breakpoint 2
                # The cache key for breakpoint 2 = [context block] + [events block]
                # So if context matches AND events match, we get a cache hit on breakpoint 2
                system_blocks.append({
                    "type": "text",
                    "text": f"[EXTRACTED EVENTS]\n{extracted_events}\n[/EXTRACTED EVENTS]",
                    "cache_control": {"type": "ephemeral"}
                })

        # Step 3: Add DSPy/custom system prompt AFTER cached blocks (not cached, can vary)
        effective_system = system_from_dspy or self.system_prompt
        if effective_system:
            system_blocks.append({
                "type": "text",
                "text": effective_system
            })

        # ============================================
        # USER MESSAGE (no caching, varies per signature)
        # ============================================
        content_blocks = self._split_prompt_for_caching(user_content)
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": content_blocks
            }
        ]
        
        # Calculate max_tokens - use instance default (self.max_tokens)
        if self.enable_thinking:
            max_tokens = kwargs.get("max_tokens", self.thinking_budget * 2)
            if max_tokens <= self.thinking_budget:
                max_tokens = self.thinking_budget + 4000
        else:
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Build payload
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 1),
            "anthropic_version": "bedrock-2023-05-31"
        }
        
        # Add thinking if enabled
        if self.enable_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
        
        # Add system with cache control
        if system_blocks:
            payload["system"] = system_blocks
        
        # ============================================
        # ENHANCED RETRY with exponential backoff
        # Optimized for large experiments (400+ questions)
        # ============================================
        max_retries = 20           # More attempts for long experiments
        base_delay = 3.0           # Start with longer delay
        max_delay = 300.0          # Cap at 5 minutes
        jitter_factor = 0.5        # Randomness to prevent thundering herd
        sustained_threshold = 3    # After 3 consecutive throttles, switch to sustained mode
        sustained_delay = 120.0    # 2 minutes for sustained throttling
        
        consecutive_throttles = 0
        total_wait_time = 0.0

        for attempt in range(max_retries):
            try:
                # Add 1M context beta header if enabled (goes in payload body)
                if self.enable_1m_context:
                    payload["anthropic_beta"] = ["context-1m-2025-08-07"]

                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    body=json.dumps(payload)
                )

                output = json.loads(response["body"].read())

                # UPDATE: Capture token usage (including cache tokens)
                if "usage" in output:
                    usage = output["usage"]
                    self.last_usage = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)
                    }
                else:
                    # Reset if not present
                    self.last_usage = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0
                    }

                # Extract text from response
                text_parts = []
                for content_block in output["content"]:
                    if content_block["type"] == "text":
                        text_parts.append(content_block["text"])

                # Success! Log if we had to retry
                if attempt > 0:
                    _output(f"      ✓ Succeeded after {attempt} retries (waited {total_wait_time:.1f}s total)")

                return " ".join(text_parts).strip()

            except Exception as e:
                error_str = str(e)

                # Check for FATAL errors that shouldn't be retried
                is_daily_limit = (
                    "Too many tokens per day" in error_str or
                    ("daily" in error_str.lower() and "limit" in error_str.lower())
                )

                if is_daily_limit:
                    _output("\n      ❌ FATAL: Daily token limit exceeded")
                    _output(f"         Error: {error_str[:200]}")
                    _output("         No point in retrying - you've hit your daily quota")
                    _output("\n         💡 To resume later, use: --resume <experiment_dir>")
                    raise RuntimeError(f"Daily token limit exceeded: {error_str}") from e

                # Check if throttling error (rate limit, not daily limit)
                is_throttled = (
                    "ThrottlingException" in error_str or
                    "throttling" in error_str.lower() or
                    "rate limit" in error_str.lower() or
                    "429" in error_str or
                    "Too many requests" in error_str
                ) and not is_daily_limit
                
                # Also handle 503 Service Unavailable as retryable
                is_service_unavailable = (
                    "503" in error_str or
                    "ServiceUnavailable" in error_str or
                    "service unavailable" in error_str.lower() or
                    "temporarily unavailable" in error_str.lower()
                )
                
                is_retryable = is_throttled or is_service_unavailable

                if is_retryable and attempt < max_retries - 1:
                    consecutive_throttles += 1
                    
                    # Determine delay based on throttling pattern
                    if consecutive_throttles >= sustained_threshold:
                        # Sustained throttling: use fixed longer delay
                        delay = sustained_delay + random.uniform(0, sustained_delay * jitter_factor)
                        throttle_type = "sustained"
                    else:
                        # Normal exponential backoff with cap
                        exp_delay = base_delay * (2 ** min(attempt, 6))  # Cap exponential growth
                        delay = min(exp_delay, max_delay) + random.uniform(0, base_delay * jitter_factor)
                        throttle_type = "transient"
                    
                    total_wait_time += delay
                    error_type = "Throttled" if is_throttled else "Service unavailable"
                    
                    _output(f"      ⚠️  {error_type} ({throttle_type}, attempt {attempt+1}/{max_retries})")
                    _output(f"         Waiting {delay:.1f}s... (total wait: {total_wait_time:.1f}s)")
                    time.sleep(delay)
                    continue

                # Check for "Request is too long" error
                is_too_long = (
                    "Request is too long" in error_str or
                    "too long" in error_str.lower() or
                    "exceeds" in error_str.lower() or
                    "maximum context length" in error_str.lower()
                )

                if is_too_long:
                    _output("      ❌ FATAL: Request exceeds model context limit")
                    _output(f"         Context caching: {'ENABLED' if self.enable_context_caching else 'DISABLED'}")
                    _output(f"         Prompt length: ~{len(prompt)//4} tokens")
                    if not self.enable_context_caching:
                        _output("         💡 Try: enable_context_caching=True")
                    raise

                # Re-raise if not retryable or max retries reached
                if attempt == max_retries - 1:
                    _output(f"\n      ❌ Max retries ({max_retries}) exceeded after {total_wait_time:.1f}s total wait")
                    _output(f"         Last error: {error_str[:200]}")
                    _output("\n         💡 The API may be experiencing issues. Try again later.")
                    _output("         💡 To resume, use: --resume <experiment_dir>")
                raise

        raise RuntimeError(f"Max retries ({max_retries}) exceeded due to throttling (waited {total_wait_time:.1f}s)")
