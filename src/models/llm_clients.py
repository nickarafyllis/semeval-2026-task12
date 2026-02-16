"""
AWS Bedrock LLM Client Classes

This module contains client wrappers for different LLM models available through AWS Bedrock:
- Claude (Anthropic): Multiple variants including thinking mode and caching
- Llama (Meta): Instruction-tuned models  
- DeepSeek: Reasoning models with thinking extraction

Each client handles:
- Model-specific prompt formatting
- API invocation with retry logic
- Response parsing and error handling
- Structured output via Tool Use (100% reliable JSON)
- Optional features (caching, thinking, etc.)
"""

import json
from typing import Optional, Dict, List


# ============================================================================
# SHARED SCHEMA DEFINITIONS
# ============================================================================

# Default schema for abductive reasoning task
DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Brief reasoning analyzing each option A-D based on context (3-6 sentences)"
        },
        "answer": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["A", "B", "C", "D"]
            },
            "description": "Array of one or more correct answer letters (always return as array, even for single answer)",
            "minItems": 1,
            "maxItems": 4
        }
    },
    "required": ["analysis", "answer"],
    "additionalProperties": False
}



# ============================================================================
# CLAUDE CLIENTS
# ============================================================================

class ChatClaudeUncached:
    """
    Claude client without prompt caching.
    
    Suitable for testing, one-off queries, or scenarios where prompt caching
    is not beneficial (e.g., unique prompts every time).
    
    Attributes:
        model_id: AWS Bedrock model identifier
        bedrock_runtime_client: Boto3 Bedrock runtime client
        system_prompt: Optional system-level instructions
        max_retries: Maximum retry attempts on failures
        base_delay: Base delay in seconds for exponential backoff
    
    Example:
        >>> client = ChatClaudeUncached(model_id, bedrock_client, "You are helpful")
        >>> client.add_user_message("What is 2+2?")
        >>> response = client.generate()
        >>> print(response)
    """

    def __init__(self, model_id, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2
        self.payload = {
            "messages": [],
            "max_tokens": 5000,
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": 0.3,
            "top_k": 0
        }

        if self.system_prompt:
            self.payload["system"] = self.system_prompt

    def set_system_prompt(self, system_prompt: str):
        """Set or update the system prompt"""
        self.system_prompt = system_prompt
        self.payload["system"] = system_prompt

    def add_user_message(self, message):
        """Add a user message to the conversation"""
        self.payload["messages"].append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })

    def generate(self):
        """Generate assistant response and append to conversation history"""
        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            body=json.dumps(self.payload)
        )

        output_binary = response["body"].read()
        output_json = json.loads(output_binary)
        output = output_json["content"][0]["text"]

        self.payload["messages"].append({
            "role": "assistant",
            "content": [{"type": "text", "text": output}]
        })

        return output

    def reset(self):
        """Reset conversation but keep system prompt"""
        self.payload["messages"] = []
        if self.system_prompt:
            self.payload["system"] = self.system_prompt


class ChatClaude:
    """
    Claude client with prompt caching and structured output support.
    
    Uses Anthropic's prompt caching to reduce latency and costs when reusing
    large context (e.g., documents, instructions) across multiple queries.
    
    Supports structured output via Tool Use for 100% reliable JSON responses.
    
    Attributes:
        model_id: AWS Bedrock model identifier
        bedrock_runtime_client: Boto3 Bedrock runtime client
        system_prompt: Optional system-level instructions
        max_retries: Maximum retry attempts on failures
        base_delay: Base delay in seconds for exponential backoff
    
    Example:
        >>> client = ChatClaude(model_id, bedrock_client, "You are an expert")
        >>> 
        >>> # Structured output (guaranteed valid JSON)
        >>> schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        >>> result = client.generate_structured("What is 2+2?", schema)
        >>> print(result)  # {'answer': 'Four'}
    """

    def __init__(self, model_id, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2
        self.base_payload = {
            "max_tokens": 3000,
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": 0.3,
            "top_k": 0
        }
        if self.system_prompt:
            self.base_payload["system"] = self.system_prompt

        # Track last API call usage
        self.last_usage = None

    def generate_structured(self, messages: List[Dict], output_schema: Dict = None) -> Dict:
        """
        Generate structured output using Bedrock Tool Use (100% reliable JSON).
        
        Args:
            messages: List of message dicts in Claude format 
                     Example: [{"role": "user", "content": [{"text": "..."}]}]
            output_schema: JSON schema defining the expected output structure.
                          If None, uses DEFAULT_OUTPUT_SCHEMA for abductive reasoning.
        
        Returns:
            Dict: Parsed JSON matching the schema (already a Python dict)
        
        Example:
            >>> messages = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]
            >>> result = client.generate_structured(messages, schema)
            >>> print(result)  # {'answer': ['A'], 'analysis': '...'}
        """
        if output_schema is None:
            output_schema = DEFAULT_OUTPUT_SCHEMA

        # Define tool with schema
        tool_config = {
            "tools": [{
                "toolSpec": {
                    "name": "provide_structured_answer",
                    "description": "Provide answer in the required structured format",
                    "inputSchema": {"json": output_schema}
                }
            }],
            "toolChoice": {
                "any": {} 
            }
        }

        # System prompt
        system = []
        if self.system_prompt:
            system = [{"text": self.system_prompt}]

        # Call Converse API with tool
        response = self.bedrock_runtime_client.converse(
            modelId=self.model_id,
            system=system,
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={
                "temperature": 0.3,
                "maxTokens": 3000
            }
        )

        # Track usage if available
        if "usage" in response:
            self.last_usage = response["usage"]

        # Extract tool use result
        output = response['output']['message']['content']

        for block in output:
            if 'toolUse' in block:
                tool_use = block['toolUse']
                if tool_use['name'] == 'provide_structured_answer':
                    return tool_use['input']

        raise ValueError("No structured output found in response")

    def generate_isolated(self, messages):
        """
        Generate response for isolated messages without conversation history.

        Args:
            messages: List of message dicts in Anthropic format:
                     [{"role": "user", "content": [...]}]

        Returns:
            str: Raw text response from Claude (first text block)
        """
        payload = self.base_payload.copy()
        payload["messages"] = messages

        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            body=json.dumps(payload)
        )

        output_binary = response["body"].read()
        output_json = json.loads(output_binary)

        # Track usage if available
        if "usage" in output_json:
            self.last_usage = output_json["usage"]

        # Extract first text content block (raw string)
        return output_json["content"][0]["text"]



class ChatClaudeThinking:
    """
    Claude client with extended thinking mode.
    
    Enables Claude's reasoning capabilities where the model explicitly shows
    its step-by-step thinking process before providing the final answer.
    """

    def __init__(self, model_id: str, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2
        self.base_payload = {
            "max_tokens": 5000,
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": 1, #extended thinking
            "thinking": {
                "type": "enabled",
                "budget_tokens": 3000
            }
        }

        if self.system_prompt:
            self.base_payload["system"] = self.system_prompt

        # Track last API call usage
        self.last_usage = None

    def generate_isolated(self, messages: List[Dict]) -> Dict[str, str]:
        """
        Generate response with thinking process extraction.
        
        Args:
            messages: List of message dicts in Converse API format:
                    [{"role": "user", "content": [{"text": "..."}]}]
        
        Returns:
            Dict with 'thinking' (reasoning process) and 'answer' (final response)
        """
        # Convert from Converse API format to InvokeModel format
        invoke_messages = []

        for msg in messages:
            role = msg["role"]
  
            # Extract text from content blocks
            if isinstance(msg["content"], list):
                # Content is list of blocks: [{"text": "..."}, ...]
                text_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                content_text = "\n".join(text_parts)
            else:
                # Content is already a string
                content_text = msg["content"]

            invoke_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content_text}]
            })

        payload = self.base_payload.copy()
        payload["messages"] = invoke_messages

        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            body=json.dumps(payload)
        )

        output_binary = response['body'].read()
        output_json = json.loads(output_binary)

        # Track usage if available
        if "usage" in output_json:
            self.last_usage = output_json["usage"]

        content = output_json["content"]

        # Extract thinking and answer
        thinking_content = ""
        final_answer = ""

        # Content is a list of blocks
        for block in content:
            if isinstance(block, dict):
                if "thinking" in block:
                    thinking_content = block["thinking"]
                elif "text" in block:
                    final_answer = block["text"]

        return {
            "thinking": thinking_content,
            "answer": final_answer
        }


    def generate_structured(self, messages: List[Dict], output_schema: Dict = None) -> Dict:
        """
        DEPRECATED: Thinking mode does not support structured output (tool use).
        
        AWS Bedrock error: "Thinking may not be enabled when tool_choice forces tool use."
        
        Use generate_isolated() instead and parse the text response.
        """
        raise NotImplementedError(
            "ChatClaudeThinking does not support generate_structured(). "
            "AWS Bedrock does not allow thinking mode with forced tool use. "
            "Use generate_isolated() instead."
        )


# ============================================================================
# LLAMA CLIENT
# ============================================================================

class ChatLlama:
    """
    Llama instruction-tuned model client with structured output support.
    
    Handles Meta's Llama chat format with proper instruction formatting,
    system prompts, and Bedrock-specific payload structure.
    
    Supports structured output via Tool Use.
    
    Attributes:
        model_id: AWS Bedrock model identifier (e.g., llama-3.3-70b)
        bedrock_runtime_client: Boto3 Bedrock runtime client
        system_prompt: Optional system-level instructions
        max_retries: Maximum retry attempts on failures
        base_delay: Base delay in seconds for exponential backoff
    
    Example:
        >>> client = ChatLlama(model_id, bedrock_client, "You are helpful")
        >>> 
        >>> # Structured output
        >>> result = client.generate_structured("Classify: Python is a language", schema)
        >>> print(result)  # {'category': 'programming', 'confidence': 'high'}
    """

    def __init__(self, model_id, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2

    def generate_isolated(self, user_prompt):
        """Generate response for Llama with correct payload format"""

        # Format prompt in Llama instruction format
        if self.system_prompt:
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{self.system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        else:
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        # Llama Bedrock payload format
        payload = {
            "prompt": formatted_prompt,
            "max_gen_len": 3000,
            "temperature": 0.3,
        }

        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            body=json.dumps(payload)
        )

        output_binary = response["body"].read()
        output_json = json.loads(output_binary)
        return output_json["generation"]



# ============================================================================
# DEEPSEEK CLIENTS
# ============================================================================

class ChatDeepSeek:
    """
    DeepSeek R1 client with thinking extraction (pure text).
    """

    def __init__(self, model_id, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 3
        self.base_delay = 1.5

    def generate_isolated(self, user_prompt: str) -> Dict:
        """
        Call DeepSeek R1 with text prompt and return raw response.
        
        Args:
            user_prompt: User's text prompt
        
        Returns:
            Dict with 'raw_response' key
        """
        try:
            if self.system_prompt:
                formatted_prompt = f"""<｜begin▁of▁sentence｜><｜System｜>{self.system_prompt}

<｜User｜>{user_prompt}<｜Assistant｜><think>
"""
            else:
                formatted_prompt = f"""<｜begin▁of▁sentence｜><｜User｜>{user_prompt}<｜Assistant｜><think>
"""

            payload = {
                "prompt": formatted_prompt,
                "max_tokens": 3000,
                "temperature": 0.3
            }

            response = self.bedrock_runtime_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(payload)
            )

            output_binary = response["body"].read()
            output_json = json.loads(output_binary)
            raw_response = output_json["choices"][0]["text"]

            return {"raw_response": raw_response}

        except Exception as e:
            return {"raw_response": f"ERROR::{str(e)}"}

    @staticmethod
    def split_thinking_and_answer(raw_response: str) -> Dict:
        """
        Extract <think>...</think> reasoning and final answer text.
        
        Args:
            raw_response: Raw text from model
        
        Returns:
            Dict with 'thinking' and 'answer_text' keys
        """
        thinking = ""
        answer_text = raw_response

        if "</think>" in raw_response:
            parts = raw_response.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            answer_text = parts[1].strip()
        else:
            # Heuristic fallback
            lines = raw_response.splitlines()
            cut = 0
            for i, ln in enumerate(lines):
                if any(k in ln.lower() for k in ["answer:", "final answer", "therefore", "conclusion"]):
                    cut = i
                    break
            thinking = "\n".join(lines[:cut]).strip()
            answer_text = "\n".join(lines[cut:]).strip() or raw_response

        return {"thinking": thinking, "answer_text": answer_text}

class ChatDeepSeekV31:
    """
    DeepSeek V3.1 client with reasoning support (from additionalModelRequestFields).
    """

    def __init__(self, model_id, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2

    def generate_isolated(self, user_prompt: str,  reasoning_effort: str = "high") -> Dict:
        """
        Generate response with specified reasoning effort.
        
        Args:
            user_prompt: Prompt
            reasoning_effort: "off", "low", "medium", "high" (default "high")
        
        Returns:
            {'thinking': ..., 'answer': ..., 'flag': ...}
        """
        flag = False

        try:
            messages = []
            system = []

            if self.system_prompt:
                system = [{"text": self.system_prompt}]

            messages.append({
                "role": "user",
                "content": [{"text": user_prompt}]
            })

            # Configure additional fields based on level
            additional_fields = {}
            if reasoning_effort != "off":
                # Map to effort level (assume AWS supports these; fallback to high if invalid)
                effort_map = {
                    "low": "low",
                    "medium": "medium",
                    "high": "high"
                }
                mapped_effort = effort_map.get(reasoning_effort, "high")
                additional_fields["reasoning_effort"] = mapped_effort
                print(f"   Reasoning effort: {mapped_effort}")

            response = self.bedrock_runtime_client.converse(
                modelId=self.model_id,
                system=system,
                messages=messages,
                inferenceConfig={
                    "temperature": 0.5 if reasoning_effort != "off" else 0.3,  # Higher temp for reasoning
                    "maxTokens": 5000 if reasoning_effort != "off" else 3000  # More tokens for thinking
                },
                additionalModelRequestFields=additional_fields
            )

            # Extract both reasoning and final answer (like in your Colab)
            content = response['output']['message']['content']
            thinking = ""
            final_answer = ""

            for item in content:
                if 'reasoningContent' in item:
                    reasoning_data = item['reasoningContent']
                    if 'reasoningText' in reasoning_data:
                        thinking = reasoning_data['reasoningText'].get('text', '')
                elif 'text' in item:
                    final_answer += item['text']  # Concat if multiple text blocks

            return {
                "thinking": thinking,
                "answer": final_answer,
                "flag": flag
            }

        except Exception as e:
            flag = True  # Set flag to True when error occurs
            return {
                "thinking": "",
                "answer": f"ERROR::{str(e)}",
                "flag": flag
            }


# ============================================================================
# KIMI CLIENT
# ============================================================================

class ChatKimi:
    """
    Kimi (Moonshot AI) client with reasoning extraction.

    Kimi K2 Thinking provides step-by-step reasoning process before final answer,
    similar to DeepSeek's thinking mode.

    Uses OpenAI-compatible API format through Bedrock with reasoning tags.

    Attributes:
        model_id: AWS Bedrock model identifier (moonshot.kimi-k2-thinking)
        bedrock_runtime_client: Boto3 Bedrock runtime client
        system_prompt: Optional system-level instructions
        max_retries: Maximum retry attempts on failures
        base_delay: Base delay in seconds for exponential backoff

    Example:
        >>> client = ChatKimi(model_id, bedrock_client, "You are helpful")
        >>> result = client.generate_isolated("What is 2+2?")
        >>> print(result)  # {'thinking': '...', 'answer': '...'}
    """

    def __init__(self, model_id: str, bedrock_runtime_client, system_prompt: Optional[str] = None):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2
        self.last_usage = None

    def generate_isolated(self, user_prompt: str) -> Dict:
        """
        Generate response with thinking process extraction.

        Args:
            user_prompt: User's text prompt

        Returns:
            Dict with 'thinking' (reasoning in <reasoning> tags) and 'answer' (final response)
        """
        try:
            messages = []

            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })

            messages.append({
                "role": "user",
                "content": user_prompt
            })

            payload = {
                "messages": messages,
                "max_tokens": 5000,  # Higher limit for reasoning models
                "temperature": 0.3
            }

            response = self.bedrock_runtime_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(payload)
            )

            output_binary = response["body"].read()
            output_json = json.loads(output_binary)

            # Track usage (convert OpenAI format to Claude format for compatibility)
            if "usage" in output_json:
                usage = output_json["usage"]
                self.last_usage = {
                    "inputTokens": usage.get("prompt_tokens", 0),
                    "outputTokens": usage.get("completion_tokens", 0)
                }

            # Extract content from OpenAI-compatible response
            raw_response = output_json["choices"][0]["message"]["content"]

            return self.split_thinking_and_answer(raw_response)

        except Exception as e:
            return {
                "thinking": "",
                "answer": f"ERROR::{str(e)}"
            }

    @staticmethod
    def split_thinking_and_answer(raw_response: str) -> Dict:
        """
        Extract <reasoning>...</reasoning> tags and final answer text.

        Args:
            raw_response: Raw text from model

        Returns:
            Dict with 'thinking' and 'answer' keys
        """
        thinking = ""
        answer_text = raw_response

        if "</reasoning>" in raw_response:
            parts = raw_response.split("</reasoning>", 1)
            thinking = parts[0].replace("<reasoning>", "").strip()
            answer_text = parts[1].strip()

        return {"thinking": thinking, "answer": answer_text}


# ============================================================================
# GEMINI CLIENT
# ============================================================================

class ChatGemini:
    """
    Google Gemini client with context caching support.

    Uses Google's genai SDK for Gemini 3 Flash Preview.
    Supports context caching for repeated document contexts.

    Attributes:
        model_id: Gemini model identifier (e.g., "gemini-3-flash-preview")
        client: Google genai.Client instance
        system_prompt: Optional system-level instructions
        max_retries: Maximum retry attempts on failures
        base_delay: Base delay in seconds for exponential backoff
        use_caching: Whether to use context caching
        cached_content: Active cache object (if caching enabled)

    Example:
        >>> from configs.google_config import get_gemini_client
        >>> client = get_gemini_client()
        >>> chat = ChatGemini("gemini-3-flash-preview", client, "You are helpful")
        >>> response = chat.generate_isolated("What is 2+2?")
    """

    def __init__(self, model_id: str, gemini_client, system_prompt: Optional[str] = None,
                 use_caching: bool = True, temperature: float = 1.0, thinking_level: str = "high"):
        self.model_id = model_id
        self.client = gemini_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2
        self.use_caching = use_caching
        self.cached_content = None
        self.last_usage = None
        self.temperature = temperature
        self.thinking_level = thinking_level  # "low", "medium", "high", or None

        # Import types for configuration
        try:
            from google.genai import types
            self.types = types
        except ImportError:
            raise ImportError("google-genai package required. Install with: pip install google-genai")

    def create_cache(self, context_text: str, ttl_minutes: int = 30) -> str:
        """
        Create a context cache for repeated use.

        Args:
            context_text: The context documents to cache
            ttl_minutes: Cache time-to-live in minutes (default: 30)

        Returns:
            Cache name/ID for reference
        """
        if not self.use_caching:
            return None

        # Google Gemini requires minimum 2,048 tokens for caching (as of 2026)
        # Rough estimate: 4 chars per token
        estimated_tokens = len(context_text) // 4
        GEMINI_MIN_CACHE_TOKENS = 2048

        if estimated_tokens < GEMINI_MIN_CACHE_TOKENS:
            print(f"   ⚠️  Context too small for Gemini caching (~{estimated_tokens:,} tokens, need {GEMINI_MIN_CACHE_TOKENS:,})")
            print(f"   → Using standard (non-cached) API calls")
            self.cached_content = None
            return None

        try:
            # Build cache contents
            cache_contents = []

            # Add system instruction if present
            if self.system_prompt:
                cache_contents.append(
                    self.types.Content(
                        role="user",
                        parts=[self.types.Part(text=f"System Instructions:\n{self.system_prompt}")]
                    )
                )

            # Add context documents
            cache_contents.append(
                self.types.Content(
                    role="user",
                    parts=[self.types.Part(text=context_text)]
                )
            )

            # Create the cache
            self.cached_content = self.client.caches.create(
                model=self.model_id,
                config=self.types.CreateCachedContentConfig(
                    contents=cache_contents,
                    ttl=f"{ttl_minutes * 60}s",
                    display_name=f"context_cache_{hash(context_text[:100]) % 10000}"
                )
            )

            return self.cached_content.name

        except Exception as e:
            print(f"   Warning: Cache creation failed: {e}")
            self.cached_content = None
            return None

    def clear_cache(self):
        """Clear the current context cache."""
        if self.cached_content:
            try:
                self.client.caches.delete(name=self.cached_content.name)
            except Exception:
                pass  # Ignore deletion errors
            self.cached_content = None

    def generate_isolated(self, user_prompt: str, use_cache: bool = True) -> Dict:
        """
        Generate response for isolated prompt with retry on truncation.

        Uses exponential token limit increases on truncation retries.
        Thinking tokens count toward the output limit, so retries get higher limits.

        Args:
            user_prompt: The user's prompt/question
            use_cache: Whether to use cached context (if available)

        Returns:
            Dict with 'raw_response' key containing the text response
        """
        base_max_tokens = 65000  # Base token limit (increased for all stages to prevent truncation)
        max_retries = 5  # Number of retry attempts

        for attempt in range(max_retries):
            try:
                # Aggressive exponential scaling: 50000 → 75000 → 112500 → 168750 → 253125
                current_max_tokens = int(base_max_tokens * (1 ** attempt))

                # Add concise brevity instruction on retries
                retry_suffix = ""
                if attempt > 0:
                    retry_suffix = (
                        f"\n\nIMPORTANT: Response was truncated. You now have {current_max_tokens} tokens. "
                        f"Be concise but complete all XML tags."
                    )

                current_prompt = user_prompt + retry_suffix

                # Build generation config with attempt-specific token limit
                gen_config_params = {
                    'temperature': self.temperature,
                    'max_output_tokens': current_max_tokens,
                }

                # Add thinking config if thinking_level is specified (Gemini 3 models)
                if self.thinking_level:
                    gen_config_params['thinking_config'] = self.types.ThinkingConfig(
                        thinking_level=self.thinking_level.upper()  # Convert to "LOW", "MEDIUM", "HIGH"
                    )

                gen_config = self.types.GenerateContentConfig(**gen_config_params)

                # Use cached context if available
                if use_cache and self.cached_content:
                    cached_config_params = {
                        'cached_content': self.cached_content.name,
                        'temperature': self.temperature,
                        'max_output_tokens': current_max_tokens,
                    }

                    # Add thinking config if thinking_level is specified
                    if self.thinking_level:
                        cached_config_params['thinking_config'] = self.types.ThinkingConfig(
                            thinking_level=self.thinking_level.upper()
                        )

                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=current_prompt,
                        config=self.types.GenerateContentConfig(**cached_config_params)
                    )
                else:
                    # Build full prompt with system instruction
                    contents = []
                    if self.system_prompt:
                        contents.append(
                            self.types.Content(
                                role="user",
                                parts=[self.types.Part(text=f"Instructions:\n{self.system_prompt}")]
                            )
                        )
                        contents.append(
                            self.types.Content(
                                role="model",
                                parts=[self.types.Part(text="I understand. I'll follow these instructions.")]
                            )
                        )

                    contents.append(
                        self.types.Content(
                            role="user",
                            parts=[self.types.Part(text=current_prompt)]
                        )
                    )

                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=gen_config
                    )

                # Track usage for this attempt (accumulate across retries)
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    total_prompt = getattr(usage, 'prompt_token_count', 0) or 0
                    cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                    new_input_tokens = total_prompt - cached_tokens

                    # Accumulate usage across retries
                    if attempt == 0:
                        self.last_usage = {
                            'input_tokens': new_input_tokens,
                            'output_tokens': getattr(usage, 'candidates_token_count', 0) or 0,
                            'cache_read_input_tokens': cached_tokens,
                            'cache_creation_input_tokens': 0
                        }
                    else:
                        # Add to existing usage
                        self.last_usage['input_tokens'] += new_input_tokens
                        self.last_usage['output_tokens'] += getattr(usage, 'candidates_token_count', 0) or 0
                        self.last_usage['cache_read_input_tokens'] += cached_tokens
                else:
                    if attempt == 0:
                        self.last_usage = {
                            'input_tokens': 0,
                            'output_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'cache_creation_input_tokens': 0
                        }

                # Check if response was truncated or blocked
                is_truncated = False
                is_blocked = False
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    
                    if finish_reason:
                        from google.genai.types import FinishReason
                        if finish_reason not in (FinishReason.STOP, 1):
                            if finish_reason in (FinishReason.SAFETY, 3):
                                is_blocked = True
                                print(f"🚫 Response BLOCKED by safety filters (finish_reason={finish_reason})")
                            else:
                                is_truncated = True
                                if attempt < max_retries - 1:
                                    print(f"⚠️ Response truncated (attempt {attempt+1}/{max_retries}), retrying with brevity instructions...")
                                else:
                                    print(f"⚠️ Response truncated after {max_retries} attempts (finish_reason={finish_reason})")

                # If blocked, return error string
                if is_blocked:
                    return {"raw_response": f"ERROR::Response blocked by safety filters (reason: {finish_reason})"}

                # If not truncated, return success
                if not is_truncated:
                    text = response.text if response.text else ""
                    # Ensure text is never None
                    if text is None:
                        text = ""

                    # Retry on empty response (up to max_retries)
                    if not text or not text.strip():
                        if attempt < max_retries - 1:
                            print(f"⚠️ Empty response received from Gemini (attempt {attempt+1}/{max_retries}), retrying...")
                            continue  # Retry
                        else:
                            print(f"⚠️ Empty response received after {max_retries} attempts")
                            return {"raw_response": "ERROR::Empty response from Gemini after all retries"}

                    return {"raw_response": text}

                # If truncated and this was the last attempt, return partial response
                if attempt == max_retries - 1:
                    text = response.text if response.text else ""
                    # Ensure text is never None
                    if text is None:
                        text = ""
                    return {"raw_response": text}

            except Exception as e:
                # On error, set default usage and return error
                if attempt == 0:
                    self.last_usage = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'cache_creation_input_tokens': 0
                    }
                return {"raw_response": f"ERROR::{str(e)}"}

    def generate_with_context(self, context_text: str, question_text: str) -> Dict:
        """
        Generate response with explicit context (no caching).

        Useful for one-off questions or when caching is disabled.

        Args:
            context_text: Context documents
            question_text: The question/prompt

        Returns:
            Dict with 'raw_response' key
        """
        full_prompt = f"{context_text}\n\n{question_text}"

        # Temporarily disable cache for this call
        old_cache = self.cached_content
        self.cached_content = None

        result = self.generate_isolated(full_prompt, use_cache=False)

        self.cached_content = old_cache
        return result


class ChatGeminiCached(ChatGemini):
    """
    Gemini client optimized for context caching workflows.

    Automatically manages cache lifecycle per topic.
    """

    def __init__(self, model_id: str, gemini_client, system_prompt: Optional[str] = None,
                 temperature: float = 1.0, thinking_level: str = "high"):
        super().__init__(model_id, gemini_client, system_prompt, use_caching=True, 
                        temperature=temperature, thinking_level=thinking_level)
        self._current_topic = None

    def set_topic_context(self, topic_id: str, context_text: str, ttl_minutes: int = 30):
        """
        Set context for a specific topic (creates cache).

        Args:
            topic_id: Topic identifier
            context_text: Context documents for this topic
            ttl_minutes: Cache TTL
        """
        # Clear previous cache if topic changed
        if self._current_topic != topic_id:
            self.clear_cache()
            self._current_topic = topic_id
            self.create_cache(context_text, ttl_minutes)

    def reset(self):
        """Reset client state (clear cache)."""
        self.clear_cache()
        self._current_topic = None


# ============================================================================
# OPENAI CLIENT
# ============================================================================

class ChatOpenAI:
    """
    OpenAI client with structured output support.

    Supports GPT 5.2 and other OpenAI models with:
    - Standard chat completions
    - JSON mode for structured output
    - Automatic prompt caching (for prompts > 1024 tokens)
    - Usage tracking for cost calculation

    Attributes:
        model_id: OpenAI model identifier (e.g., "gpt-5.2")
        openai_client: OpenAI client instance
        system_prompt: Optional system-level instructions
        max_retries: Maximum retry attempts on failures
        base_delay: Base delay in seconds for exponential backoff
        temperature: Sampling temperature (default: 1.0)

    Example:
        >>> from configs.openai_config import get_openai_client
        >>> client = get_openai_client()
        >>> chat = ChatOpenAI("gpt-5.2", client, "You are helpful")
        >>> response = chat.generate_isolated([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model_id: str, openai_client, system_prompt: Optional[str] = None,
                 temperature: float = 1.0, reasoning_effort: str = None):
        self.model_id = model_id
        self.openai_client = openai_client
        self.system_prompt = system_prompt
        self.max_retries = 4
        self.base_delay = 2
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # "low", "medium", "high" for o1/o3/gpt-5.2
        self.last_usage = None

    def generate_isolated(self, prompt: str, use_cache: bool = True, n: int = 1) -> Dict:
        """
        Generate response for isolated prompt.

        OpenAI automatically caches prompts > 1024 tokens.
        Cache hits are tracked via usage.prompt_tokens_details.cached_tokens.

        Args:
            prompt: The user's prompt/question
            use_cache: Whether caching is desired (OpenAI caches automatically)
            n: Number of completions to generate (default=1). When n>1, returns
               a list of responses under 'raw_responses' key.

        Returns:
            Dict with 'raw_response' (n=1) or 'raw_responses' (n>1) key
        """
        try:
            messages = []

            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            # Build API parameters
            api_params = {
                "model": self.model_id,
                "messages": messages,
                "max_completion_tokens": 16000,
                "temperature": self.temperature
            }

            if n > 1:
                api_params["n"] = n

            # Add reasoning_effort if specified (for o1/o3/gpt-5.2 models)
            if self.reasoning_effort:
                api_params["reasoning_effort"] = self.reasoning_effort

            response = self.openai_client.chat.completions.create(**api_params)

            # Track usage for cost calculation (including reasoning tokens)
            if response.usage:
                usage = response.usage
                # OpenAI provides cached_tokens in prompt_tokens_details
                cached_tokens = 0
                if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

                # For reasoning models, completion_tokens includes reasoning tokens
                # OpenAI charges for reasoning tokens as part of output
                self.last_usage = {
                    'input_tokens': usage.prompt_tokens,  # Total prompt tokens (CostEstimator will subtract cached)
                    'output_tokens': usage.completion_tokens,  # Includes reasoning tokens for o1/o3/gpt-5.2
                    'cache_read_input_tokens': cached_tokens,
                    'cache_creation_input_tokens': 0  # OpenAI doesn't report this separately
                }

            if n > 1:
                texts = [
                    (choice.message.content if choice.message.content else "")
                    for choice in response.choices
                ]
                return {"raw_responses": texts}
            else:
                text = response.choices[0].message.content
                return {"raw_response": text if text else ""}

        except Exception as e:
            self.last_usage = {
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_read_input_tokens': 0,
                'cache_creation_input_tokens': 0
            }
            if n > 1:
                return {"raw_responses": [f"ERROR::{str(e)}"] * n}
            return {"raw_response": f"ERROR::{str(e)}"}

    def generate_structured(self, messages: List[Dict], output_schema: Dict = None) -> Dict:
        """
        Generate structured output using JSON mode.

        Args:
            messages: List of message dicts in OpenAI format
                     Example: [{"role": "user", "content": "..."}]
            output_schema: JSON schema (used for validation, not enforced by API)

        Returns:
            Dict: Parsed JSON response
        """
        if output_schema is None:
            output_schema = DEFAULT_OUTPUT_SCHEMA

        try:
            # Add system prompt if not already present
            full_messages = []
            if self.system_prompt and (not messages or messages[0].get("role") != "system"):
                full_messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            full_messages.extend(messages)

            # Build API parameters
            api_params = {
                "model": self.model_id,
                "messages": full_messages,
                "max_completion_tokens": 16000,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"}
            }

            # Add reasoning_effort if specified
            if self.reasoning_effort:
                api_params["reasoning_effort"] = self.reasoning_effort

            response = self.openai_client.chat.completions.create(**api_params)

            # Track usage (including reasoning tokens)
            if response.usage:
                usage = response.usage
                cached_tokens = 0
                if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

                self.last_usage = {
                    'input_tokens': usage.prompt_tokens,  # Total prompt tokens (CostEstimator will subtract cached)
                    'output_tokens': usage.completion_tokens,  # Includes reasoning tokens
                    'cache_read_input_tokens': cached_tokens,
                    'cache_creation_input_tokens': 0
                }

            text = response.choices[0].message.content
            return json.loads(text) if text else {}

        except Exception as e:
            raise ValueError(f"Structured output failed: {str(e)}")


class ChatOpenAICached(ChatOpenAI):
    """
    OpenAI client optimized for context caching workflows.

    OpenAI automatically caches prompts > 1024 tokens for ~1 hour.
    This class tracks topic context to maximize cache hits by
    keeping context consistent within topics.

    Attributes:
        _current_topic: Currently active topic ID
        _cached_context: Cached context text for the current topic
    """

    def __init__(self, model_id: str, openai_client, system_prompt: Optional[str] = None,
                 temperature: float = 1.0, reasoning_effort: str = None):
        super().__init__(model_id, openai_client, system_prompt, temperature, reasoning_effort)
        self._current_topic = None
        self._cached_context = None

    def set_topic_context(self, topic_id: str, context_text: str, ttl_minutes: int = 60):
        """
        Set context for a specific topic.

        OpenAI automatically caches prompts > 1024 tokens.
        This method stores the context to ensure consistency across questions.

        Args:
            topic_id: Topic identifier
            context_text: Context documents for this topic
            ttl_minutes: Ignored (OpenAI manages cache TTL automatically)
        """
        if self._current_topic != topic_id:
            self._current_topic = topic_id
            self._cached_context = context_text

    def create_cache(self, context_text: str, ttl_minutes: int = 60) -> str:
        """
        Store context for caching (OpenAI caches automatically).

        Args:
            context_text: The context documents to cache
            ttl_minutes: Ignored (OpenAI manages TTL)

        Returns:
            Cache identifier (topic_id or hash)
        """
        self._cached_context = context_text
        # Return a hash as cache identifier
        return f"cache_{hash(context_text[:100]) % 10000}"

    def clear_cache(self):
        """Clear the current context cache."""
        self._cached_context = None
        self._current_topic = None

    def reset(self):
        """Reset client state."""
        self.clear_cache()


# ============================================================================
# CLIENT REGISTRY
# ============================================================================

LLM_CLIENTS = {
    "claude": ChatClaude,
    "claude_uncached": ChatClaudeUncached,
    "claude_thinking": ChatClaudeThinking,
    "llama": ChatLlama,
    "deepseek": ChatDeepSeek,
    "deepseek_v31": ChatDeepSeekV31,
    "kimi": ChatKimi,
    "gemini": ChatGemini,
    "gemini_cached": ChatGeminiCached,
    "openai": ChatOpenAI,
    "openai_cached": ChatOpenAICached,
}


def get_client_class(model_family: str):
    """
    Get the appropriate client class for a model family.

    Args:
        model_family: One of "claude", "claude_uncached", "claude_thinking", 
                     "llama", "deepseek", "deepseek_v31"

    Returns:
        Client class

    Raises:
        ValueError: If model_family is not recognized
        
    Example:
        >>> ClientClass = get_client_class("claude")
        >>> client = ClientClass(model_id, bedrock_client, system_prompt)
        >>> 
        >>> # Use structured output (100% reliable JSON)
        >>> result = client.generate_structured(prompt, schema)
        >>> print(result)  # Already parsed dict!
    """
    if model_family not in LLM_CLIENTS:
        raise ValueError(f"Unknown model family: {model_family}. "
                        f"Available: {list(LLM_CLIENTS.keys())}")
    return LLM_CLIENTS[model_family]
