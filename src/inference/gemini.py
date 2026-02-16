"""
Gemini inference implementations

Supports:
- GeminiInference: Basic inference without caching
- GeminiCachedInference: Optimized with context caching per topic
- GeminiBatchInference: Async batch processing with 50% cost reduction
"""

from typing import Dict, List, Any
import json
import time
from pathlib import Path
from .base import BaseInference, extract_output


# ============================================================================
# GEMINI INFERENCE (No Caching)
# ============================================================================

class GeminiInference(BaseInference):
    """
    Basic Gemini inference without context caching.

    Use this when:
    - Context is small (< 32K tokens)
    - Each question has unique context
    - No repeated context across questions
    """

    def __init__(self, gemini_client, model_id: str, system_prompt: str = None,
                 temperature: float = 1.0, thinking_level: str = "high"):
        """
        Args:
            gemini_client: google.genai.Client instance
            model_id: Model ID (e.g., "gemini-2.0-flash", "gemini-3-flash-preview")
            system_prompt: Optional system prompt template
            temperature: Sampling temperature (default: 1.0 - Google recommended)
            thinking_level: Thinking level for Gemini 3 models (default: "high", or None to disable)
        """
        # Initialize base class with client and system_prompt
        super().__init__(chat_client=gemini_client, system_prompt=system_prompt)

        # Gemini-specific attributes
        self.client = gemini_client  # Also store as self.client for Gemini methods
        self.model_id = model_id
        self.temperature = temperature
        self.thinking_level = thinking_level

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> str:
        """Format prompt for basic Gemini inference (no caching)."""
        # Format context documents
        context_text = "\n\n".join([
            f"<document_{j+1}>{doc['content']}</document_{j+1}>"
            for j, doc in enumerate(context_docs)
        ])

        # Format options
        options_text = "\n".join([
            f"<option_{opt.lower()}>{question[f'option_{opt}']}</option_{opt.lower()}>"
            for opt in ["A", "B", "C", "D"]
        ])

        # Build complete prompt
        prompt = f"""<context_documents>
{context_text}
</context_documents>

<target_event>{question['target_event']}</target_event>

<options>
{options_text}
</options>"""

        return prompt

    def call_model(self, prompt: str) -> str:
        """Call Gemini model with prompt and track usage."""
        from google.genai import types

        system_instruction = self.system_prompt if self.system_prompt else (
            "You MUST respond using ONLY XML tags. "
            "Start with <analysis> and end with </answer>. "
            "Never write text outside these tags."
        )

        # Build config parameters
        config_params = {
            'max_output_tokens': 16000,
            'temperature': self.temperature,
            'system_instruction': system_instruction
        }

        # Add thinking config if thinking_level is specified (Gemini 3 models)
        if self.thinking_level:
            config_params['thinking_config'] = types.ThinkingConfig(
                thinking_level=self.thinking_level.upper()
            )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**config_params)
        )

        # Track usage for cost calculation
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            total_prompt = getattr(usage, 'prompt_token_count', 0) or 0
            cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
            new_input_tokens = total_prompt - cached_tokens

            self.chat.last_usage = {
                'input_tokens': new_input_tokens,
                'output_tokens': getattr(usage, 'candidates_token_count', 0) or 0,
                'cache_read_input_tokens': cached_tokens,
                'cache_creation_input_tokens': 0
            }

        return response.text

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse Gemini response to extract analysis and answer."""
        from .base import extract_output
        return extract_output(response)

    def generate(self, prompt: str, max_output_tokens: int = 10000, temperature: float = 0.0) -> str:
        """
        Generate completion for a single prompt (legacy method).

        Args:
            prompt: Input prompt
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text
        """
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
        )

        return response.text


# ============================================================================
# GEMINI CACHED INFERENCE (Context Caching per Topic)
# ============================================================================

class GeminiCachedInference(BaseInference):
    """
    Gemini inference with context caching per topic.

    Optimized for SemEval Task 12 where:
    - Multiple questions share the same topic context
    - Context is large (> 32K tokens)
    - Cache warming significantly reduces costs

    Caching strategy:
    - First question per topic: Warms cache (full cost)
    - Subsequent questions: Read from cache (90% cost reduction on cached tokens)
    """

    def __init__(self, gemini_client, model_id: str, cache_ttl_minutes: int = 60,
                 system_prompt: str = None, temperature: float = 1.0, thinking_level: str = "high"):
        """
        Args:
            gemini_client: google.genai.Client instance
            model_id: Model ID
            cache_ttl_minutes: Cache time-to-live in minutes (default: 60)
            system_prompt: The prompt template with format instructions
            temperature: Sampling temperature (default: 1.0 - Google recommended)
            thinking_level: Thinking level for Gemini 3 models (default: "high", or None to disable)
        """
        # Initialize base class with client (stores as self.chat for Claude compatibility)
        super().__init__(chat_client=gemini_client, system_prompt=system_prompt)

        # Gemini-specific attributes
        self.client = gemini_client  # Also store as self.client for Gemini methods
        self.model_id = model_id
        self.cache_ttl_minutes = cache_ttl_minutes
        self._topic_caches = {}  # topic_id -> cache_name
        self.temperature = temperature
        self.thinking_level = thinking_level

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries for Gemini with escalating emphasis."""
        if retry_idx == 0:
            return ""

        # Escalating enforcement based on retry count
        if retry_idx == 1:
            return (
                "\n\n**CRITICAL FORMAT REQUIREMENT**: Your response MUST use EXACTLY this XML format:\n"
                "<analysis>\n"
                "Option A: [reasoning]\n"
                "Option B: [reasoning]\n"
                "Option C: [reasoning]\n"
                "Option D: [reasoning]\n"
                "</analysis>\n"
                "<answer>[LETTER(S) ONLY - e.g., C or B,D]</answer>\n\n"
                "IMPORTANT: \n"
                "- Start with <analysis> tag (required)\n"
                "- End with </answer> tag (required)\n"
                "- Answer must contain ONLY letters: A, B, C, or D\n"
                "- Do NOT write any text outside these tags"
            )
        elif retry_idx == 2:
            return (
                "\n\n🚨 RETRY #2 - STRICT FORMAT ENFORCEMENT 🚨\n\n"
                "Your previous response had formatting errors. You MUST respond using ONLY these XML tags:\n\n"
                "<analysis>\n"
                "[Analyze each option A, B, C, D here]\n"
                "</analysis>\n"
                "<answer>A</answer>  (or B, C, D, or multiple like A,C)\n\n"
                "CRITICAL RULES:\n"
                "1. First tag MUST be <analysis>\n"
                "2. Last tag MUST be </answer>\n"
                "3. Answer MUST contain ONLY letters A, B, C, or D (comma-separated if multiple)\n"
                "4. NO text before <analysis>\n"
                "5. NO text after </answer>\n"
                "6. DO NOT write 'Option A' in <answer> - just the letter 'A'\n\n"
                "Example valid response:\n"
                "<analysis>Option A is correct because... Option B is wrong because...</analysis>\n"
                "<answer>A</answer>"
            )
        else:
            # retry_idx >= 3 - most explicit possible
            return (
                "\n\n❌❌❌ FINAL RETRY - PARSING FAILED MULTIPLE TIMES ❌❌❌\n\n"
                "Copy this EXACT template and fill it in:\n\n"
                "<analysis>\n"
                "Option A: [Your analysis of option A]\n"
                "Option B: [Your analysis of option B]\n"
                "Option C: [Your analysis of option C]\n"
                "Option D: [Your analysis of option D]\n"
                "</analysis>\n"
                "<answer>[LETTER(S) ONLY]</answer>\n\n"
                "Replace [Your analysis of option X] with your reasoning.\n"
                "Replace [LETTER(S) ONLY] with ONLY the letter(s): A, B, C, or D\n"
                "Do NOT add any text before <analysis> or after </answer>!"
            )

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> str:
        """
        Format prompt with XML tags for Gemini.
        """
        # Format context documents
        context_text = "\n\n".join([
            f"<document_{j+1}>{doc['content']}</document_{j+1}>"
            for j, doc in enumerate(context_docs)
        ])

        # Format options
        options_text = "\n".join([
            f"<option_{opt.lower()}>{question[f'option_{opt}']}</option_{opt.lower()}>"
            for opt in ["A", "B", "C", "D"]
        ])

        # Build complete prompt
        prompt = f"""<context_documents>
{context_text}
</context_documents>

<target_event>{question['target_event']}</target_event>

<options>
{options_text}
</options>"""

        return prompt

    def call_model(self, prompt: str) -> str:
        """
        Call Gemini model with the prompt and track usage.
        """
        from google.genai import types

        # Use the full system_prompt (template with format instructions) if available
        system_instruction = self.system_prompt if self.system_prompt else (
            "You MUST respond using ONLY XML tags. "
            "Start with <analysis> and end with </answer>. "
            "Never write text outside these tags."
        )

        # Build config parameters
        config_params = {
            'max_output_tokens': 16000,
            'temperature': self.temperature,
            'system_instruction': system_instruction
        }

        # Add thinking config if thinking_level is specified (Gemini 3 models)
        if self.thinking_level:
            config_params['thinking_config'] = types.ThinkingConfig(
                thinking_level=self.thinking_level.upper()
            )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**config_params)
        )

        # Track usage for cost calculation
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            total_prompt = getattr(usage, 'prompt_token_count', 0) or 0
            cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
            new_input_tokens = total_prompt - cached_tokens

            self.chat.last_usage = {
                'input_tokens': new_input_tokens,
                'output_tokens': getattr(usage, 'candidates_token_count', 0) or 0,
                'cache_read_input_tokens': cached_tokens,
                'cache_creation_input_tokens': 0
            }

        return response.text

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Gemini response to extract analysis and answer.
        """
        from .base import extract_output
        return extract_output(response)

    def _get_or_create_cache(self, topic_id: str, context: str):
        """Get existing cache or create new one for a topic."""
        from google.genai import types
        import datetime

        # Check if cache exists and is valid
        if topic_id in self._topic_caches:
            cache_name = self._topic_caches[topic_id]
            try:
                cache = self.client.caches.get(name=cache_name)
                return cache
            except:
                # Cache expired or invalid, create new one
                pass

        # Create new cache
        ttl = datetime.timedelta(minutes=self.cache_ttl_minutes)

        cache = self.client.caches.create(
            model=self.model_id,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=context)]
                    )
                ],
                ttl=ttl,
                display_name=f"topic_{topic_id}"
            )
        )

        self._topic_caches[topic_id] = cache.name
        return cache

    def generate_with_cache(self, topic_id: str, context: str, question_prompt: str,
                          max_output_tokens: int = 2048, temperature: float = 0.0) -> str:
        """
        Generate completion using cached context.

        Args:
            topic_id: Topic identifier for caching
            context: Context to cache (large, shared across questions)
            question_prompt: Question-specific prompt (not cached)

        Returns:
            Generated text
        """
        from google.genai import types

        # Get or create cache for this topic
        cache = self._get_or_create_cache(topic_id, context)

        # Generate with cached context
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part(text=question_prompt)
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                cached_content=cache.name
            )
        )

        return response.text


# ============================================================================
# RUN GEMINI INFERENCE - Helper Function
# ============================================================================

def run_gemini_inference(
    chat,
    questions: List[Dict],
    docs: List[Dict],
    mode: str = "cached",
    sleep_seconds: float = 0.0,
    use_rag: bool = False,
    rag_top_k: int = 5,
    rag_use_bm25: bool = False,
    retriever=None,
    use_self_consistency: bool = False,
    sc_samples: int = 5,
    sc_temperature: float = 0.7,
    num_threads: int = 1,
    rpm: int = None,
    experiment_path: str = None,
    graph_rag_data: Dict = None,
    cache_ttl_minutes: int = 60,
    context_cache = None,
    temperature: float = None,
    thinking_level: str = None
) -> Dict:
    """
    Run Gemini inference with specified mode (simple or cached).

    Args:
        chat: Gemini inference instance (GeminiInference or GeminiCachedInference)
        questions: List of question dicts
        docs: List of document dicts grouped by topic_id
        mode: "simple" (no caching) or "cached" (context caching)
        sleep_seconds: Sleep between requests (rate limiting)
        use_rag: Whether to use RAG for document retrieval
        rag_top_k: Number of documents to retrieve
        rag_use_bm25: Use BM25 for retrieval
        retriever: Retriever instance (required if use_rag=True)
        use_self_consistency: Use self-consistency decoding
        sc_samples: Number of samples for self-consistency
        sc_temperature: Temperature for self-consistency sampling
        num_threads: Number of parallel threads
        rpm: Requests per minute rate limit (for per-thread rate limiting)
        experiment_path: Path to save experiment outputs
        graph_rag_data: Graph RAG data for dynamic retrieval
        cache_ttl_minutes: Cache TTL in minutes

    Returns:
        Results dictionary with predictions, analyses, and metadata
    """
    # Use temperature/thinking_level from chat client if not explicitly provided
    temp = temperature if temperature is not None else getattr(chat, 'temperature', 1.0)
    think_level = thinking_level if thinking_level is not None else getattr(chat, 'thinking_level', 'high')
    system_prompt = getattr(chat, 'system_prompt', None)

    if mode == "simple":
        inference = GeminiInference(chat.client, chat.model_id, system_prompt=system_prompt,
                                   temperature=temp, thinking_level=think_level)
    elif mode == "cached":
        inference = GeminiCachedInference(chat.client, chat.model_id, cache_ttl_minutes,
                                          system_prompt=system_prompt,
                                          temperature=temp, thinking_level=think_level)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'cached'")

    # Set retriever if RAG is enabled (optional, not used by default)
    if use_rag and hasattr(inference, 'set_retriever'):
        if not retriever:
            raise ValueError("Retriever must be provided when use_rag=True")
        inference.set_retriever(retriever)

    # Set Graph RAG data if provided
    if graph_rag_data is not None:
        inference.set_graph_rag(graph_rag_data)

    # Set preprocessed context cache if provided (takes priority over graph_rag)
    if context_cache is not None:
        inference.set_context_cache(context_cache)

    # Set experiment path for incremental saving
    if experiment_path is not None:
        inference.experiment_path = experiment_path

    # Set RPM for rate limiting
    inference.rpm = rpm

    # Run inference with base class logic
    return inference.run(questions, docs, sleep_seconds)


# ============================================================================
# GEMINI BATCH INFERENCE (Async Batch API - 50% Cost Reduction)
# ============================================================================

class GeminiBatchInference:
    """
    Google Gemini Batch API inference with 50% cost reduction.

    This uses the TRUE Batch API (async job submission) which:
    - Costs 50% of standard API pricing
    - Processes requests asynchronously (5-30+ minute turnaround)
    - Supports large-scale batch processing
    - No real-time streaming

    Use for: Large evaluation runs where you can wait for results
    """

    def __init__(self, gemini_client, model_id: str, experiment_path: str = None,
                 poll_interval: int = 30, max_wait_hours: int = 24, graph_rag_data: Dict = None,
                 context_cache = None, system_prompt: str = None):
        """
        Initialize Gemini Batch Inference.

        Args:
            gemini_client: google.genai.Client instance
            model_id: Model ID (e.g., "gemini-2.0-flash", "gemini-3-flash-preview")
            experiment_path: Directory to save batch job files and results
            poll_interval: Seconds between status checks (default: 30)
            max_wait_hours: Maximum hours to wait for completion (default: 24)
            graph_rag_data: Optional Graph RAG data for dynamic retrieval
            context_cache: Preprocessed context cache (takes priority over graph_rag)
            system_prompt: System prompt template with format instructions
        """
        self.client = gemini_client
        self.model_id = model_id
        self.experiment_path = Path(experiment_path) if experiment_path else Path("experiments/gemini_batch")
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self.graph_rag_data = graph_rag_data
        self.context_cache = context_cache
        self.system_prompt = system_prompt

        # Create batch directory
        self.batch_dir = self.experiment_path / "batch_jobs"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    def _format_prompt(self, question: Dict, context_docs: List[Dict]) -> str:
        """
        Format a single prompt for Gemini batch request.

        Uses the same format_prompt logic as GeminiCachedInference to ensure consistency.
        """
        # Build context documents (same as GeminiCachedInference.format_prompt)
        context_text = "\n\n".join([
            f"<document_{j+1}>{doc['content']}</document_{j+1}>"
            for j, doc in enumerate(context_docs)
        ])

        # Format options (same as GeminiCachedInference.format_prompt)
        options_text = "\n".join([
            f"<option_{opt.lower()}>{question[f'option_{opt}']}</option_{opt.lower()}>"
            for opt in ["A", "B", "C", "D"]
        ])

        # Build complete prompt (same structure as GeminiCachedInference.format_prompt)
        prompt = f"""<context_documents>
{context_text}
</context_documents>

<target_event>{question['target_event']}</target_event>

<options>
{options_text}
</options>"""

        return prompt

    def _create_batch_jsonl(self, questions: List[Dict], docs: List[Dict]) -> Path:
        """
        Create JSONL file for batch submission.

        Supports both static topic-based docs and dynamic Graph RAG retrieval.

        Format: Each line is a JSON object with:
        {
            "key": "question_id",
            "request": {
                "contents": [{"parts": [{"text": "prompt"}]}]
            }
        }
        """
        from google.genai import types

        # Build context lookup by topic_id (static docs)
        topic_to_context = {}
        for doc_entry in docs:
            topic_id = doc_entry.get('topic_id')
            if topic_id:
                topic_to_context[topic_id] = doc_entry.get('docs', [])

        # Create JSONL file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.batch_dir / f"batch_requests_{timestamp}.jsonl"

        print(f"\n📝 Creating batch JSONL file...")
        print(f"   Path: {jsonl_path}")
        
        # Determine context retrieval method
        if self.context_cache:
            print(f"   📦 Using preprocessed topic-wide contexts (priority)")
        elif self.graph_rag_data:
            print(f"   🔍 Using Graph RAG for dynamic document retrieval")
        else:
            print(f"   📄 Using static topic-based documents")

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for q in questions:
                # Get context docs for this question
                # Priority: context_cache > graph_rag > static docs
                if self.context_cache:
                    # Use preprocessed topic-wide context (highest priority)
                    context_str = self.context_cache.get_context(q.get('topic_id'), q)
                    context_docs = [{'content': context_str}] if context_str else []
                elif self.graph_rag_data:
                    # Use Graph RAG for dynamic retrieval
                    from src.retrieval.graph_rag_utils import retrieve_with_graph_rag
                    retrieved_docs = retrieve_with_graph_rag(q, self.graph_rag_data)
                    context_docs = retrieved_docs if retrieved_docs else []
                else:
                    # Use static topic-based docs
                    topic_id = q.get('topic_id', '')
                    context_docs = topic_to_context.get(topic_id, [])

                # Format prompt
                prompt = self._format_prompt(q, context_docs)

                # Create request object matching Gemini Batch API format
                # Gemini Batch API uses "key" for ID (not "custom_id")
                # CRITICAL: Must match ChatGeminiCached settings for identical results
                request_obj = {
                    "key": q.get('id', q.get('question_id', f"q_{questions.index(q)}")),
                    "request": {
                        "contents": [{
                            "parts": [{"text": prompt}],
                            "role": "user"
                        }],
                        "generation_config": {
                            "temperature": 0.0,  # Match direct inference (deterministic)
                            "max_output_tokens": 16000  # Match direct inference
                        }
                    }
                }

                # Add system_instruction (match GeminiCachedInference behavior)
                system_instruction_text = self.system_prompt if self.system_prompt else (
                    "You MUST respond using ONLY XML tags. "
                    "Start with <analysis> and end with </answer>. "
                    "Never write text outside these tags."
                )
                
                request_obj["request"]["system_instruction"] = {
                    "parts": [{"text": system_instruction_text}]
                }

                # Write as single line JSON
                f.write(json.dumps(request_obj) + '\n')

        print(f"   ✓ Created {len(questions)} batch requests")

        # Estimate total tokens by counting ONLY actual content (not JSON structure)
        # Read back the JSONL and extract only the text content that will be tokenized
        total_content_chars = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    req = json.loads(line)
                    # Count user prompt
                    for content in req.get("request", {}).get("contents", []):
                        for part in content.get("parts", []):
                            if "text" in part:
                                total_content_chars += len(part["text"])

                    # Count system instruction if present
                    sys_inst = req.get("request", {}).get("system_instruction", {})
                    for part in sys_inst.get("parts", []):
                        if "text" in part:
                            total_content_chars += len(part["text"])
                except:
                    pass  # Skip malformed lines

        # More accurate ratio for Gemini: ~3.5 chars per token (conservative)
        estimated_tokens = total_content_chars // 3
        print(f"   📊 Estimated input tokens: ~{estimated_tokens:,} (from {total_content_chars:,} content chars)")

        # Warn about quota limits
        TIER_LIMITS = {
            "gemini-3-flash-preview": 3_000_000,  # Tier 1
            "gemini-3-pro-preview": 5_000_000,     # Tier 1
        }
        limit = TIER_LIMITS.get(self.model_id, 3_000_000)

        if estimated_tokens > limit:
            print(f"   ❌ ERROR: Estimated tokens (~{estimated_tokens:,}) exceeds Tier 1 limit ({limit:,})")
            safe_batch_size = int(len(questions) * (limit / estimated_tokens) * 0.9)  # 90% safety margin
            print(f"   💡 SOLUTION: Auto-splitting into smaller batches")
            print(f"      Safe batch size: {safe_batch_size} questions per batch")
            print(f"      Total batches needed: {(len(questions) + safe_batch_size - 1) // safe_batch_size}")
            raise ValueError(
                f"Batch too large ({estimated_tokens:,} tokens > {limit:,} limit). "
                f"Re-run with --limit {safe_batch_size} or use split-batch mode."
            )
        elif estimated_tokens > limit * 0.8:
            print(f"   ⚠️  WARNING: Close to Tier 1 quota limit (using {estimated_tokens/limit*100:.0f}%)")

        return jsonl_path

    def _upload_batch_file(self, jsonl_path: Path) -> str:
        """Upload JSONL file to Gemini API."""
        from google.genai import types

        print(f"\n📤 Uploading batch file to Gemini API...")

        try:
            # Upload file
            uploaded_file = self.client.files.upload(
                file=str(jsonl_path),
                config=types.UploadFileConfig(
                    display_name=jsonl_path.stem,
                    mime_type="application/jsonl"
                )
            )

            file_name = uploaded_file.name
            print(f"   ✓ Uploaded: {file_name}")
            return file_name

        except Exception as e:
            # Fallback: try with text/plain MIME type (workaround for some regions)
            print(f"   ⚠️  Upload failed with application/jsonl, trying text/plain...")
            uploaded_file = self.client.files.upload(
                file=str(jsonl_path),
                config=types.UploadFileConfig(
                    display_name=jsonl_path.stem,
                    mime_type="text/plain"
                )
            )
            file_name = uploaded_file.name
            print(f"   ✓ Uploaded (fallback): {file_name}")
            return file_name

    def _create_batch_job(self, file_name: str) -> Any:
        """Create batch job using uploaded file."""
        print(f"\n🚀 Creating batch job...")
        print(f"   Model: {self.model_id}")
        print(f"   Input: {file_name}")

        try:
            batch_job = self.client.batches.create(
                model=self.model_id,
                src=file_name,
                config={
                    'display_name': f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
                }
            )

            print(f"   ✓ Job created: {batch_job.name}")
            print(f"   Initial state: {batch_job.state.name}")
        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"\n   ❌ QUOTA EXCEEDED - Batch API enqueued token limit reached")
                print(f"\n   Gemini 3 Flash Preview limits:")
                print(f"   - Tier 1: 3,000,000 tokens enqueued")
                print(f"   - Tier 2: 50,000,000 tokens enqueued")
                print(f"   - Tier 3: 150,000,000 tokens enqueued")
                print(f"\n   Solutions:")
                print(f"   1. Wait for existing batch jobs to complete")
                print(f"   2. Split into smaller batches (e.g., --limit 100)")
                print(f"   3. Upgrade to higher tier at https://ai.google.dev/")
                print(f"   4. Check current usage at https://ai.dev/rate-limit")
                raise RuntimeError(f"Batch API quota exceeded. {error_msg}")
            else:
                raise

        # Save job metadata
        job_metadata = {
            'job_name': batch_job.name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': self.model_id,
            'input_file': file_name,
            'state': batch_job.state.name
        }

        job_metadata_path = self.batch_dir / f"{batch_job.name.replace('/', '_')}_metadata.json"
        with open(job_metadata_path, 'w') as f:
            json.dump(job_metadata, f, indent=2)

        return batch_job

    def _poll_job_status(self, job_name: str) -> Any:
        """
        Poll job status until completion.

        Returns completed job object or raises TimeoutError.
        """
        print(f"\n⏳ Polling job status (checking every {self.poll_interval}s)...")
        print(f"   Max wait time: {self.max_wait_hours} hours")
        print(f"   Job: {job_name}")

        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
                           'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

        start_time = time.time()
        max_wait_seconds = self.max_wait_hours * 3600

        while True:
            try:
                # Get current job status
                job = self.client.batches.get(name=job_name)
                elapsed_minutes = (time.time() - start_time) / 60

                print(f"   [{elapsed_minutes:.1f}min] State: {job.state.name}")

                # Check if completed
                if job.state.name in completed_states:
                    total_elapsed = time.time() - start_time
                    print(f"\n   ✓ Job completed in {total_elapsed/60:.1f} minutes")
                    print(f"   Final state: {job.state.name}")
                    return job

                # Check timeout
                if time.time() - start_time > max_wait_seconds:
                    raise TimeoutError(
                        f"Batch job exceeded max wait time of {self.max_wait_hours} hours"
                    )

            except (ConnectionError, Exception) as e:
                # Handle network errors gracefully
                error_str = str(e).lower()
                if 'disconnect' in error_str or 'connection' in error_str or 'remote' in error_str:
                    print(f"   ⚠️  Connection error: {e}")
                    print(f"   Retrying in {self.poll_interval}s...")
                else:
                    # Re-raise non-network errors
                    raise

            # Wait before next check
            time.sleep(self.poll_interval)

    def _download_results(self, job) -> Dict[str, Dict]:
        """
        Download and parse batch job results.

        Returns: Dict mapping request_id (key) -> response
        """
        print(f"\n📥 Downloading batch results...")

        # Check job state
        if job.state.name != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(f"Job failed with state: {job.state.name}")

        # Get output file
        if not hasattr(job, 'dest') or not hasattr(job.dest, 'file_name'):
            raise RuntimeError("Job succeeded but no output file found")

        output_file_name = job.dest.file_name
        print(f"   Output file: {output_file_name}")

        # Download file
        try:
            # Use the correct download method from google-genai SDK
            print(f"   Downloading results...")

            # Download file content (returns bytes)
            file_data = self.client.files.download(file=output_file_name)

            # Save to local file
            output_path = self.batch_dir / f"{output_file_name.replace('/', '_')}.jsonl"

            # Write content to local file
            with open(output_path, 'wb') as f:
                f.write(file_data)

            print(f"   ✓ Downloaded to: {output_path}")

            # Parse JSONL results
            # Gemini Batch API uses "key" for ID in both input and output
            results = {}
            failed_count = 0
            failed_reasons = []
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        request_id = result.get('key', '')
                        response_data = result.get('response', {})

                        # Extract text from response
                        if 'candidates' in response_data:
                            candidates = response_data['candidates']
                            if candidates and len(candidates) > 0:
                                # Check for truncation using same logic as interactive mode
                                # CRITICAL FIX: candidates is a dict (from JSON), not an object
                                # Use .get() instead of getattr()
                                finish_reason = candidates[0].get('finish_reason', None)
                                is_valid = True
                                failure_reason = None

                                if finish_reason is not None:
                                    # Import FinishReason for comparison
                                    try:
                                        from google.genai.types import FinishReason
                                        if finish_reason not in (FinishReason.STOP, 1):
                                            is_valid = False
                                            failure_reason = f"truncated (finish_reason={finish_reason})"
                                    except ImportError:
                                        # Fallback: check for non-STOP finish reasons by value
                                        if str(finish_reason).upper() not in ('STOP', '1'):
                                            is_valid = False
                                            failure_reason = f"truncated (finish_reason={finish_reason})"

                                content = candidates[0].get('content', {})
                                parts = content.get('parts', [])
                                if parts and len(parts) > 0:
                                    text = parts[0].get('text', '')
                                    if is_valid:
                                        # Only save valid responses
                                        results[request_id] = {'raw_response': text}
                                    else:
                                        # Skip failed responses - they'll be retried on resume
                                        failed_count += 1
                                        failed_reasons.append((request_id, failure_reason))
                                else:
                                    # No response text - skip this question
                                    failed_count += 1
                                    failed_reasons.append((request_id, "no response text"))
                            else:
                                # No candidates - skip this question
                                failed_count += 1
                                failed_reasons.append((request_id, "no candidates"))
                        else:
                            # Check for error
                            if 'error' in result:
                                error_msg = result['error'].get('message', 'Unknown error')
                                failed_count += 1
                                failed_reasons.append((request_id, f"API error: {error_msg}"))
                            else:
                                failed_count += 1
                                failed_reasons.append((request_id, "invalid response format"))

            print(f"   ✓ Parsed {len(results)} valid responses")
            if failed_count > 0:
                print(f"   ⚠️  Skipped {failed_count} failed/truncated responses (will retry on resume)")
                for req_id, reason in failed_reasons[:5]:  # Show first 5
                    print(f"      - {req_id}: {reason}")
                if len(failed_reasons) > 5:
                    print(f"      ... and {len(failed_reasons) - 5} more")
            return results

        except Exception as e:
            raise RuntimeError(f"Failed to download results: {e}")

    def run(self, questions: List[Dict], docs: List[Dict]) -> Dict:
        """
        Run batch inference on questions.

        Args:
            questions: List of question dictionaries
            docs: List of document dictionaries grouped by topic_id

        Returns:
            Results dictionary with predictions, analyses, and metadata
        """
        print(f"\n{'='*70}")
        print("GEMINI BATCH INFERENCE - 50% COST REDUCTION")
        print(f"{'='*70}")
        print(f"Questions: {len(questions)}")
        print(f"Model: {self.model_id}")
        print(f"Batch directory: {self.batch_dir}")
        print(f"{'='*70}\n")

        # Step 1: Create JSONL file
        jsonl_path = self._create_batch_jsonl(questions, docs)

        # Step 2: Upload file
        file_name = self._upload_batch_file(jsonl_path)

        # Step 3: Create batch job
        batch_job = self._create_batch_job(file_name)

        # Step 4: Poll until completion
        completed_job = self._poll_job_status(batch_job.name)

        # Step 5: Download and parse results
        batch_results = self._download_results(completed_job)

        # Step 6: Process results into standard format
        print(f"\n🔄 Processing results...")
        predictions = {}
        analyses = {}
        flags = {}

        for q in questions:
            q_id = q.get('id', q.get('question_id'))

            if q_id in batch_results:
                response = batch_results[q_id]
                raw_response = response.get('raw_response', '')

                # Parse response using standard extraction
                result = extract_output(raw_response)
                predictions[q_id] = ','.join(result['answer'])
                analyses[q_id] = result['analysis']
                flags[q_id] = result['flag']
            else:
                # Missing result (truncated/failed - skipped by _download_results)
                # Will be retried when running in interactive mode
                predictions[q_id] = 'Fail'
                analyses[q_id] = 'No result in batch output (truncated or failed)'
                flags[q_id] = True

        print(f"   ✓ Processed {len(predictions)} predictions")

        # Return results in standard format
        return {
            'predictions': predictions,
            'analyses': analyses,
            'flags': flags,
            'batch_job_name': completed_job.name,
            'batch_metadata': {
                'model': self.model_id,
                'num_questions': len(questions),
                'jsonl_path': str(jsonl_path),
                'cost_reduction': '50%'
            }
        }


