"""
OpenAI inference implementations

Supports:
- OpenAIInference: Basic inference without explicit caching
- OpenAICachedInference: Optimized with context caching per topic
- run_openai_sc_inference: Self-consistency via OpenAI n parameter,
  saving each sample as a separate experiment directory
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import BaseInference, extract_output


# ============================================================================
# OPENAI INFERENCE (No Caching)
# ============================================================================

class OpenAIInference(BaseInference):
    """
    Basic OpenAI inference without explicit context caching.

    Use this when:
    - Context is small (< 1024 tokens)
    - Each question has unique context
    - No repeated context across questions

    Note: OpenAI automatically caches prompts > 1024 tokens for ~1 hour.
    """

    def __init__(self, openai_client, model_id: str, system_prompt: str = None,
                 temperature: float = 1.0):
        """
        Args:
            openai_client: ChatOpenAI or ChatOpenAICached instance
            model_id: Model ID (e.g., "gpt-5.2")
            system_prompt: Optional system prompt template
            temperature: Sampling temperature (default: 1.0)
        """
        super().__init__(chat_client=openai_client, system_prompt=system_prompt)

        self.client = openai_client
        self.model_id = model_id
        self.temperature = temperature

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> str:
        """Format prompt for OpenAI inference."""
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
        """Call OpenAI model with prompt and track usage."""
        response = self.client.generate_isolated(prompt)

        # Track usage for cost calculation
        if hasattr(self.client, 'last_usage') and self.client.last_usage:
            self.chat.last_usage = self.client.last_usage

        raw_response = response.get('raw_response', '')
        return raw_response if raw_response else ""

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse OpenAI response to extract analysis and answer."""
        return extract_output(response)

    def get_model_name(self) -> str:
        """Get display name for progress bar."""
        return "OpenAI"

    def get_progress_color(self) -> str:
        """Get progress bar color."""
        return "green"


# ============================================================================
# OPENAI CACHED INFERENCE (Context Caching per Topic)
# ============================================================================

class OpenAICachedInference(BaseInference):
    """
    OpenAI inference with context caching per topic.

    Optimized for SemEval Task 12 where:
    - Multiple questions share the same topic context
    - Context is large (> 1024 tokens)
    - OpenAI automatically caches matching prompt prefixes

    Caching strategy:
    - First question per topic: May warm cache
    - Subsequent questions: Benefit from automatic cache (if prompts match)
    """

    def __init__(self, openai_client, model_id: str, cache_ttl_minutes: int = 60,
                 system_prompt: str = None, temperature: float = 1.0):
        """
        Args:
            openai_client: ChatOpenAICached instance
            model_id: Model ID
            cache_ttl_minutes: Ignored (OpenAI manages cache TTL)
            system_prompt: The prompt template with format instructions
            temperature: Sampling temperature (default: 1.0)
        """
        super().__init__(chat_client=openai_client, system_prompt=system_prompt)

        self.client = openai_client
        self.model_id = model_id
        self.cache_ttl_minutes = cache_ttl_minutes
        self.temperature = temperature

    def supports_caching(self) -> bool:
        """Enable topic-based grouping for cache efficiency."""
        return True

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries."""
        if retry_idx == 0:
            return ""

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
                "\n\n RETRY #2 - STRICT FORMAT ENFORCEMENT\n\n"
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
                "6. DO NOT write 'Option A' in <answer> - just the letter 'A'"
            )
        else:
            return (
                "\n\n FINAL RETRY - PARSING FAILED MULTIPLE TIMES\n\n"
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
        """Format prompt with XML tags for OpenAI."""
        context_text = "\n\n".join([
            f"<document_{j+1}>{doc['content']}</document_{j+1}>"
            for j, doc in enumerate(context_docs)
        ])

        options_text = "\n".join([
            f"<option_{opt.lower()}>{question[f'option_{opt}']}</option_{opt.lower()}>"
            for opt in ["A", "B", "C", "D"]
        ])

        prompt = f"""<context_documents>
{context_text}
</context_documents>

<target_event>{question['target_event']}</target_event>

<options>
{options_text}
</options>"""

        return prompt

    def call_model(self, prompt: str) -> str:
        """Call OpenAI model with the prompt and track usage."""
        response = self.client.generate_isolated(prompt)

        # Track usage for cost calculation
        if hasattr(self.client, 'last_usage') and self.client.last_usage:
            self.chat.last_usage = self.client.last_usage

        raw_response = response.get('raw_response', '')
        return raw_response if raw_response else ""

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse OpenAI response to extract analysis and answer."""
        return extract_output(response)

    def get_model_name(self) -> str:
        """Get display name for progress bar."""
        return "OpenAI-Cached"

    def get_progress_color(self) -> str:
        """Get progress bar color."""
        return "green"


# ============================================================================
# RUN OPENAI INFERENCE - Helper Function
# ============================================================================

def run_openai_inference(
    chat,
    questions: List[Dict],
    docs: List[Dict],
    mode: str = "cached",
    sleep_seconds: float = 0.0,
    num_threads: int = 1,
    experiment_path: str = None,
    graph_rag_data: Dict = None,
    context_cache = None,
    temperature: float = None
) -> Dict:
    """
    Run OpenAI inference with specified mode (simple or cached).

    Args:
        chat: OpenAI chat client instance (ChatOpenAI or ChatOpenAICached)
        questions: List of question dicts
        docs: List of document dicts grouped by topic_id
        mode: "simple" (no caching) or "cached" (context caching)
        sleep_seconds: Sleep between requests (rate limiting)
        num_threads: Number of parallel threads
        experiment_path: Path to save experiment outputs
        graph_rag_data: Graph RAG data for dynamic retrieval
        context_cache: Preprocessed context cache (takes priority)
        temperature: Sampling temperature

    Returns:
        Results dictionary with predictions, analyses, and metadata
    """
    temp = temperature if temperature is not None else getattr(chat, 'temperature', 1.0)
    system_prompt = getattr(chat, 'system_prompt', None)

    if mode == "simple":
        inference = OpenAIInference(chat, chat.model_id, system_prompt=system_prompt,
                                    temperature=temp)
    elif mode == "cached":
        inference = OpenAICachedInference(chat, chat.model_id, cache_ttl_minutes=60,
                                          system_prompt=system_prompt,
                                          temperature=temp)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'cached'")

    # Set Graph RAG data if provided
    if graph_rag_data is not None:
        inference.set_graph_rag(graph_rag_data)

    # Set preprocessed context cache if provided (takes priority over graph_rag)
    if context_cache is not None:
        inference.set_context_cache(context_cache)

    # Set experiment path for incremental saving
    if experiment_path is not None:
        inference.experiment_path = experiment_path

    # Set number of threads
    inference.num_threads = num_threads

    # Run inference with base class logic
    return inference.run(questions, docs, sleep_seconds)


# ============================================================================
# RUN OPENAI SC INFERENCE - Self-Consistency via n parameter
# ============================================================================

def run_openai_sc_inference(
    chat,
    questions: List[Dict],
    docs: List[Dict],
    sc_samples: int = 3,
    sc_temperature: float = 1.0,
    mode: str = "cached",
    sleep_seconds: float = 0.0,
    graph_rag_data: Dict = None,
    context_cache = None,
    temperature: float = None,
    experiment_paths: List[str] = None
) -> List[Dict]:
    """
    Run OpenAI inference using the n parameter to get multiple completions per call.

    Returns a list of N result dicts (one per sample), each containing
    {predictions, analyses, thinkings, cost_tracker}. The caller saves each as a separate
    experiment directory for later combination via combine_experiments_sc.py.

    Args:
        chat: OpenAI chat client instance
        questions: List of question dicts
        docs: List of document dicts grouped by topic_id
        sc_samples: Number of completions per question (default: 3)
        sc_temperature: Temperature for SC sampling (default: 1.0, OpenAI recommended)
        mode: "simple" or "cached"
        sleep_seconds: Sleep between requests
        graph_rag_data: Graph RAG data for dynamic retrieval
        context_cache: Preprocessed context cache
        temperature: Base temperature (overridden by sc_temperature)
        experiment_paths: List of N experiment paths for incremental saving (optional)

    Returns:
        List of N result dicts, one per sample index
    """
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from src.utils.cost_tracker import CostTracker

    system_prompt = getattr(chat, 'system_prompt', None)

    # Create a temporary inference instance for format_prompt and parse_response
    temp = sc_temperature
    if mode == "simple":
        inference = OpenAIInference(chat, chat.model_id, system_prompt=system_prompt,
                                    temperature=temp)
    else:
        inference = OpenAICachedInference(chat, chat.model_id, cache_ttl_minutes=60,
                                          system_prompt=system_prompt,
                                          temperature=temp)

    # Set context sources
    if graph_rag_data is not None:
        inference.set_graph_rag(graph_rag_data)
    if context_cache is not None:
        inference.set_context_cache(context_cache)
    # Prepare topic docs
    topic2docs = {}
    for d in docs:
        topic_id = d.get("topic_id")
        if topic_id not in topic2docs:
            topic2docs[topic_id] = []
        topic2docs[topic_id].extend(d.get("docs", []))

    # Initialize N result sets with cost trackers
    # Use a single cost tracker for the total run (costs are shared across samples)
    cost_tracker = CostTracker(chat.model_id)
    sample_results = []
    for i in range(sc_samples):
        sample_results.append({
            "predictions": {},
            "analyses": {},
            "thinkings": {},
        })

    # Override temperature for SC sampling
    original_temp = chat.temperature
    chat.temperature = sc_temperature

    print(f"\n🔀 Self-Consistency mode: n={sc_samples}, temperature={sc_temperature}")
    print(f"   Each API call returns {sc_samples} completions\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]OpenAI-SC"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing", total=len(questions))

        for q in questions:
            q_id = q["id"]

            # Resolve context docs
            ctx_docs = topic2docs.get(q["topic_id"], [])
            if inference.context_cache:
                context_str = inference._get_context_from_cache(q)
                if context_str:
                    ctx_docs = [{'content': context_str}]
            elif inference.graph_rag_data:
                ctx_docs = inference._retrieve_with_graph_rag(q)

            # Format the prompt
            prompt = inference.format_prompt(q, ctx_docs)

            # Single API call with n=sc_samples
            try:
                response = chat.generate_isolated(prompt, n=sc_samples)
                raw_responses = response.get('raw_responses', [])

                # Track cost via CostTracker (pass the chat object which has last_usage)
                if hasattr(chat, 'last_usage') and chat.last_usage:
                    cost_tracker.track("sc_inference", chat)

                # Parse each completion
                for i, raw_text in enumerate(raw_responses):
                    parsed = extract_output(raw_text)
                    answer = parsed.get('answer', ['Fail'])
                    analysis = parsed.get('analysis', '')

                    if isinstance(answer, list):
                        pred = ','.join(answer) if answer else 'Fail'
                    else:
                        pred = answer if answer else 'Fail'

                    sample_results[i]["predictions"][q_id] = pred
                    sample_results[i]["analyses"][q_id] = analysis

            except Exception as e:
                print(f"   Error on {q_id}: {e}")
                for i in range(sc_samples):
                    sample_results[i]["predictions"][q_id] = "Fail"
                    sample_results[i]["analyses"][q_id] = f"ERROR: {e}"

            # Incremental saving after each question
            if experiment_paths:
                for i in range(sc_samples):
                    if i < len(experiment_paths) and experiment_paths[i]:
                        results_path = Path(experiment_paths[i]) / "results.json"
                        with open(results_path, 'w', encoding='utf-8') as f:
                            json.dump(sample_results[i], f, indent=2, ensure_ascii=False)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            progress.advance(task)

    # Restore original temperature
    chat.temperature = original_temp

    # Print cost summary
    cost_tracker.print_summary()

    # Attach cost tracker to each sample (divided equally)
    cost_summary = cost_tracker.get_summary()
    for i in range(sc_samples):
        per_sample_cost = dict(cost_summary)
        # Divide total cost equally across samples for per-sample tracking
        for key in ['total_cost', 'total_input_tokens', 'total_output_tokens',
                     'total_cache_write_tokens', 'total_cache_read_tokens', 'total_tokens']:
            if key in per_sample_cost:
                per_sample_cost[key] = per_sample_cost[key] / sc_samples
        sample_results[i]["cost_tracker"] = per_sample_cost

    return sample_results
