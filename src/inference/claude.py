"""
Claude inference implementations

All 3 Claude variants (simple, optimized, thinking) using the base class.
"""

from typing import Dict, List, Any
from .base import BaseInference, extract_output

# ============================================================================
# CLAUDE VARIANTS
# ============================================================================

# ============================================================================
# OLD IMPLEMENTATIONS (Text-based, from your Colab - NO structured output)
# ============================================================================

class ClaudeOldSimpleInference(BaseInference):
    """
    OLD Simple Claude - text-based InvokeModel, no caching.

    Matches your original claude_abductive_reasoning (no optimized grouping).
    """

    def __init__(self, chat_client, use_rag=False, rag_top_k=20, rag_use_bm25=True,
                 use_self_consistency=False, sc_samples=3, sc_temperature=0.7, num_threads=None, use_batch=False, batch_size: int = 8, experiment_path: str = None):
        super().__init__(
            chat_client,
            use_rag=use_rag,
            rag_top_k=rag_top_k,
            rag_use_bm25=rag_use_bm25,
            use_self_consistency=use_self_consistency,
            sc_samples=sc_samples,
            sc_temperature=sc_temperature,
            num_threads=num_threads,
            batch_size=batch_size,
            experiment_path=experiment_path
        )
        self.use_batch = use_batch
        self.retriever = None

    def set_retriever(self, retriever):
        self.retriever = retriever

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries."""
        if retry_idx == 0:
            return ""
        return (
            "\n\nCRITICAL: You MUST output your final answer in EXACTLY this format:\n"
            "<analysis>[Your reasoning for each option]</analysis>\n"
            "<answer>[One or more letters: A, B, C, or D separated by commas]</answer>\n"
            "Do NOT include 'Option' before letters. Just the letter(s)."
        )

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> List[Dict]:
        """
        Format prompt exactly like your Colab (messages array for InvokeModel).

        Returns: messages list for generate_isolated
        """
        # Build context (like your context_text)
        context_text = "\n\n".join([
            f"<document_{i+1}>: {doc['content']}</document_{i+1}>"
            for i, doc in enumerate(context_docs)
        ])

        # No cache_control in simple mode
        context_message = {
            "type": "text",
            "text": f"""<context_documents>
<topic_id>{question.get('topic_id', '')}</topic_id>
{context_text}
</context_documents>"""
        }

        # Question-specific content
        question_text = f"""<target_event>{question["target_event"]}</target_event>

<options>
<option_a>{question["option_A"]}</option_a>
<option_b>{question["option_B"]}</option_b>
<option_c>{question["option_C"]}</option_c>
<option_d>{question["option_D"]}</option_d>
</options>
"""

        # Messages array (like your Colab)
        messages = [{
            "role": "user",
            "content": [context_message, {"type": "text", "text": question_text}]
        }]

        return messages
    
    def call_model(self, prompt: List[Dict]) -> str:
        """
        Call OLD Claude with generate_isolated (InvokeModel, returns raw text).
        """
        return self.chat.generate_isolated(prompt)  # Returns string
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse with extract_output (like your Colab).
        """
        if not isinstance(response, str):
            response = str(response)
        
        # Your original extract_output parsing
        result = extract_output(response)
        
        return {
            "answer": result["answer"],       # Already list from extract_output
            "analysis": result["analysis"],
            "flag": result["flag"]
        }
    
    def supports_caching(self) -> bool:
        return False
    
    def get_progress_color(self) -> str:
        return "magenta"


class ClaudeOldOptimizedInference(BaseInference):
    """
    OLD Optimized Claude - text-based with topic grouping & prompt caching.

    - Groups by topic_id
    - Caches context with ephemeral cache_control
    - Uses generate_isolated + extract_output
    """

    def __init__(self, chat_client, use_rag=False, rag_top_k=20, rag_use_bm25=True,
                 use_self_consistency=False, sc_samples=3, sc_temperature=0.7, num_threads=None, use_batch=False, batch_size: int = 8, experiment_path: str = None):
        super().__init__(
            chat_client,
            use_rag=use_rag,
            rag_top_k=rag_top_k,
            rag_use_bm25=rag_use_bm25,
            use_self_consistency=use_self_consistency,
            sc_samples=sc_samples,
            sc_temperature=sc_temperature,
            num_threads=num_threads,
            batch_size=batch_size,
            experiment_path=experiment_path
        )
        self.use_batch = use_batch
        self.retriever = None

    def set_retriever(self, retriever):
        self.retriever = retriever

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries."""
        if retry_idx == 0:
            return ""
        return (
            "\n\nCRITICAL: You MUST output your final answer in EXACTLY this format:\n"
            "<analysis>[Your reasoning for each option]</analysis>\n"
            "<answer>[One or more letters: A, B, C, or D separated by commas]</answer>\n"
            "Do NOT include 'Option' before letters. Just the letter(s)."
        )

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> List[Dict]:
        """
        Format prompt with cache_control (like your Colab optimized).

        Returns: messages list with cached context
        """
        # Build cacheable context (IDENTICAL for all questions in topic)
        context_text = "\n\n".join([
            f"<document_{i+1}>: {doc['content']}</document_{i+1}>"
            for i, doc in enumerate(context_docs)
        ])

        # Context message with ephemeral caching (YOUR EXACT LOGIC)
        context_message = {
            "type": "text",
            "text": f"""<context_documents>
<topic_id>{question.get('topic_id', '')}</topic_id>
{context_text}
</context_documents>""",
            "cache_control": {"type": "ephemeral"}  # ← Prompt caching!
        }

        # Question-specific content (unique per question)
        question_text = f"""<target_event>{question["target_event"]}</target_event>

<options>
<option_a>{question["option_A"]}</option_a>
<option_b>{question["option_B"]}</option_b>
<option_c>{question["option_C"]}</option_c>
<option_d>{question["option_D"]}</option_d>
</options>
"""

        # Messages array (cached context + question)
        messages = [{
            "role": "user",
            "content": [context_message, {"type": "text", "text": question_text}]
        }]

        return messages
    
    def call_model(self, prompt: List[Dict]) -> str:
        """
        Call OLD Claude with generate_isolated (InvokeModel, returns raw text).
        """
        return self.chat.generate_isolated(prompt)
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse with extract_output (YOUR EXACT PARSING).
        """
        if not isinstance(response, str):
            response = str(response)
        
        # Your original extract_output
        result = extract_output(response)
        
        return {
            "answer": result["answer"],
            "analysis": result["analysis"],
            "flag": result["flag"]
        }
    
    def supports_caching(self) -> bool:
        return True  # Enable topic-based grouping
    
    def get_progress_color(self) -> str:
        return "blue"


class ClaudeSimpleInference(BaseInference):
    """
    Simple Claude inference - no caching, processes each question independently.

    Use for: Quick tests, complete isolation between questions
    """

    def __init__(self, chat_client, use_rag=False, rag_top_k=20, rag_use_bm25=True,
                 use_self_consistency=False, sc_samples=3, sc_temperature=0.7, num_threads=None, use_batch=False, batch_size: int = 8, experiment_path: str = None):
        super().__init__(
            chat_client,
            use_rag=use_rag,
            rag_top_k=rag_top_k,
            rag_use_bm25=rag_use_bm25,
            use_self_consistency=use_self_consistency,
            sc_samples=sc_samples,
            sc_temperature=sc_temperature,
            num_threads=num_threads,
            batch_size=batch_size,
            experiment_path=experiment_path
        )
        self.use_batch = use_batch
        self.retriever = None

    def set_retriever(self, retriever):
        self.retriever = retriever

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries."""
        if retry_idx == 0:
            return ""
        return (
            "\n\nCRITICAL: You MUST output your final answer in EXACTLY this format:\n"
            "<analysis>[Your reasoning for each option]</analysis>\n"
            "<answer>[One or more letters: A, B, C, or D separated by commas]</answer>\n"
            "Do NOT include 'Option' before letters. Just the letter(s)."
        )

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> List[Dict]:
        """
        Format prompt with XML tags for Claude.
        """
        # Format context documents
        context_text = "\n\n".join([
            f"<document_{j+1}>{doc['content']}</document_{j+1}>"
            for j, doc in enumerate(context_docs)
        ])

        # Format options
        options_text = "\n".join([
            f"<option_{opt}>{question[f'option_{opt}']}</option_{opt}>"
            for opt in ["A", "B", "C", "D"]
        ])

        # Build user prompt
        user_content = f"""<context>
{context_text}
</context>

<target_event>
{question["target_event"]}
</target_event>

<options>
{options_text}
</options>

Analyze which option(s) are plausible causes of the target event based on the context.
Provide your analysis in <analysis> tags and your answer in <answer> tags.
Answer format: A single letter (A, B, C, or D) or multiple letters separated by commas (e.g., A,B).
"""

        return [{"role": "user", "content": [{"text": user_content}] }]

    def call_model(self, prompt: List[Dict]) -> str:
        """Call Claude model."""
        return self.chat.generate_structured(prompt)

    def parse_response(self, response: Dict) -> Dict:
        return {
            "answer": response["answer"],
            "analysis": response["analysis"],
            "flag": False  # Never fails!
        }
    
    def supports_caching(self) -> bool:
        return False

    def get_progress_color(self) -> str:
        return "magenta"


class ClaudeOptimizedInference(ClaudeSimpleInference):
    """
    Optimized Claude with prompt caching - groups questions by topic.
    """

    def supports_caching(self) -> bool:
        return True  # Enable caching optimization

    def get_progress_color(self) -> str:
        return "blue"


class ClaudeThinkingInference(ClaudeSimpleInference):
    """
    Claude with extended thinking mode.
    
    NOTE: AWS Bedrock does not support thinking mode with forced tool use,
    so this uses text-based response with extract_output parsing instead
    of structured output.
    
    Returns: predictions, analyses, AND thinkings
    """

    def call_model(self, prompt: List[Dict]) -> Dict:
        """
        Call Claude Thinking WITHOUT structured output.
        
        Uses generate_isolated() which returns {'thinking': '...', 'answer': 'text'}
        instead of generate_structured() with tool use.
        """
        # Call the OLD generate_isolated method (returns thinking + text)
        return self.chat.generate_isolated(prompt)

    def parse_response(self, response: Dict) -> Dict:
        """
        Parse thinking response with text extraction.
        
        Response structure from generate_isolated:
        {
            'thinking': '...',  # Reasoning process
            'answer': '...'     # Text with <analysis> and <answer> tags
        }
        """
        if not isinstance(response, dict):
            return {
                "answer": ["Fail"],
                "analysis": "Parse error: invalid response type",
                "thinking": "",
                "flag": True
            }
        
        # Extract thinking
        thinking = response.get("thinking", "")
        
        # Extract text answer
        answer_text = response.get("answer", "")
        
        # Parse the text with extract_output (from base.py)
        parsed = extract_output(answer_text)
        
        # Normalize answer to array
        answer_value = parsed.get("answer", ["Fail"])
        if isinstance(answer_value, str):
            answer_value = [answer_value]
        elif not isinstance(answer_value, list):
            answer_value = ["Fail"]
        
        return {
            "answer": answer_value,
            "analysis": parsed.get("analysis", ""),
            "thinking": thinking,
            "flag": parsed.get("flag", False)
        }

    def supports_caching(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        return True

    def get_progress_color(self) -> str:
        return "cyan"


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_claude_inference(
    chat_client,
    questions: List[Dict],
    docs: List[Dict],
    mode: str = 'optimized',
    sleep_seconds: int = 0,
    use_rag: bool = False,
    rag_top_k: int = 20,
    rag_use_bm25: bool = True,
    retriever=None,
    use_self_consistency: bool = False,
    sc_samples: int = 3,
    sc_temperature: float = 0.7,
    num_threads: int = None,
    use_batch: bool = False,
    batch_size: int = 8,
    experiment_path: str = None,
    graph_rag_data: Dict = None,
    context_cache = None
) -> Dict:
    """
    Run Claude inference with optional RAG and parallel processing.

    Args:
        chat_client: Claude client instance
        questions: List of question dictionaries
        docs: List of document dictionaries
        mode: 'simple', 'optimized', or 'thinking'
        sleep_seconds: Rate limiting delay between requests
        use_rag: Enable RAG retrieval
        rag_top_k: Top-k chunks for RAG
        rag_use_bm25: Use BM25 + embeddings for retrieval
        retriever: Retriever instance (optional)
        use_self_consistency: Enable self-consistency voting
        sc_samples: Number of samples for self-consistency
        sc_temperature: Temperature for SC sampling
        num_threads: Number of parallel threads (default: CPU count)
        use_batch: Use official batched inference
        batch_size: Number of questions to process in parallel
        graph_rag_data: Dict with Graph RAG components (graphs, query_embeddings, embedder, etc.)
        context_cache: Preprocessed context cache (takes priority over graph_rag)

    Returns:
        Results dictionary with predictions, analyses, and optionally thinkings
    """
    
    if use_batch and mode == 'simple':
        print("   INFO: --use-batch is specified, forcing mode to 'optimized' to enable prompt caching.")
        mode = 'optimized'

    inference_classes = {
        'simple': ClaudeOldSimpleInference,
        'optimized': ClaudeOldOptimizedInference,
        'thinking': ClaudeThinkingInference
    }
    
    if mode not in inference_classes:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(inference_classes.keys())}")
    
    inference_class = inference_classes[mode]
    
    # Create inference instance with all parameters including num_threads and experiment_path
    inference = inference_class(
        chat_client,
        use_rag=use_rag,
        rag_top_k=rag_top_k,
        rag_use_bm25=rag_use_bm25,
        use_self_consistency=use_self_consistency,
        sc_samples=sc_samples,
        sc_temperature=sc_temperature,
        num_threads=num_threads,
        use_batch=use_batch,
        batch_size=batch_size,
        experiment_path=experiment_path
    )

    # Set retriever if RAG is enabled
    if use_rag:
        if retriever is None:
            raise ValueError("Retriever must be provided when use_rag=True")
        inference.set_retriever(retriever)

    # Set Graph RAG data if provided
    if graph_rag_data is not None:
        inference.set_graph_rag(graph_rag_data)

    # Set preprocessed context cache if provided (takes priority over graph_rag)
    if context_cache is not None:
        inference.set_context_cache(context_cache)

    # Run inference with base class logic (handles caching + threading)
    return inference.run(questions, docs, sleep_seconds)
