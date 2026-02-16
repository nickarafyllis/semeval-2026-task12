"""
Base inference class with common logic for all models

This base class implements the template method pattern, extracting all common logic
from the 6 inference functions into a single, reusable implementation.

Each model-specific implementation only needs to override 2-3 methods.
"""

import re
import time
import random
import gc
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from threading import Lock


# ============================================================================
# ADAPTIVE RATE LIMITER
# ============================================================================

class AdaptiveRateLimiter:
    """
    Per-thread adaptive rate limiter that adjusts based on throttling feedback.

    - Starts with a reasonable delay between requests
    - Increases delay when throttling is detected
    - Gradually decreases delay on successful requests
    - Uses token bucket algorithm for smooth rate limiting
    - NOT thread-safe by design - each thread gets its own instance
    """

    def __init__(self, initial_delay: float = 1.0, min_delay: float = 0.5, max_delay: float = 30.0,
                 rpm: int = None):
        self._delay = initial_delay
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._last_request_time = 0.0
        self._consecutive_successes = 0
        self._throttle_count = 0
        
        # Calculate delay based on RPM if provided
        if rpm is not None:
            self._delay = 60.0 / rpm
            self._initial_delay = self._delay
            self._min_delay = min(self._delay * 0.5, min_delay)
        else:
            self._initial_delay = initial_delay

    def wait(self, question_id: str = None):
        """Wait appropriate time before making next request."""
        now = time.time()
        elapsed = now - self._last_request_time
        wait_time = max(0, self._delay - elapsed)

        if wait_time > 0:
            time.sleep(wait_time)

        self._last_request_time = time.time()

    def record_success(self):
        """Record successful request - gradually decrease delay."""
        self._consecutive_successes += 1
        # Decrease delay after 5 consecutive successes
        if self._consecutive_successes >= 5:
            self._delay = max(self._min_delay, self._delay * 0.8)
            self._consecutive_successes = 0

    def record_throttle(self, error_msg: str = None, question_id: str = None):
        """Record throttling - increase delay significantly."""
        self._throttle_count += 1
        self._consecutive_successes = 0
        
        # Determine reason
        if error_msg and "throttl" in error_msg.lower():
            reason = "Rate limit exceeded (throttling)"
        elif error_msg:
            reason = f"API error: {error_msg[:80]}"
        else:
            reason = "Rate limit exceeded"
        
        # Exponential backoff on throttle
        self._delay = min(self._max_delay, self._delay * 2.0 + random.uniform(1, 3))
        
        qid_str = f" ({question_id[:8]})" if question_id else ""
        tqdm.write(f"   🔄 Throttle #{self._throttle_count}{qid_str}: {reason}")
        tqdm.write(f"      ⏱️  Waiting {self._delay:.1f}s before retry (current delay)")

    def get_stats(self) -> dict:
        """Get current rate limiter stats."""
        return {
            'current_delay': self._delay,
            'throttle_count': self._throttle_count
        }


def is_throttling_error(exception: Exception) -> bool:
    """Check if an exception is a throttling/rate limit error."""
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()

    throttle_indicators = [
        'throttl',
        'rate limit',
        'ratelimit',
        'too many requests',
        '429',
        'serviceunvailable',
        'serviceunavailable',
        'capacity',
        'overloaded',
    ]

    return any(indicator in error_str or indicator in error_type
               for indicator in throttle_indicators)

from src.utils.cost_tracker import CostTracker

# ============================================================================
# RESPONSE PARSING
# ============================================================================

def extract_output(response):
    """
    Parse <analysis> and <answer> tags from a structured response.
    Validates that answer letters are only A, B, C, D.
    Falls back to natural language parsing if XML tags are missing.

    Enhanced with:
    - More robust fallback patterns
    - Better handling of malformed XML
    - Extraction of partial answers from broken responses
    """
    flag = False
    try:
        # Extract analysis
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
            if not analysis:
                print(response)
                analysis = ""
                flag=True
        else:
            analysis = ""
            flag=True
            print(f"⚠️ Parse failed. Raw response:\n{response[:500]}...")

        # Extract answer from XML tags first
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        answer_text = answer_match.group(1).strip() if answer_match else ""

        # If no XML answer found, try COMPREHENSIVE fallback patterns
        if not answer_text:
            # Expanded fallback patterns in priority order
            fallback_patterns = [
                # Standard patterns
                r'<answer>\s*([A-D,\s]+)',  # Unclosed <answer> tag
                r'Answer:\s*\*{0,2}Option\s+([A-D])\*{0,2}',  # Answer: **Option C**
                r'Answer:\s*\*{0,2}([A-D](?:\s*,\s*[A-D])*)\*{0,2}',  # Answer: C or Answer: C, D
                r'\*{2}Option\s+([A-D])\*{2}',  # **Option C**
                r'(?:the answer is|correct answer is|i choose|i select)\s*\*{0,2}(?:Option\s+)?([A-D](?:\s*,\s*[A-D])*)\*{0,2}',  # the answer is C
                r'(?:^|\n)\s*Option\s+([A-D])\s*(?:\n|$)',  # "Option C" on its own line
                r'(?:^|\n)\s*([A-D])\s*(?:\n|$)',  # Just "C" on a line
                # Very aggressive last resort patterns
                r'\b([A-D])\s+is\s+(?:the\s+)?correct',  # "C is correct"
                r'\bcorrect\s+answer\s*:\s*([A-D])',  # "correct answer: C"
                r'select\s+([A-D])\b',  # "select C"
            ]

            for pattern in fallback_patterns:
                fallback_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if fallback_match:
                    answer_text = fallback_match.group(1).upper()
                    # Mark as flag since we had to use fallback parsing
                    flag = True
                    print(f"   ℹ️  Used fallback pattern to extract answer: {answer_text}")
                    break

        # Parse answer letters (handle multiple answers like "B,D" or "B, D")
        if answer_text:
            # Clean and split the answer - remove "Option" prefix if present
            answer_text = re.sub(r'\bOption\s+', '', answer_text, flags=re.IGNORECASE)
            answer_letters = [letter.strip().upper() for letter in re.split(r'[,\s]+', answer_text) if letter.strip()]

            # Validate that all letters are A, B, C, or D
            valid_letters = {'A', 'B', 'C', 'D'}
            answer_letters = [l for l in answer_letters if l in valid_letters]

            # Remove duplicates while preserving order
            answer_letters = list(dict.fromkeys(answer_letters))

            if not answer_letters:
                answer_letters = ["Fail"]
                flag = True
                print(f"   ⚠️  No valid letters found in answer_text: '{answer_text}'")
        else:
            answer_letters = ["Fail"]  # Default fallback
            flag=True
            print(f"   ⚠️  No answer found in response")

        return {
            "analysis": analysis,
            "answer": answer_letters,
            "raw_response": response,
            "flag": flag
        }

    except Exception as e:
        flag=True
        print(f"   ❌ Exception in extract_output: {e}")
        return {
            "analysis": "",
            "answer": ["Fail"],  # Default fallback
            "raw_response": response,
            "flag": flag
        }
        
# ============================================================================
# RAG FILTERING (Dynamic)
# ============================================================================

def filter_relevant_docs(question: Dict, all_docs: List[Dict], 
                         min_score: float = 1.0, 
                         max_docs: int = 10,
                         method: str = "keyword") -> List[Dict]:
    """
    Filter documents dynamically based on relevance score.
    Returns all docs above min_score threshold (up to max_docs).
    
    Args:
        question: Question dictionary with 'target_event'
        all_docs: All available documents
        min_score: Minimum relevance score (default: 1.0)
        max_docs: Maximum documents to return (default: 10)
        method: 'keyword' for simple matching
    
    Returns:
        All documents with score >= min_score (up to max_docs)
    """
    if not all_docs:
        return []
    
    target = question.get('target_event', '').lower()
    
    if method == "keyword":
        # Score each document
        doc_scores = []
        for doc in all_docs:
            content = doc.get('content', '').lower()
            
            # Score from target event keywords (meaningful words only)
            target_words = [w for w in target.split() if len(w) > 3]
            score = sum(1 for word in target_words if word in content)
            
            # Bonus: score from options
            for opt in ['A', 'B', 'C', 'D']:
                opt_key = f'option_{opt}'
                if opt_key in question:
                    opt_text = question[opt_key].lower()
                    opt_words = [w for w in opt_text.split() if len(w) > 3]
                    score += sum(0.5 for word in opt_words if word in content)
            
            doc_scores.append((doc, score))
        
        # Filter by threshold
        relevant = [(doc, score) for doc, score in doc_scores if score >= min_score]
        
        # Sort by score (descending) and take top max_docs
        relevant.sort(key=lambda x: x[1], reverse=True)
        filtered_docs = [doc for doc, score in relevant[:max_docs]]
        
        return filtered_docs
    
    else:
        return all_docs[:max_docs]

# ============================================================================
# BASE INFERENCE CLASS
# ============================================================================

class BaseInference(ABC):
    """
    Abstract base class for all model inference implementations.

    This class contains ALL common logic:
    - Topic-based document preparation
    - Question grouping for caching
    - Parallel processing per topic (cache-aware)
    - Progress tracking
    - Rate limiting
    - Result aggregation
    - Retry handling with exponential backoff

    Subclasses only need to implement:
    - format_prompt() - How to format the prompt
    - call_model() - How to call the model
    - parse_response() - How to parse the response
    """

    def __init__(self, chat_client, system_prompt: Optional[str] = None,
                use_rag: bool = False,
                rag_top_k: int = 20,
                rag_use_bm25: bool = True,
                use_self_consistency: bool = False,
                sc_samples: int = 3,
                sc_temperature: float = 1.0,
                num_threads: int = None,
                batch_size: int = 8,
                experiment_path: Optional[str] = None,
                rpm: int = None):
        self.chat = chat_client
        self.system_prompt = system_prompt
        self.max_retries = getattr(chat_client, "max_retries", 7)
        self.base_delay = getattr(chat_client, "base_delay", 1.5)

        # RAG config
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.rag_use_bm25 = rag_use_bm25
        self.retriever = None

        # Graph RAG config
        self.graph_rag_data = None

        # Preprocessed context cache (takes priority over graph_rag_data)
        self.context_cache = None

        # Self-consistency config
        self.use_self_consistency = use_self_consistency
        self.sc_samples = sc_samples
        self.sc_temperature = sc_temperature
        self.original_temperature = getattr(chat_client, 'temperature', 0.0)

        # Threading config
        self.num_threads = num_threads if num_threads is not None else cpu_count()
        self.batch_size = batch_size

        # RPM config (for rate limiting)
        self.rpm = rpm

        # Cost tracking
        self.cost_tracker = None  # Will be initialized in run() with model name

        # Incremental saving
        self.experiment_path = experiment_path
        
    def build_enforce_schema_retry_suffix(self, _retry_idx: int) -> str:
        """
        Optional: subclasses can override to enforce stricter schema in retries.
        Default: empty (no extra enforcement).
        """
        return ""

    def set_graph_rag(self, graph_rag_data: dict):
        """Set Graph RAG data for document-level graph retrieval."""
        self.graph_rag_data = graph_rag_data

    def set_context_cache(self, context_cache):
        """Set preprocessed context cache (takes priority over graph_rag_data)."""
        self.context_cache = context_cache

    def _get_context_from_cache(self, question: Dict) -> str:
        """Get context from preprocessed cache if available."""
        if not self.context_cache:
            return None
        topic_id = question.get('topic_id')
        return self.context_cache.get_context(topic_id, question)

    def _retrieve_with_graph_rag(self, question: Dict) -> List[Dict]:
        """Retrieve documents using Graph RAG traversal via centralized utility."""
        from src.retrieval.graph_rag_utils import retrieve_with_graph_rag

        if not self.graph_rag_data:
            return []

        return retrieve_with_graph_rag(question, self.graph_rag_data)

    def _extract_model_name(self, model_id: str) -> str:
        """Extract short model name from full AWS model ID."""
        # Examples:
        # "us.anthropic.claude-opus-4-5-20251101-v1:0" -> "claude-opus-4.5"
        # "us.anthropic.claude-sonnet-4-5-20250929-v1:0" -> "claude-sonnet-4.5"
        # "moonshot.kimi-k2-thinking" -> "kimi-k2-thinking"

        if "opus-4-5" in model_id:
            return "claude-opus-4.5"
        elif "opus-4-1" in model_id:
            return "claude-opus-4.1"
        elif "opus-4" in model_id:
            return "claude-opus-4.0"
        elif "sonnet-4-5" in model_id:
            return "claude-sonnet-4.5"
        elif "sonnet-4" in model_id:
            return "claude-sonnet-4.0"
        elif "haiku-4-5" in model_id:
            return "claude-haiku-4.5"
        elif "haiku" in model_id and "3-5" in model_id:
            return "claude-haiku-3.5"
        elif "llama" in model_id:
            return "llama-3.3-70b"
        elif "deepseek" in model_id and "r1" in model_id:
            return "deepseek-r1"
        elif "deepseek" in model_id:
            return "deepseek-v3.1"
        elif "kimi" in model_id or "moonshot" in model_id:
            return "kimi-k2-thinking"
        elif "gemini" in model_id:
            return "gemini-3-flash-preview"
        elif "gpt-5.2" in model_id:
            return "gpt-5.2"
        elif "gpt-5" in model_id:
            return "gpt-5"
        elif "gpt-4.1" in model_id:
            return "gpt-4.1"
        elif "gpt-4o-mini" in model_id:
            return "gpt-4o-mini"
        elif "gpt-4o" in model_id:
            return "gpt-4o"
        elif "o3-mini" in model_id:
            return "o3-mini"
        elif "o3" in model_id:
            return "o3"
        elif "o1-mini" in model_id:
            return "o1-mini"
        elif "o1" in model_id:
            return "o1"
        else:
            return "claude-sonnet-4.5"  # Default fallback

    def _track_usage(self, question_id: str = None):
        """Track usage from last API call if available."""
        if hasattr(self, 'chat') and hasattr(self.chat, 'last_usage') and self.chat.last_usage:
            usage = self.chat.last_usage

            # Handle multiple API formats:
            # - InvokeModel API (old): input_tokens, cache_creation_input_tokens, cache_read_input_tokens
            # - Converse API: inputTokens, cacheWriteInputTokens, cacheReadInputTokens
            input_tokens = usage.get('inputTokens') if 'inputTokens' in usage else usage.get('input_tokens', 0)
            output_tokens = usage.get('outputTokens') if 'outputTokens' in usage else usage.get('output_tokens', 0)

            # Check each format in order
            if 'cacheCreationInputTokens' in usage:
                cache_write = usage['cacheCreationInputTokens']
            elif 'cacheWriteInputTokens' in usage:
                cache_write = usage['cacheWriteInputTokens']
            else:
                cache_write = usage.get('cache_creation_input_tokens', 0)

            if 'cacheReadInputTokens' in usage:
                cache_read = usage['cacheReadInputTokens']
            else:
                cache_read = usage.get('cache_read_input_tokens', 0)

            self.cost_tracker.add_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_write,
                cache_read_tokens=cache_read,
                metadata={'question_id': question_id} if question_id else None
            )
            # Reset usage
            self.chat.last_usage = None

    # ========================================================================
    # MAIN INFERENCE FLOW (Template Method)
    # ========================================================================

    def run(self, questions: List[Dict], docs: List[Dict], sleep_seconds: int = 0) -> Dict[str, Any]:
        """
        Main inference entry point. Uses template method pattern.

        Args:
            questions: List of question dictionaries
            docs: List of document dictionaries
            sleep_seconds: Seconds to sleep between requests (rate limiting)

        Returns:
            Dictionary with 'predictions', 'analyses', and optionally 'thinkings'
        """
        # Initialize cost tracker
        # Handle both Claude classes (self.chat.model_id) and Gemini classes (self.model_id)
        if hasattr(self, 'model_id'):
            model_id = self.model_id
        elif hasattr(self, 'chat'):
            model_id = getattr(self.chat, 'model_id', 'unknown')
        else:
            model_id = 'unknown'
        # Extract short model name from full ID (e.g., "claude-opus-4.5" from full path)
        model_name = self._extract_model_name(model_id)
        self.cost_tracker = CostTracker(model_name)

        # Prepare data structures
        topic2docs = self._prepare_topic_docs(docs)

        # Initialize results
        results = self._initialize_results()

        # Choose execution path based on model capabilities
        if self.supports_caching():
            return self._run_with_caching(questions, topic2docs, sleep_seconds, results)
        else:
            return self._run_simple(questions, topic2docs, sleep_seconds, results)

    # ========================================================================
    # EXECUTION STRATEGIES
    # ========================================================================   
    
    def _run_simple(self, questions, topic2docs, sleep_seconds, results):
        """Simple sequential processing (no caching, no parallelization)."""
        # Create rate limiter for this execution
        rate_limiter = AdaptiveRateLimiter(
            initial_delay=0.0,
            min_delay=0.0,
            max_delay=30.0,
            rpm=self.rpm
        )
        
        with self._create_progress_bar(len(questions)) as pbar:
            for q in questions:
                try:
                    if hasattr(self, 'chat') and hasattr(self.chat, 'reset'):
                        self.chat.reset()

                    ctx_docs = topic2docs.get(q["topic_id"], [])

                    # Priority: preprocessed context cache > graph_rag > standard RAG > static
                    if self.context_cache:
                        # Use preprocessed context (already formatted string)
                        context_str = self._get_context_from_cache(q)
                        if context_str:
                            ctx_docs = [{'content': context_str}]  # Wrap as single doc
                    elif self.graph_rag_data:
                        ctx_docs = self._retrieve_with_graph_rag(q)
                    elif self.use_rag and self.retriever:
                        ctx_docs = self.retriever.retrieve(q['target_event'], top_k=self.rag_top_k)

                    parsed = self._process_single_question(q, ctx_docs, rate_limiter)
                    self._add_result(results, q["id"], parsed)

                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

                except Exception as e:
                    self._handle_error(results, q["id"], e)

                pbar.update(1)

        # Add cost tracking to results
        results['cost_tracker'] = self.cost_tracker

        return results

    def _run_with_caching(self, questions: List[Dict], topic2docs: Dict,
                          sleep_seconds: int, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimized execution with intelligent thread utilization.
        
        Strategy per topic:
        1. First question: Cache write (sequential)
        2. Remaining questions: Cache reads (parallel with num_threads)
        
        Note: With RAG enabled, each question gets different docs (less cache benefit).
        """
        # Group questions by topic
        topic_questions = self._group_by_topic(questions)
        
        print(f"\n📊 Processing {len(topic_questions)} topics with {self.num_threads} threads per topic\n")

        # Create a single progress bar for all questions
        with self._create_progress_bar(len(questions)) as pbar:
            for topic_idx, (topic_id, topic_qs) in enumerate(topic_questions.items(), 1):
                pbar.write(f"\n🔖 Topic {topic_idx}/{len(topic_questions)}: {topic_id} ({len(topic_qs)} questions)")

                # Reset chat for new topic
                if hasattr(self, 'chat') and hasattr(self.chat, 'reset'):
                    self.chat.reset()

                # Get context docs for this topic
                ctx_docs = topic2docs.get(topic_id, [])

                # Process this topic with parallel execution
                topic_results = self._process_topic_parallel(
                    topic_qs,
                    ctx_docs,
                    sleep_seconds,
                    pbar
                )

                # Merge results
                for id, parsed in topic_results.items():
                    self._add_result(results, id, parsed)

                pbar.write(f"   ✓ Topic {topic_idx} completed: {len(topic_results)}/{len(topic_qs)} questions")

                # Clear topic results to free memory
                del topic_results
                gc.collect()

        # Add cost tracking to results
        results['cost_tracker'] = self.cost_tracker

        return results

    def _process_topic_parallel(self, topic_questions: List[Dict], ctx_docs: List[Dict],
                                 sleep_seconds: int, pbar: tqdm) -> Dict[str, Dict[str, Any]]:
        """
        Process all questions in a topic with parallel execution after cache warm-up.
        
        Steps:
        1. First question: Sequential (warms cache)
        2. Remaining questions: Parallel (hit cache) with per-thread rate limiters
        
        Args:
            topic_questions: Questions for this topic
            ctx_docs: Context documents for this topic
            sleep_seconds: Rate limiting delay
            pbar: Progress bar to update
        
        Returns:
            Dictionary mapping question_id -> parsed result
        """
        topic_results = {}
        
        if not topic_questions:
            return topic_results
        
        # ========================================================================
        # STEP 1: Cache Warm-up (First Question)
        # ========================================================================

        first_q = topic_questions[0]
        
        # Create rate limiter for first question
        rate_limiter = AdaptiveRateLimiter(
            initial_delay=0.0,
            min_delay=0.0,
            max_delay=30.0,
            rpm=self.rpm
        )

        try:
            # Priority: preprocessed context cache > graph_rag > standard RAG > static
            if self.context_cache:
                context_str = self._get_context_from_cache(first_q)
                filtered_docs = [{'content': context_str}] if context_str else ctx_docs
            elif self.graph_rag_data:
                filtered_docs = self._retrieve_with_graph_rag(first_q)
            elif self.use_rag and self.retriever:
                filtered_docs = self.retriever.retrieve(first_q['target_event'], top_k=self.rag_top_k)
            else:
                filtered_docs = ctx_docs

            # Process first question (sequential)
            parsed = self._process_single_question(first_q, filtered_docs, rate_limiter)
            topic_results[first_q["id"]] = parsed

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        except Exception as e:
            pbar.write(f"   ✗ First question {first_q['id'][:8]} failed: {e}")
            topic_results[first_q["id"]] = {
                "answer": ["Fail"],
                "analysis": f"Error: {str(e)}",
                "flag": True
            }

        pbar.update(1)

        # ========================================================================
        # STEP 2: Parallel Processing (Remaining Questions)
        # ========================================================================

        remaining = topic_questions[1:]
        if not remaining:
            return topic_results

        # Process remaining questions in parallel (each thread gets its own rate limiter)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_id = {}
            for q in remaining:
                # Prepare docs for this question
                # Priority: preprocessed context cache > graph_rag > standard RAG > static
                if self.context_cache:
                    context_str = self._get_context_from_cache(q)
                    filtered_docs = [{'content': context_str}] if context_str else ctx_docs
                elif self.graph_rag_data:
                    filtered_docs = self._retrieve_with_graph_rag(q)
                elif self.use_rag and self.retriever:
                    filtered_docs = self.retriever.retrieve(q['target_event'], top_k=self.rag_top_k)
                else:
                    filtered_docs = ctx_docs
                
                # Submit task with per-thread rate limiter
                future = executor.submit(
                    self._process_single_question_with_sleep,
                    q,
                    filtered_docs,
                    sleep_seconds
                )
                future_to_id[future] = q["id"]
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                id = future_to_id[future]
                try:
                    parsed = future.result()
                    topic_results[id] = parsed
                except Exception as e:
                    pbar.write(f"   ✗ Question {id[:8]} failed: {e}")
                    topic_results[id] = {
                        "answer": ["Fail"],
                        "analysis": f"Error: {str(e)}",
                        "flag": True
                    }

                pbar.update(1)
        
        return topic_results

    def _process_single_question_with_sleep(self, q: Dict, ctx_docs: List[Dict],
                                             sleep_seconds: int) -> Dict[str, Any]:
        """
        Wrapper for _process_single_question with rate limiting.
        Used by parallel executor - creates its own rate limiter.
        """
        # Create rate limiter for this thread
        rate_limiter = AdaptiveRateLimiter(
            initial_delay=0.0,
            min_delay=0.0,
            max_delay=30.0,
            rpm=self.rpm
        )
        
        result = self._process_single_question(q, ctx_docs, rate_limiter)
        
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        
        return result

    def _process_single_question(self, q: Dict, ctx_docs: List[Dict], 
                                 rate_limiter: AdaptiveRateLimiter = None) -> Dict[str, Any]:
        """
        Process a single question with optional self-consistency.
        
        If SC enabled: samples N times and aggregates via majority vote.
        Otherwise: standard single-sample processing.
        
        Args:
            q: Question dictionary
            ctx_docs: Context documents
            rate_limiter: Per-thread rate limiter (creates own if None)
        """
        if self.use_self_consistency:
            return self._process_with_self_consistency(q, ctx_docs, rate_limiter)
        else:
            return self._process_single_sample(q, ctx_docs, rate_limiter)

    def _process_single_sample(self, q: Dict, ctx_docs: List[Dict], 
                             rate_limiter: AdaptiveRateLimiter = None) -> Dict[str, Any]:
        """
        Process a single question with ONE sample (standard mode).

        Note: 
        - AWS models (Claude, Llama, DeepSeek, Kimi): boto3 handles API retries 
          automatically with max_attempts=10, mode='adaptive'
        - Gemini: No auto-retry, uses rate limiter to prevent throttling
        
        This method only retries on parsing failures.
        
        Args:
            q: Question dictionary
            ctx_docs: Context documents
            rate_limiter: Per-thread rate limiter (creates own if None)
        """
        prompt = self.format_prompt(q, ctx_docs)
        qid = q.get('id', 'unknown')[:8]

        # Create rate limiter if not provided (per-thread)
        if rate_limiter is None:
            rate_limiter = AdaptiveRateLimiter(
                initial_delay=0.0,
                min_delay=0.0,
                max_delay=30.0,
                rpm=self.rpm
            )

        # Retry loop for parsing failures only
        retry_idx = 0
        parsed = None
        response_data = None
        last_error = None

        while retry_idx <= self.max_retries:
            enforced = self.build_enforce_schema_retry_suffix(retry_idx)
            prompt_plus = self._append_enforcement(prompt, enforced)

            # Log retry attempt explicitly
            if retry_idx > 0:
                tqdm.write(f"   🔄 RETRY #{retry_idx}/{self.max_retries} for {qid} (parsing failed)")

            try:
                # Rate limiting before request (per-thread, so doesn't affect other threads)
                rate_limiter.wait(qid)
                response_data = self.call_model(prompt_plus)
                rate_limiter.record_success()
                self._track_usage(question_id=q.get('id'))
                parsed = self.parse_response(response_data)
                last_error = None

                # Log successful parse after retry
                if retry_idx > 0 and parsed and not parsed.get("flag", True):
                    tqdm.write(f"   ✓ Retry #{retry_idx} succeeded for {qid}")

            except Exception as e:
                last_error = e
                parsed = {"answer": ["Fail"], "analysis": f"ERROR::{e}", "flag": True}

                # Check if this is a throttling error
                if is_throttling_error(e):
                    error_name = type(e).__name__
                    error_msg = str(e)[:120]
                    rate_limiter.record_throttle(error_msg, qid)
                    # For AWS: boto3 is retrying automatically
                    # For Gemini: rate limiter will handle the delay
                else:
                    error_name = type(e).__name__
                    error_msg = str(e)[:100]
                    tqdm.write(f"   ⚠️  API error for {qid}: {error_name}: {error_msg}")
                break  # Don't retry API errors - boto3 already retried for AWS

            # Success - no flag means clean parse
            if parsed and not parsed.get("flag", True):
                break

            # Log what was wrong with the response
            if parsed and parsed.get("flag", True):
                answer = parsed.get("answer", [])
                if answer == ["Fail"]:
                    tqdm.write(f"      ❌ Failed to extract answer from response (attempt {retry_idx + 1})")
                else:
                    tqdm.write(f"      ⚠️  Parsed with fallback pattern (attempt {retry_idx + 1})")

            # Parsing failure - retry with schema enforcement
            retry_idx += 1
            if retry_idx <= self.max_retries:
                wait_time = self.base_delay * retry_idx  # Linear backoff
                tqdm.write(f"      ⏱️  Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

        if not parsed or parsed.get("flag", True):
            error_msg = f"Retry failed (last error: {last_error})" if last_error else "Retry failed"
            if last_error:
                error_summary = f"{type(last_error).__name__}: {str(last_error)[:80]}"
            else:
                error_summary = "parsing failed"
            tqdm.write(f"⚠️  Retry failed for {qid} after {retry_idx} attempts: {error_summary}")
            return {"answer": ["Fail"], "analysis": error_msg, "flag": True}

        return parsed

    def _detect_none_option(self, question: Dict) -> Optional[str]:
        """
        Detect which option (if any) represents "None of the above".
        
        Args:
            question: Question dictionary with option_A, option_B, etc.
        
        Returns:
            'A', 'B', 'C', 'D', or None if no "None" option found
        """
        # Common phrases that indicate "None of the above"
        none_phrases = [
            'None of the others are correct causes.',
        ]
        
        # Check each option
        for opt_letter in ['A', 'B', 'C', 'D']:
            opt_key = f'option_{opt_letter}'
            if opt_key in question:
                opt_text = question[opt_key].lower().strip()
                
                # Check if option text contains any "none" phrase
                if any(phrase in opt_text for phrase in none_phrases):
                    return opt_letter
        
        return None

    def _process_with_self_consistency(self, q: Dict, ctx_docs: List[Dict], 
                                      rate_limiter: AdaptiveRateLimiter = None) -> Dict[str, Any]:
        """
        Process a single question with self-consistency (multiple samples + aggregation).
        
        Features:
        - Samples N times with higher temperature for diversity
        - Uses percentage-based voting (N-independent, threshold=35% --> so its actually 0.66 in 3 and 0.4 in 5)
        - Conflict resolution for "None of the above" options
        - Retry logic for each sample (via _process_single_sample)
        - Medium-detail analysis with sample predictions and votes
        
        Args:
            q: Question dictionary
            ctx_docs: Context documents
            rate_limiter: Per-thread rate limiter (creates own if None)
        
        Returns:
            Result dictionary with aggregated answer
        """
        from collections import Counter
        
        # Create rate limiter if not provided (per-thread)
        if rate_limiter is None:
            rate_limiter = AdaptiveRateLimiter(
                initial_delay=0.0,
                min_delay=0.0,
                max_delay=30.0,
                rpm=self.rpm
            )
        
        # ========================================================================
        # SAMPLING PHASE
        # ========================================================================
        
        # Set temperature for diverse sampling
        original_temp = None
        if hasattr(self, 'chat') and hasattr(self.chat, 'temperature'):
            original_temp = self.chat.temperature
            self.chat.temperature = self.sc_temperature
        
        # Collect N samples (each with retry logic via _process_single_sample)
        samples = []
        for _ in range(self.sc_samples):
            # Call _process_single_sample which includes retry logic
            result = self._process_single_sample(q, ctx_docs, rate_limiter)
            
            # Extract answer (normalize to list)
            answer = result.get('answer', ['Fail'])
            if not isinstance(answer, list):
                answer = [answer] if answer else ['Fail']
            
            samples.append(answer)
        
        # Restore original temperature
        if original_temp is not None and hasattr(self, 'chat'):
            self.chat.temperature = original_temp
        
        # ========================================================================
        # VOTE COUNTING
        # ========================================================================
        
        # Count how many samples contain each option
        option_sample_counts = Counter()
        for sample in samples:
            # Use set() to count each option once per sample
            for option in set(sample):
                option_sample_counts[option] += 1
        
        # Remove 'Fail' from voting
        valid_counts = {
            opt: count 
            for opt, count in option_sample_counts.items() 
            if opt != 'Fail'
        }
        
        # If all samples failed, return Fail
        if not valid_counts:
            return {
                'answer': ['Fail'],
                'analysis': f"SC-{self.sc_samples}: All samples failed after retries",
                'samples': samples,
                'flag': False
            }
        
        # Calculate percentages (N-independent!)
        option_percentages = {
            opt: count / self.sc_samples
            for opt, count in valid_counts.items()
        }
        raw_percentages_to_store = option_percentages.copy()
        
        # ========================================================================
        # CONFLICT RESOLUTION: Handle "None of the above"
        # ========================================================================
        
        THRESHOLD = 0.35  # Hardcoded 35% threshold
        
        none_option = self._detect_none_option(q)
        conflict_resolved = False
        
        if none_option and none_option in option_percentages:
            none_percentage = option_percentages[none_option]
            other_options = {
                opt: pct 
                for opt, pct in option_percentages.items() 
                if opt != none_option
            }
            
            if other_options:
                max_other_percentage = max(other_options.values())
                
                # CONFLICT DETECTION: Both "None" and others have significant support
                if none_percentage >= THRESHOLD and max_other_percentage >= THRESHOLD:
                    conflict_resolved = True
                    
                    if none_percentage > max_other_percentage:
                        # "None" wins → return ONLY None
                        sample_summary = ', '.join([str(s) for s in samples])
                        vote_summary = ', '.join([f"{opt}:{pct:.0%}" for opt, pct in sorted(option_percentages.items())])
                        
                        return {
                            'answer': [none_option],
                            'analysis': f"SC-{self.sc_samples}: [{sample_summary}] → Votes: {vote_summary} → CONFLICT: 'None' wins ({none_percentage:.0%} vs {max_other_percentage:.0%})",
                            'samples': samples,
                            'flag': False
                        }
                    else:
                        # Others win → exclude None from voting
                        option_percentages = other_options
        
        # ========================================================================
        # STANDARD AGGREGATION (after conflict resolution)
        # ========================================================================
        
        # Select options above 30% threshold
        confident_options = sorted([
            opt 
            for opt, percentage in option_percentages.items()
            if percentage >= THRESHOLD
        ])
        
        # Fallback: if no option meets threshold, return most common
        if not confident_options:
            confident_options = [max(option_percentages.items(), key=lambda x: x[1])[0]]
        
        # ========================================================================
        # BUILD MEDIUM-DETAIL ANALYSIS
        # ========================================================================
        
        # Sample summary: [['A'], ['A', 'B'], ['A'], ...]
        sample_summary = ', '.join([str(s) for s in samples])
        
        # Vote summary: A:80%, B:40%, C:20%
        vote_summary = ', '.join([f"{opt}:{pct:.0%}" for opt, pct in sorted(option_percentages.items())])
        
        # Build analysis string
        if conflict_resolved and none_option not in confident_options:
            # Conflict resolved in favor of others
            analysis = f"SC-{self.sc_samples}: [{sample_summary}] → Votes: {vote_summary} → CONFLICT: Others win (None excluded) → Selected: {confident_options}"
        else:
            # Normal aggregation
            analysis = f"SC-{self.sc_samples}: [{sample_summary}] → Votes: {vote_summary} → Selected: {confident_options}"
        
        return {
            'answer': confident_options,
            'analysis': analysis,
            'samples': samples,
            'flag': False,
            'raw_percentages': raw_percentages_to_store
        }

    def _append_enforcement(self, prompt: Any, enforce_suffix: str) -> Any:
        """
        Append enforcement to prompt only if enforce_suffix is non-empty.
        Works for both str prompts and messages lists.
        """
        if not enforce_suffix:
            return prompt

        if isinstance(prompt, str):
            return prompt + "\n" + enforce_suffix

        if isinstance(prompt, list) and len(prompt) > 0 and "content" in prompt[0]:
            content = prompt[0]["content"]
            if isinstance(content, list):
                content.append({"type": "text", "text": enforce_suffix})
            return prompt

        return prompt

    # ========================================================================
    # HELPER METHODS (Common logic)
    # ========================================================================

    def _prepare_topic_docs(self, docs: List[Dict]) -> Dict[Any, List[Dict]]:
        """Map topic_id to documents."""
        topic2docs: Dict[Any, List[Dict]] = {}
        for d in docs:
            topic_id = d.get("topic_id")
            if topic_id not in topic2docs:
                topic2docs[topic_id] = []
            topic2docs[topic_id].extend(d.get("docs", []))
        return topic2docs

    def _group_by_topic(self, questions: List[Dict]) -> Dict[Any, List[Dict]]:
        """Group questions by topic_id for caching."""
        topic_questions: Dict[Any, List[Dict]] = {}
        for q in questions:
            topic_id = q["topic_id"]
            topic_questions.setdefault(topic_id, []).append(q)
        return topic_questions

    def _initialize_results(self) -> Dict[str, Any]:
        """Initialize results dictionary."""
        results = {
            'predictions': {},
            'analyses': {}
        }
        if self.supports_thinking():
            results['thinkings'] = {}
        return results

    def _add_result(self, results: Dict[str, Any], question_id: str, parsed: Dict[str, Any]):
        """Add a single result to the results dictionary and save incrementally."""
        results['predictions'][question_id] = parsed.get('answer', ['Fail'])

        if 'analysis' in parsed:
            results['analyses'][question_id] = parsed['analysis']

        if 'thinking' in parsed and 'thinkings' in results:
            results['thinkings'][question_id] = parsed['thinking']

        if 'raw_percentages' in parsed:
            if 'percentages' not in results:
                results['percentages'] = {}
            results['percentages'][question_id] = parsed.get('raw_percentages')

        # Save incrementally if experiment path is set
        if self.experiment_path:
            from src.experiments.manager import save_experiment_results_incremental
            save_experiment_results_incremental(
                experiment_path=self.experiment_path,
                question_id=question_id,
                prediction=parsed.get('answer', ['Fail']),
                analysis=parsed.get('analysis', ''),
                thinking=parsed.get('thinking')
            )

    def _handle_error(self, results: Dict[str, Any], question_id: str, error: Exception):
        """Handle inference errors gracefully."""
        tqdm.write(f"⚠️  Error on question {question_id}: {error}")
        results['predictions'][question_id] = ['Fail']  # Fallback
        results['analyses'][question_id] = f"Error: {str(error)}"

    def _create_progress_bar(self, total: int) -> tqdm:
        """Create standardized progress bar."""
        return tqdm(
            total=total,
            desc=f"🔮 {self.get_model_name()} Processing",
            unit="question",
            colour=self.get_progress_color(),
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    # ========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> Any:
        """
        Format the prompt for the model.

        Args:
            question: Question dictionary
            context_docs: List of context documents

        Returns:
            Formatted prompt (type depends on model - could be string, list, etc.)
        """

    @abstractmethod
    def call_model(self, prompt: Any) -> Any:
        """
        Call the model with the formatted prompt.

        Args:
            prompt: Formatted prompt from format_prompt()

        Returns:
            Raw response (str or dict, depending on model)
        """

    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]:
        """
        Parse the model's response.

        Args:
            response: Raw response from call_model()

        Returns:
            Dictionary with 'answer' (list[str]), 'analysis' (str), 
            optionally 'thinking' (str), and 'flag' (bool)
        """

    # ========================================================================
    # CONFIGURATION METHODS (Can be overridden)
    # ========================================================================

    def supports_caching(self) -> bool:
        """Whether this model supports prompt caching."""
        return False

    def supports_thinking(self) -> bool:
        """Whether this model supports thinking mode."""
        return False

    def get_model_name(self) -> str:
        """Get display name for progress bar."""
        return self.__class__.__name__.replace('Inference', '')

    def get_progress_color(self) -> str:
        """Get progress bar color."""
        return "blue"
