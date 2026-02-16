#!/usr/bin/env python3
"""
Optimize Zeroshot Prompts Using GEPA (Streamlined Version)

This script uses DSPy's GEPA optimizer to improve the zeroshot prompt for
abductive causal reasoning tasks (SemEval 2026 Task 12).

KEY DESIGN PRINCIPLES:
======================
1. SIMPLE SIGNATURE: One-sentence docstring that GEPA can optimize
   - Gives the optimizer room to discover better instructions
   - Avoids over-constraining with prescriptive initial prompts

2. CONCISE REFLECTION PROMPT: Focused guidance for the reflection model
   - Task description + common fixes
   - No verbose domain-specific instructions that overwhelm the model

3. STREAMLINED METRIC: Clear, actionable feedback
   - Simple quality checks (depth, causal terms)
   - Concise error messages with fix suggestions

INPUTS:
  - context_documents: Context documents for the topic
  - target_event: Target event to find causes for
  - options: Answer options A, B, C, D

OUTPUTS:
  - analysis: Step-by-step causal analysis
  - answer: Selected answer(s)

Usage:
  python scripts/optimize_prompts.py --data-dir data/raw/semeval2026-task12-dataset
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pickle
import random
from collections import defaultdict

import subprocess
import threading

import dspy
from dspy.teleprompt import GEPA
from dspy.evaluate import Evaluate
from tqdm import tqdm

# Memory management
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import gepa utils for Pareto front extraction
try:
    from gepa.gepa_utils import find_dominator_programs
    HAS_GEPA_UTILS = True
except ImportError:
    HAS_GEPA_UTILS = False

# Network resilience
import time
from urllib3.exceptions import ProtocolError, MaxRetryError
from requests.exceptions import ConnectionError, Timeout, RequestException

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dspy_bedrock_lm import BedrockClaudeLM
from src.models.dspy_gemini_lm import GeminiLM
from src.utils.cost_tracker import CostTracker
from src.evaluation.scoring import normalize_answer, calculate_match_type
from configs.aws_config import get_bedrock_client, get_model_id
from configs.google_config import get_gemini_client, get_gemini_model_id

# DSPy signatures for prompt optimization
import dspy

class ZeroshotStructuredBaseline(dspy.Signature):
    """Given context documents and a question about causal reasoning, identify the correct answer(s)."""
    context = dspy.InputField(desc="Context documents")
    question = dspy.InputField(desc="Target event and answer options")
    analysis = dspy.OutputField(desc="Brief analysis of each option")
    answer = dspy.OutputField(desc="Letter(s) of correct option(s)")

# Disable LiteLLM (use native Bedrock)
os.environ["LITELLM_DISABLED"] = "true"
os.environ["DSPY_USE_LITELLM"] = "0"


# =====================================================================
# MEMORY MANAGEMENT UTILITIES
# =====================================================================

def cleanup_memory(verbose: bool = False):
    """
    Safe memory cleanup to reduce RAM usage without losing important data.

    This function:
    - Runs Python's garbage collector to free unreferenced objects
    - Clears PyTorch CUDA cache (GPU memory only, doesn't affect CPU)
    - Does NOT clear any active model weights, checkpoints, or training data

    Safe to call periodically during long-running operations.
    """
    if verbose:
        print("   🧹 Cleaning up memory...")

    # Force garbage collection - only collects objects with no references
    # This is SAFE: it won't delete anything still in use
    collected = gc.collect()

    # Clear PyTorch CUDA cache if available
    # This only affects GPU memory, not model weights or data
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if verbose:
        print(f"   🧹 Collected {collected} unreferenced objects")


# =====================================================================
# NETWORK RESILIENCE UTILITIES
# =====================================================================
def is_network_error(error: Exception) -> bool:
    """Check if error is a transient network error (retryable)."""
    # Check exception types
    if isinstance(error, (ConnectionError, Timeout, ProtocolError, MaxRetryError)):
        return True

    error_str = str(error).lower()

    # Check error messages
    network_indicators = [
        "connection",
        "timeout",
        "network",
        "unreachable",
        "connection reset",
        "broken pipe",
        "name resolution",
        "dns",
    ]

    return any(indicator in error_str for indicator in network_indicators)


def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            # Check if this is a retryable error
            if not is_network_error(e):
                # Not a network error, re-raise immediately
                raise

            # Last attempt, re-raise
            if attempt == max_retries:
                print(f"\n❌ Network error persisted after {max_retries} retries")
                raise

            # Calculate delay with exponential backoff
            if jitter:
                import random
                delay_with_jitter = delay * (0.5 + random.random())
            else:
                delay_with_jitter = delay

            delay_with_jitter = min(delay_with_jitter, max_delay)

            print(f"\n⚠️  Network error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            print(f"   Retrying in {delay_with_jitter:.1f}s...")

            time.sleep(delay_with_jitter)

            # Increase delay for next attempt
            delay = min(delay * exponential_base, max_delay)


# =====================================================================
# LOGGING SETUP
# =====================================================================

class TeeOutput:
    """Tee-like object that writes to both terminal and log file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_logging(output_dir: Path, log_filename: str = "optimization.log"):
    """Setup logging to capture all output to both terminal and log file."""
    log_path = output_dir / log_filename
    log_file = open(log_path, 'a', encoding='utf-8')

    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"OPTIMIZATION RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*80}\n\n")
    log_file.flush()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)

    print(f"📝 Logging to: {log_path}")

    return log_file, original_stdout, original_stderr


def teardown_logging(log_file, original_stdout, original_stderr):
    """Restore original stdout/stderr and close log file."""
    if log_file and not log_file.closed:
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"OPTIMIZATION ENDED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*80}\n")
        log_file.flush()
        log_file.close()

    sys.stdout = original_stdout
    sys.stderr = original_stderr


# =====================================================================
# CHECKPOINT AND DAILY LIMIT MANAGEMENT
# =====================================================================
class DailyLimitTracker:
    """Track daily API usage and costs to prevent exceeding budget."""

    def __init__(self, daily_limit_usd: float = None):
        self.daily_limit_usd = daily_limit_usd
        self.total_cost = 0.0
        self.total_calls = 0
        self.exceeded = False

    def add_cost(self, cost: float):
        """Add cost from an API call."""
        self.total_cost += cost
        self.total_calls += 1

        if self.daily_limit_usd and self.total_cost >= self.daily_limit_usd:
            self.exceeded = True

    def is_exceeded(self) -> bool:
        """Check if daily limit has been exceeded."""
        return self.exceeded

    def get_remaining(self) -> float:
        """Get remaining budget."""
        if not self.daily_limit_usd:
            return float('inf')
        return max(0, self.daily_limit_usd - self.total_cost)

    def get_summary(self) -> Dict:
        """Get usage summary."""
        return {
            'total_cost': self.total_cost,
            'total_calls': self.total_calls,
            'daily_limit': self.daily_limit_usd,
            'remaining': self.get_remaining(),
            'exceeded': self.exceeded
        }


class CheckpointManager:
    """Manage checkpoints for resuming GEPA optimization."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.pkl"
        self.state_file = self.checkpoint_dir / "state.json"

    def save_checkpoint(self, gepa_optimizer, iteration: int,
                       train_examples: List, val_examples: List,
                       daily_tracker: DailyLimitTracker = None,
                       cost_tracker: CostTracker = None,
                       metadata: Dict = None):
        """Save full checkpoint state."""
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'train_examples': train_examples,
            'val_examples': val_examples,
            'daily_tracker': daily_tracker.get_summary() if daily_tracker else None,
            'cost_tracker': cost_tracker.get_summary() if cost_tracker else None,
            'metadata': metadata or {}
        }

        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        state_info = {
            'iteration': iteration,
            'timestamp': checkpoint_data['timestamp'],
            'num_train': len(train_examples) if train_examples else 0,
            'num_val': len(val_examples) if val_examples else 0,
            'daily_tracker': daily_tracker.get_summary() if daily_tracker else None,
            'cost_tracker': cost_tracker.get_summary() if cost_tracker else None,
            'metadata': metadata or {}
        }

        with open(self.state_file, 'w') as f:
            json.dump(state_info, f, indent=2)

        print(f"\n💾 Checkpoint saved: {self.checkpoint_file}")
        if daily_tracker:
            print(f"   Cost so far: ${daily_tracker.total_cost:.4f} ({daily_tracker.total_calls} calls)")

    def has_checkpoint(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_file.exists()

    def load_checkpoint(self) -> Dict:
        """Load checkpoint state."""
        if not self.has_checkpoint():
            return None

        print(f"\n📂 Loading checkpoint from: {self.checkpoint_file}")

        with open(self.checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        print(f"   Iteration: {checkpoint_data['iteration']}")
        print(f"   Timestamp: {checkpoint_data['timestamp']}")
        if checkpoint_data.get('daily_tracker'):
            dt = checkpoint_data['daily_tracker']
            print(f"   Previous cost: ${dt['total_cost']:.4f} ({dt['total_calls']} calls)")

        return checkpoint_data

    def clear_checkpoint(self):
        """Remove checkpoint files."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.state_file.exists():
            self.state_file.unlink()


class ParetoProgressDisplay:
    """Display live Pareto front progress during optimization."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / "pareto_progress.json"
        self.prompts_dir = self.output_dir / "live_prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        self.iterations = []

    def update(self, iteration: int, candidates, scores, pareto_indices=None, best_idx=None):
        """Update progress with current state."""
        iteration_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'num_candidates': len(candidates) if candidates else 0,
            'scores': [float(s) for s in scores] if scores else [],
            'best_score': float(max(scores)) if scores else 0.0,
            'pareto_size': len(pareto_indices) if pareto_indices else 0,
            'pareto_indices': sorted(list(pareto_indices)) if pareto_indices else [],
            'best_idx': best_idx
        }

        self.iterations.append(iteration_data)

        with open(self.progress_file, 'w') as f:
            json.dump({
                'iterations': self.iterations,
                'latest': iteration_data
            }, f, indent=2)

        if candidates and pareto_indices:
            for idx in pareto_indices:
                if idx < len(candidates):
                    try:
                        instructions = extract_instructions_from_candidate(candidates[idx])
                        if instructions:
                            prompt_file = self.prompts_dir / f"iter{iteration:03d}_idx{idx}_score{scores[idx]:.4f}.txt"
                            with open(prompt_file, 'w') as f:
                                f.write(f"# Iteration: {iteration}\n")
                                f.write(f"# Candidate Index: {idx}\n")
                                f.write(f"# Score: {scores[idx]:.4f}\n")
                                f.write("# In Pareto Front: YES\n")
                                f.write(f"# Best: {'YES' if idx == best_idx else 'NO'}\n")
                                f.write(f"# {'='*60}\n\n")
                                f.write(instructions)
                    except Exception as e:
                        print(f"   ⚠️ Could not extract prompt for idx={idx}: {e}")

        print(f"\n📊 Iteration {iteration} Progress:")
        print(f"   Candidates: {iteration_data['num_candidates']}")
        print(f"   Best Score: {iteration_data['best_score']:.4f}")
        if pareto_indices:
            print(f"   Pareto Front: {len(pareto_indices)} programs")
            print(f"   Pareto Indices: {sorted(list(pareto_indices))}")

    def print_final_summary(self):
        """Print final optimization summary."""
        if not self.iterations:
            return

        print(f"\n{'='*70}")
        print("OPTIMIZATION PROGRESS SUMMARY")
        print(f"{'='*70}")
        print(f"Total Iterations: {len(self.iterations)}")

        best_scores = [it['best_score'] for it in self.iterations]
        print("Score Progression:")
        for i, score in enumerate(best_scores):
            print(f"   Iter {i}: {score:.4f}")

        final = self.iterations[-1]
        print("\nFinal State:")
        print(f"   Candidates: {final['num_candidates']}")
        print(f"   Best Score: {final['best_score']:.4f}")
        print(f"   Pareto Front Size: {final['pareto_size']}")
        print(f"{'='*70}\n")


# =====================================================================
# UTILITIES
# =====================================================================

def load_gepa_state(warm_start_dir: Path) -> Dict[str, Any]:
    """Load GEPA state from a previous experiment directory for warm-starting."""
    gepa_state_file = warm_start_dir / "gepa_checkpoints" / "gepa_state.bin"
    
    if not gepa_state_file.exists():
        raise FileNotFoundError(f"GEPA state file not found: {gepa_state_file}")
    
    print(f"\n🔥 WARM-START: Loading GEPA state from: {gepa_state_file}")
    
    with open(gepa_state_file, 'rb') as f:
        gepa_state = pickle.load(f)
    
    # Extract key information
    num_candidates = len(gepa_state.get('program_candidates', []))
    scores = gepa_state.get('program_full_scores_val_set', [])
    best_score = max(scores) if scores else 0.0
    best_idx = scores.index(best_score) if scores else 0
    
    print(f"   📊 Loaded {num_candidates} existing candidates")
    print(f"   📊 Best score: {best_score:.4f} (candidate {best_idx})")
    print(f"   📊 Score range: {min(scores):.4f} - {max(scores):.4f}")
    
    # Check for Pareto front info
    if 'program_at_pareto_front_valset' in gepa_state:
        # The Pareto front is stored per validation instance
        # Each instance has a set of program indices that are in its Pareto front
        # To get the global Pareto front, we need to find programs that dominate across instances
        pareto_per_instance = gepa_state['program_at_pareto_front_valset']
        
        # Count how many instances each program appears in the Pareto front
        program_counts = {}
        for instance_pareto in pareto_per_instance:
            if isinstance(instance_pareto, (set, frozenset)):
                for prog_idx in instance_pareto:
                    program_counts[prog_idx] = program_counts.get(prog_idx, 0) + 1
            elif isinstance(instance_pareto, (list, tuple)):
                for prog_idx in instance_pareto:
                    program_counts[prog_idx] = program_counts.get(prog_idx, 0) + 1
        
        # Programs in the global Pareto front appear in at least one instance
        global_pareto = set(program_counts.keys())
        print(f"   📊 Pareto front: {len(global_pareto)} programs across {len(pareto_per_instance)} validation instances")
    
    return gepa_state


def _has_none_option(options: Dict[str, Any]) -> bool:
    phrases = [
        "none of the others",
        "none of the above",
        "none are correct",
        "none of the others are correct causes"
    ]
    for key in ["A", "B", "C", "D", "option_A", "option_B", "option_C", "option_D"]:
        if key in options and isinstance(options[key], str):
            text = options[key].lower()
            if any(p in text for p in phrases):
                return True
    return False


def load_zeroshot_examples(args) -> List:
    """
    Load dev/test questions with context documents as DSPy examples.

    Loads questions and optimizes the zeroshot_structured prompt
    from run_experiment.py (system prompt + context docs + target + options).
    """
    from src.data.loader import load_dev_data, load_test_data

    data_path = Path(args.data_dir)

    if args.dataset == "test":
        questions, docs = load_test_data(data_path)
    else:
        questions, docs = load_dev_data(data_path)

    print(f"   Loaded {len(questions)} questions from {args.dataset} set")

    # Prepare topic->docs mapping
    topic2docs: Dict[str, List[Dict]] = {}
    for d in docs:
        tid = d.get("topic_id")
        topic2docs.setdefault(tid, []).extend(d.get("docs", []))

    # Optionally load preprocessed GraphRAG contexts
    context_cache = None
    if args.use_graph_rag:
        from src.retrieval.graph_rag_utils import PreprocessedContextCache
        cache_split = args.dataset if args.dataset != "val" else "train"
        context_cache = PreprocessedContextCache(split=cache_split, docs=docs)

    examples = []
    for q in questions:
        # Get context (from preprocessed cache or raw topic docs)
        # IMPORTANT: Match EXACT format from run_experiment.py's format_prompt
        if context_cache and context_cache.using_preprocessed:
            # Preprocessed cache returns already-formatted context
            context_text = context_cache.get_context(q['topic_id'], q)
        else:
            ctx_docs = topic2docs.get(q['topic_id'], [])
            # Match run_experiment.py format: <document_N>: content</document_N> (with colon)
            context_text = "\n\n".join(
                f"<document_{j+1}>: {doc['content']}</document_{j+1}>"
                for j, doc in enumerate(ctx_docs)
            )

        # Wrap context in <context_documents> tags with topic_id (EXACT match to run_experiment.py)
        context_documents = f"""<context_documents>
<topic_id>{q.get('topic_id', '')}</topic_id>
{context_text}
</context_documents>"""

        # Wrap target_event in <target_event> tags (EXACT match to run_experiment.py)
        target_event = f"<target_event>{q['target_event']}</target_event>"

        # Wrap options in <options> tags (EXACT match to run_experiment.py)
        options = f"""<options>
<option_a>{q['option_A']}</option_a>
<option_b>{q['option_B']}</option_b>
<option_c>{q['option_C']}</option_c>
<option_d>{q['option_D']}</option_d>
</options>"""

        # Test dataset doesn't have answers (hidden for competition)
        answer = q.get('answer', 'A')  # Dummy answer for test set

        example = dspy.Example(
            context_documents=context_documents,
            target_event=target_event,
            options=options,
            answer=answer
        ).with_inputs("context_documents", "target_event", "options")
        examples.append(example)

    return examples


def stratified_sample_by_topic(questions: List[Dict], max_questions: int, seed: int = 42) -> List[Dict]:
    """
    Sample questions stratified by topic_id to ensure equal representation.

    Args:
        questions: List of question dictionaries with 'topic_id' field
        max_questions: Maximum number of questions to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled list of questions with balanced topic representation
    """
    if max_questions >= len(questions):
        return questions

    random.seed(seed)

    # Group questions by topic
    topics_to_questions = defaultdict(list)
    for q in questions:
        topic_id = q.get('topic_id', 'unknown')
        topics_to_questions[topic_id].append(q)

    num_topics = len(topics_to_questions)
    questions_per_topic = max_questions // num_topics
    remainder = max_questions % num_topics

    print(f"\n📊 Stratified Sampling by Topic:")
    print(f"   Total questions: {len(questions)}")
    print(f"   Target sample size: {max_questions}")
    print(f"   Number of topics: {num_topics}")
    print(f"   Base questions per topic: {questions_per_topic}")
    if remainder > 0:
        print(f"   Extra questions to distribute: {remainder}")

    sampled_questions = []
    topic_sample_counts = {}

    # Sort topics for deterministic ordering
    sorted_topics = sorted(topics_to_questions.keys())

    # Sample from each topic
    for topic_id in sorted_topics:
        topic_questions = topics_to_questions[topic_id]

        # Calculate sample size for this topic
        sample_size = questions_per_topic
        if remainder > 0:
            sample_size += 1
            remainder -= 1

        # Don't sample more than available
        sample_size = min(sample_size, len(topic_questions))

        # Random sample from this topic
        sampled = random.sample(topic_questions, sample_size)
        sampled_questions.extend(sampled)
        topic_sample_counts[topic_id] = sample_size

        print(f"   Topic {topic_id}: sampled {sample_size}/{len(topic_questions)} questions")

    # Shuffle the final list to mix topics
    random.shuffle(sampled_questions)

    print(f"   ✅ Sampled {len(sampled_questions)} questions total\n")

    return sampled_questions


def extract_instructions_from_candidate(candidate) -> str:
    """Extract instructions from a GEPA candidate program."""
    instructions = None

    # Handle dictionary candidates (common format from GEPA state)
    if isinstance(candidate, dict):
        # Try common keys for prompt storage
        for key in ['answerer.predict', 'answerer', 'predict', 'instructions']:
            if key in candidate:
                val = candidate[key]
                # If value is a string (the prompt itself), return it
                if isinstance(val, str):
                    return val
                # If value is another dict, try to extract instructions from it
                if isinstance(val, dict) and 'instructions' in val:
                    return val['instructions']
        return None

    # Handle module candidates (DSPy objects)
    module = None

    if hasattr(candidate, 'answerer'):
        module = candidate.answerer
        if hasattr(module, 'predict'):
            module = module.predict
    elif hasattr(candidate, 'predict'):
        module = candidate.predict
    else:
        for attr_name in dir(candidate):
            if attr_name.startswith('_'):
                continue
            attr = getattr(candidate, attr_name, None)
            if attr is None:
                continue
            if hasattr(attr, 'extended_signature') or hasattr(attr, 'signature'):
                module = attr
                if hasattr(module, 'predict'):
                    module = module.predict
                break

    if module is None:
        return None

    if hasattr(module, 'extended_signature'):
        ext_sig = module.extended_signature
        if hasattr(ext_sig, 'instructions') and ext_sig.instructions:
            instructions = ext_sig.instructions

    if not instructions and hasattr(module, 'signature'):
        sig = module.signature
        if hasattr(sig, 'instructions') and sig.instructions:
            instructions = sig.instructions

    if not instructions and hasattr(module, 'signature'):
        sig = module.signature
        if hasattr(sig, '__doc__') and sig.__doc__:
            instructions = sig.__doc__

    return instructions


def extract_gepa_stats(detailed_results) -> Dict[str, Any]:
    """Extract serializable statistics from GEPA detailed_results object."""
    if not detailed_results:
        return None

    stats = {}

    try:
        if hasattr(detailed_results, 'best_idx'):
            stats['best_program_idx'] = detailed_results.best_idx

        if hasattr(detailed_results, 'best_score'):
            stats['best_score'] = float(detailed_results.best_score)

        if hasattr(detailed_results, 'candidates'):
            stats['num_candidates'] = len(detailed_results.candidates)

        if hasattr(detailed_results, 'val_aggregate_scores'):
            val_scores = detailed_results.val_aggregate_scores
            stats['validation_scores'] = {
                'all_scores': [float(s) for s in val_scores] if val_scores else [],
                'mean': float(sum(val_scores) / len(val_scores)) if val_scores else 0.0,
                'max': float(max(val_scores)) if val_scores else 0.0,
                'min': float(min(val_scores)) if val_scores else 0.0
            }

        if hasattr(detailed_results, 'train_aggregate_scores'):
            train_scores = detailed_results.train_aggregate_scores
            stats['training_scores'] = {
                'all_scores': [float(s) for s in train_scores] if train_scores else [],
                'mean': float(sum(train_scores) / len(train_scores)) if train_scores else 0.0,
                'max': float(max(train_scores)) if train_scores else 0.0,
                'min': float(min(train_scores)) if train_scores else 0.0
            }

        if HAS_GEPA_UTILS and hasattr(detailed_results, 'per_val_instance_best_candidates'):
            try:
                pareto_front = find_dominator_programs(
                    detailed_results.per_val_instance_best_candidates,
                    detailed_results.val_aggregate_scores
                )
                stats['pareto_front'] = {
                    'size': len(pareto_front),
                    'program_indices': sorted(list(pareto_front)),
                    'scores': [float(detailed_results.val_aggregate_scores[idx]) for idx in sorted(pareto_front)]
                }
            except Exception as e:
                stats['pareto_front_error'] = str(e)

        if hasattr(detailed_results, 'history'):
            history = detailed_results.history
            if history:
                stats['optimization_history'] = {
                    'num_iterations': len(history),
                    'score_progression': [float(h.get('score', 0)) for h in history if isinstance(h, dict)]
                }

        if hasattr(detailed_results, 'num_metric_calls'):
            stats['num_metric_calls'] = detailed_results.num_metric_calls

        if hasattr(detailed_results, 'num_reflections'):
            stats['num_reflections'] = detailed_results.num_reflections

    except Exception as e:
        stats['extraction_error'] = str(e)

    return stats


def dag_to_dot(parent_program_for_candidate, dominator_program_ids, best_program_idx, full_eval_scores):
    """Generate DOT graph for optimization DAG"""
    dot_lines = [
        "digraph G {",
        "    node [style=filled, shape=circle, fontsize=50];"
    ]
    n = len(parent_program_for_candidate)
    for idx in range(n):
        score = full_eval_scores[idx]
        label = f"{idx}\\n({score:.2f})"
        if idx == best_program_idx:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=cyan, fontcolor=black];')
        elif idx in dominator_program_ids:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=orange, fontcolor=black];')
        else:
            dot_lines.append(f'    {idx} [label="{label}"];')

    for child, parents in enumerate(parent_program_for_candidate):
        for parent in parents:
            if parent is not None:
                dot_lines.append(f'    {parent} -> {child};')

    dot_lines.append("}")
    return "\n".join(dot_lines)


# =====================================================================
# STREAMLINED METRIC - Concise feedback for GEPA optimization
# =====================================================================

# Expanded list of causal terms for quality checking
CAUSAL_TERMS = [
    "cause", "because", "therefore", "leads to", "results in",
    "due to", "consequently", "hence", "thus", "since", "as a result",
    "triggers", "enables", "prevents", "follows from", "implies"
]


def _calculate_answer_score(pred_answer: str, gold_answer: str) -> tuple:
    """Calculate answer correctness score and match type."""
    if not pred_answer:
        return 0.0, "empty", "No answer produced."

    try:
        pset = set(x.strip().upper() for x in pred_answer.split(",") if x.strip())
        gset = set(x.strip().upper() for x in gold_answer.split(",") if x.strip())

        if not gset:
            return 0.0, "no_gold", "No gold answer to compare."

        if pset == gset:
            return 1.0, "exact", "Correct."
        elif pset and pset.issubset(gset):
            missed = gset - pset
            return 0.5, "partial", f"Partial: missed {missed}. Fix: Check all paths."
        elif gset.issubset(pset):
            extra = pset - gset
            return 0.0, "superset", f"Over-selected: {extra}. Fix: Verify causal strength."
        else:
            return 0.0, "mismatch", f"Wrong: got {pset}, expected {gset}. Fix: Trace paths carefully."

    except Exception as e:
        return 0.0, "error", f"Parse error: {str(e)[:50]}"


def create_lm_clients(args, key_index: Optional[int] = None):
    """
    Create LM clients for task and reflection models.

    Args:
        args: Command-line arguments
        key_index: Optional key index for Gemini (overrides args.key)

    Returns:
        Tuple of (task_lm, reflection_lm, gemini_client, bedrock_client)
    """
    gemini_client = None
    bedrock_client = None
    effective_key_index = key_index if key_index is not None else args.key

    # Determine reflection model family
    reflection_family = args.reflection_model_family or args.model_family

    # Initialize task LM
    if args.model_family == "gemini":
        gemini_client = get_gemini_client(key_index=effective_key_index)
        model_id = get_gemini_model_id(args.model)

        lm = GeminiLM(
            gemini_client,
            model_id=model_id,
            model_name=f"gemini.{args.model}",
            requests_per_minute=50,
            use_cache=False,
            enable_context_caching=False
        )
        print(f"   Task LM: Gemini {model_id} (key #{effective_key_index or 'default'})")
    else:
        bedrock_client = get_bedrock_client(region="us-east-1")
        model_id = get_model_id(args.model)

        lm = BedrockClaudeLM(
            bedrock_client,
            model_id,
            model_name=f"bedrock.{args.model}",
            requests_per_minute=6,
            use_cache=False,
            enable_context_caching=False
        )
        print(f"   Task LM: Claude (Bedrock) {model_id}")

    def _reflection_prompt_text() -> str:
        return """You optimize prompts for abductive causal reasoning tasks.

TASK: Given context documents, a target event, and options (A/B/C/D), identify which options are plausible causes of the target event.

WHEN IMPROVING PROMPTS:
1. Analyze failure patterns in the traces (shallow reasoning, wrong selections, missed paths)
2. Add specific, actionable steps that address the failures
3. Keep instructions concise but complete
4. Avoid overfitting - no specific examples, entity names, or memorized answers
5. DO NOT add domain-specific knowledge, world facts, or question-specific hints
6. DO NOT add or imply any content not derivable from the input context/target/options

COMMON FIXES:
- Shallow reasoning → Add "analyze each option step-by-step"
- Wrong selections → Add "verify causal evidence exists before selecting"
- Missed paths → Add "check both direct and indirect causal links"
- Over-selection → Add "only select options with STRONG causal evidence"

OUTPUT: Return ONLY the improved prompt text. No explanations or meta-commentary."""

    class PrefixedLM(dspy.LM):
        """Wrap a DSPy LM to prepend a system prompt (Gemini-compatible)."""
        def __init__(self, base_lm, system_prompt: str):
            try:
                super().__init__(getattr(base_lm, "model_name", "prefixed"))
            except Exception:
                # Some dspy versions don't require/accept init args
                try:
                    super().__init__()
                except Exception:
                    pass
            self.base_lm = base_lm
            self.system_prompt = system_prompt
            self.last_usage = getattr(base_lm, "last_usage", {})

        def forward(self, prompt=None, messages=None, **kwargs):
            if messages is None:
                messages = []
            if prompt and not messages:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "system", "content": self.system_prompt}] + list(messages)
            response = self.base_lm.forward(messages=messages, **kwargs)
            self.last_usage = getattr(self.base_lm, "last_usage", {})
            return response

        def __getattr__(self, name):
            return getattr(self.base_lm, name)

    # Initialize reflection LM
    if reflection_family == "gemini":
        if not gemini_client:
            gemini_client = get_gemini_client(key_index=effective_key_index)

        reflection_model_id = get_gemini_model_id(args.reflection_model)

        reflection_lm = GeminiLM(
            gemini_client,
            model_id=reflection_model_id,
            model_name="gemini.reflection",
            requests_per_minute=50,
            use_cache=False,
            enable_context_caching=False
        )
        use_custom_prompt = not args.use_default_gepa_prompt
        if use_custom_prompt:
            reflection_lm = PrefixedLM(reflection_lm, _reflection_prompt_text())
            print("   ✅ Using custom reflection prompt (prefixed) for Gemini")
        print(f"   Reflection LM: Gemini {reflection_model_id}")
    else:
        if not bedrock_client:
            bedrock_client = get_bedrock_client(region="us-east-1")

        reflection_model_id = get_model_id(args.reflection_model)
        use_thinking = args.use_thinking and "opus" in args.reflection_model.lower()
        thinking_budget = 10000 if use_thinking else 0

        use_custom_prompt = not args.use_default_gepa_prompt
        system_prompt_to_use = _reflection_prompt_text() if use_custom_prompt else None

        reflection_lm = BedrockClaudeLM(
            bedrock_client,
            reflection_model_id,
            system_prompt=system_prompt_to_use,
            model_name="bedrock.reflection",
            enable_thinking=use_thinking,
            thinking_budget=thinking_budget,
            use_cache=False,
            enable_context_caching=False,
            requests_per_minute=5
        )
        print(f"   Reflection LM: Claude {reflection_model_id}")

    return lm, reflection_lm, gemini_client, bedrock_client


def dspy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Streamlined GEPA metric optimized for answer correctness.

    CRITICAL FIX: Predictor-level feedback uses 100% answer correctness.
    - We ONLY optimize for getting the right answer
    - No weight on reasoning style/quality
    - Rationale: If good reasoning helps produce correct answers, GEPA will
      naturally evolve prompts that elicit good reasoning. We don't need to
      explicitly reward style—only outcomes.

    This gives GEPA the clearest optimization signal: improve prompts that
    lead to correct answers, period.
    """
    pred_answer = (getattr(pred, "answer", "") or "").strip()
    gold_answer = (getattr(gold, "answer", "") or "").strip()

    # === Predictor-level feedback (100% answer correctness) ===
    if pred_name == "answerer":
        reasoning = getattr(pred, "analysis", "") or getattr(pred, "reasoning", "") or ""

        # Check reasoning quality for diagnostic feedback only (not used in score)
        has_depth = len(reasoning) >= 100
        has_causal = any(t in reasoning.lower() for t in CAUSAL_TERMS)

        # Score is 100% based on answer correctness
        answer_score, match_type, answer_feedback = _calculate_answer_score(pred_answer, gold_answer)

        # Build diagnostic feedback (helps GEPA understand what went wrong)
        issues = []
        if match_type != "exact":
            issues.append(f"answer_{match_type}")

        # Add reasoning diagnostics only if answer is wrong
        if match_type != "exact":
            if not has_depth:
                issues.append("shallow_reasoning")
            if not has_causal:
                issues.append("missing_causal_analysis")

        if issues:
            feedback = f"Issues: {', '.join(issues)}. {answer_feedback}"
        else:
            feedback = f"Correct. {answer_feedback}"

        return dspy.Prediction(score=answer_score, feedback=feedback)

    # === Program-level feedback (for answer correctness only) ===
    answer_score, match_type, feedback = _calculate_answer_score(pred_answer, gold_answer)
    return dspy.Prediction(score=answer_score, feedback=feedback)


# =====================================================================
# MAIN
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimize zeroshot prompts using GEPA")
    parser.add_argument("--data-dir", type=str, default="data/raw/semeval2026-task12-dataset",
                       help="Path to dataset directory")
    parser.add_argument("--dataset", type=str, default="dev",
                       choices=["dev", "test"],
                       help="Dataset split to use")
    parser.add_argument("--use-graph-rag", action="store_true",
                       help="Use preprocessed GraphRAG contexts")
    parser.add_argument("--model-family", type=str, default="claude",
                       choices=["claude", "gemini"],
                       help="Model family to use: claude (AWS Bedrock) or gemini (Google)")
    parser.add_argument("--model", type=str, default="claude-haiku-4.5",
                       help="Model to use for answerer (e.g., claude-haiku-4.5, gemini-3-flash-preview)")
    parser.add_argument("--gepa-budget", type=str, default="medium",
                       choices=["light", "medium", "heavy"],
                       help="GEPA optimization budget")
    parser.add_argument("--max-rollouts", type=int, default=None,
                       help="Override GEPA budget with specific max rollouts")
    parser.add_argument("--reflection-model", type=str, default=None,
                       help="Model to use for GEPA reflection (e.g., claude-opus-4.5, gemini-3-flash-preview). "
                            "Defaults to claude-opus-4.5 for Claude, gemini-3-flash-preview for Gemini")
    parser.add_argument("--reflection-model-family", type=str, default=None,
                       help="Model family for reflection (defaults to --model-family)")
    parser.add_argument("--use-thinking", action="store_true", default=True,
                       help="Enable extended thinking for reflection model (Claude only)")
    parser.add_argument("--skip-final-eval", action="store_true",
                       help="Skip final evaluation")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--min-val-size", type=int, default=15,
                       help="Minimum validation set size")
    parser.add_argument("--daily-limit", type=float, default=None,
                       help="Daily cost limit in USD")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous checkpoint")
    parser.add_argument("--warm-start", type=str, default=None,
                       help="Path to previous experiment directory to warm-start from (continues optimization with existing candidates)")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--log-file", type=str, default="optimization.log",
                       help="Log filename")
    parser.add_argument("--reflection-minibatch-size", type=int, default=10,
                       help="Number of examples in each reflection minibatch")
    parser.add_argument("--max-questions", type=int, default=None,
                       help="Maximum number of questions for GEPA training (train+val). "
                            "test-size additional questions will be used for final evaluation. "
                            "Total sampled = max-questions + test-size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for question sampling and GEPA")
    parser.add_argument("--use-default-gepa-prompt", action="store_true",
                       help="Use GEPA's default reflection prompt instead of custom domain-specific prompt")
    parser.add_argument("--test-size", type=int, default=200,
                       help="Number of questions to hold out for final evaluation (default: 200). "
                            "These are in addition to max-questions")
    parser.add_argument("--num-threads", type=int, default=1,
                       help="Number of threads for parallel GEPA evaluation (default: 1)")
    parser.add_argument("--key", type=int, default=None,
                       help="API key index from .google/other.json (1-6)")
    parser.add_argument("--holdout-test", action="store_true",
                       help="Hold out a test set for final evaluation (disabled by default - uses full dataset)")
    parser.add_argument("--max-merge-invocations", type=int, default=5,
                       help="Maximum merge invocations for GEPA (default: 5)")
    parser.add_argument("--exclude-none", action="store_true",
                       help="Exclude questions where any option is a NONE-of-the-others variant.")
    args = parser.parse_args()

    # Set default reflection model based on model family if not specified
    if args.reflection_model is None:
        if args.model_family == "gemini":
            args.reflection_model = "gemini-3-flash-preview"
        else:
            args.reflection_model = "claude-opus-4.5"

    print(f"\n{'='*70}")
    print("ZEROSHOT PROMPT OPTIMIZATION WITH GEPA")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"GraphRAG: {'enabled' if args.use_graph_rag else 'disabled'}")
    print(f"Model Family: {args.model_family}")
    print(f"Model: {args.model}")
    print("Signature: Simple one-sentence docstring for GEPA to optimize")
    print(f"{'='*70}\n")

    # Load questions with context docs
    print("Loading questions with context documents...")
    zeroshot_examples = load_zeroshot_examples(args)
    print(f"✅ Loaded {len(zeroshot_examples)} examples")

    # Apply max_questions sampling if specified
    if args.max_questions:
        total_needed = args.max_questions + args.test_size
        if total_needed < len(zeroshot_examples):
            random.seed(args.seed)
            random.shuffle(zeroshot_examples)
            zeroshot_examples = zeroshot_examples[:total_needed]
            print(f"   Sampled {total_needed} examples ({args.max_questions} train + {args.test_size} test)")

    # Initialize LM clients (will be created/recreated as needed)
    print("\n" + "="*70)
    print("INITIALIZING LM CLIENTS")
    print("="*70)
    lm, reflection_lm, gemini_client, bedrock_client = create_lm_clients(args)
    print("="*70 + "\n")

    dspy.settings.configure(lm=lm)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"experiments/answerer_optimization/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Results will be saved to: {output_dir}\n")

    # Setup logging
    log_file, original_stdout, original_stderr = setup_logging(output_dir, args.log_file)

    try:
        checkpoint_mgr = CheckpointManager(output_dir / "checkpoints")
        daily_tracker = DailyLimitTracker(daily_limit_usd=args.daily_limit)

        # Initialize cost tracker for detailed cost breakdown
        # Use full model identifier for cost tracking
        cost_model_name = f"{args.model_family}/{args.model}"
        cost_tracker = CostTracker(model_name=cost_model_name)

        if args.daily_limit:
            print(f"💰 Daily cost limit: ${args.daily_limit:.2f}\n")

        # Check for resume (auto-detect if checkpoint exists with completed GEPA)
        checkpoint_data = None
        # Auto-enable resume if GEPA already completed
        # Detection methods (in priority order):
        # 1. gepa_results.json exists (means full run completed successfully)
        # 2. eval_checkpoint.json exists (means GEPA finished, eval was interrupted)
        # 3. checkpoints/state.json has completed=True
        # 4. gepa_state.bin exists (GEPA finished but checkpoint may not reflect it)
        eval_checkpoint_file = output_dir / "eval_checkpoint.json"
        gepa_results_file = output_dir / "gepa_results.json"
        gepa_state_file = output_dir / "gepa_checkpoints" / "gepa_state.bin"

        # Check for completed results (means full pipeline completed)
        if gepa_results_file.exists() and gepa_state_file.exists():
            # Check if results are complete
            try:
                with open(gepa_results_file, 'r') as f:
                    existing_results = json.load(f)
                existing_total = existing_results.get('total', 0)

                # If results are complete (has all test examples), just display and exit
                if existing_total > 0:
                    print("\n✅ Experiment already completed!")
                    print(f"   Found existing results: {existing_total} examples evaluated")
                    # Note: avg_score is already stored as percentage (0-100)
                    print(f"   Score: {existing_results.get('avg_score', 0):.2f}%")
                    print(f"   Results file: {gepa_results_file}")
                    print("\n   To re-run evaluation, delete gepa_results.json first.")
                    teardown_logging(log_file, original_stdout, original_stderr)
                    sys.exit(0)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Could not parse gepa_results.json: {e}")
            except Exception as e:
                print(f"   ⚠️  Error reading gepa_results.json: {e}")

            # If we reach here, gepa_results.json exists but is empty/invalid
            # Check if there's an eval checkpoint to resume from
            if eval_checkpoint_file.exists():
                print("\n📂 Found eval checkpoint, will resume interrupted evaluation...")
                args.resume = True
            else:
                print("\n📂 Detected completed GEPA (gepa_results.json exists but incomplete)")
                print("   Will re-run final evaluation with optimized prompt...")
                args.resume = True
        # Check for eval checkpoint (means GEPA finished, eval was interrupted)
        elif eval_checkpoint_file.exists():
            print("\n📂 Detected interrupted final evaluation, resuming...")
            args.resume = True
        # Check for completed GEPA in state.json
        elif checkpoint_mgr.has_checkpoint() and (output_dir / "checkpoints" / "state.json").exists():
            with open(output_dir / "checkpoints" / "state.json") as f:
                state_data = json.load(f)
                if state_data.get('metadata', {}).get('completed'):
                    print("\n📂 Detected completed GEPA optimization (checkpoint completed=True)")
                    args.resume = True
        # Check for gepa_state.bin (GEPA saves this when it finishes)
        elif gepa_state_file.exists():
            print("\n📂 Detected GEPA state file, checking if GEPA completed...")
            # Verify it has valid candidates
            try:
                with open(gepa_state_file, 'rb') as f:
                    gepa_state = pickle.load(f)
                if gepa_state.get('program_candidates') and gepa_state.get('program_full_scores_val_set'):
                    print("   ✅ Valid GEPA state found, skipping to final evaluation")
                    args.resume = True
            except Exception as e:
                print(f"   ⚠️  Could not load GEPA state: {e}")

        if args.resume:
            # Try to load checkpoint if it exists
            if checkpoint_mgr.has_checkpoint():
                checkpoint_data = checkpoint_mgr.load_checkpoint()
                if checkpoint_data and checkpoint_data.get('daily_tracker'):
                    dt = checkpoint_data['daily_tracker']
                    daily_tracker.total_cost = dt['total_cost']
                    daily_tracker.total_calls = dt['total_calls']
                    daily_tracker.exceeded = dt['exceeded']
                    print(f"   ✅ Restored daily tracker: ${daily_tracker.total_cost:.4f} spent")

                if checkpoint_data and checkpoint_data.get('cost_tracker'):
                    ct = checkpoint_data['cost_tracker']
                    for stage_name, stage_data in ct.get('stages', {}).items():
                        cost_tracker.stages[stage_name] = stage_data
                    print(f"   ✅ Restored cost tracker: ${ct['total_cost']:.4f} total")
            else:
                # No checkpoint but we detected GEPA completion via other means
                checkpoint_data = {'metadata': {'completed': True}}

        # Handle warm-start: load best prompt from previous experiment
        warm_start_prompt = None
        if args.warm_start:
            warm_start_dir = Path(args.warm_start)
            if not warm_start_dir.exists():
                print(f"❌ Warm-start directory not found: {warm_start_dir}")
                teardown_logging(log_file, original_stdout, original_stderr)
                return
            
            # Load and display warm-start state info
            warm_start_state = load_gepa_state(warm_start_dir)
            
            # Find the best prompt from the previous run
            scores = warm_start_state.get('program_full_scores_val_set', [])
            candidates = warm_start_state.get('program_candidates', [])
            
            if scores and candidates and len(scores) == len(candidates):
                best_idx = scores.index(max(scores))
                best_score = scores[best_idx]
                best_candidate = candidates[best_idx]
                
                # Extract the prompt from the best candidate
                if isinstance(best_candidate, dict) and 'answerer.predict' in best_candidate:
                    warm_start_prompt = best_candidate['answerer.predict']
                    print("\n🔥 WARM-START: Using best prompt from previous run")
                    print(f"   📊 Best candidate: #{best_idx} with score {best_score:.4f}")
                    print(f"   📝 Prompt preview: {warm_start_prompt[:200]}...")
                else:
                    print("\n❌ WARM-START FAILED: Could not extract prompt from best candidate")
                    print(f"   Candidate type: {type(best_candidate)}")
                    if isinstance(best_candidate, dict):
                        print(f"   Available keys: {list(best_candidate.keys())}")
                    print("   ⚠️  Continuing WITHOUT warm-start (starting fresh)")
            else:
                print("\n❌ WARM-START FAILED: Could not find best prompt")
                print(f"   Scores count: {len(scores)}, Candidates count: {len(candidates)}")
                print("   ⚠️  Continuing WITHOUT warm-start (starting fresh)")
            
            # Also copy the previous cost info if available
            prev_cost_file = warm_start_dir / "cost_breakdown.json"
            if prev_cost_file.exists():
                with open(prev_cost_file, 'r') as f:
                    prev_cost = json.load(f)
                print(f"   📊 Previous run cost: ${prev_cost.get('total_cost', 0):.4f}")
                # Restore cost tracker state
                for stage_name, stage_data in prev_cost.get('stages', {}).items():
                    cost_tracker.stages[stage_name] = stage_data

        # Determine reflection model family (defaults to main model family)
        reflection_family = args.reflection_model_family or args.model_family

        # Save metadata
        num_examples = len(zeroshot_examples)
        metadata = {
            "mode": "zeroshot",
            "model_family": args.model_family,
            "model": args.model,
            "reflection_model_family": reflection_family,
            "reflection_model": args.reflection_model,
            "num_questions": num_examples,
            "signature": "simple_one_sentence_docstring",
            "warm_start_from": str(args.warm_start) if args.warm_start else None,
            "dataset": args.dataset,
            "use_graph_rag": args.use_graph_rag,
        }

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print("USING GEPA OPTIMIZER")
        if args.warm_start:
            print(f"🔥 WARM-START MODE: Continuing from {args.warm_start}")
        print(f"{'='*70}")
        print(f"Budget: {args.gepa_budget}")
        print(f"Reflection Model Family: {reflection_family}")
        print(f"Reflection Model: {args.reflection_model}")
        print(f"Num Threads: {args.num_threads}")
        print(f"{'='*70}\n")

        # Prepare training examples from loaded zeroshot data
        train_examples = zeroshot_examples
        print(f"📝 Optimizing zeroshot prompt (context docs + target + options)")
        print(f"   Optimizing the zeroshot_structured system prompt\n")

        if not train_examples:
            print("❌ No training examples were created.")
            print("   Possible causes:")
            print("   - wrong data dir path")
            print("   - --exclude-none filtered everything")
            print("   Exiting.")
            return

        print(f"Created {len(train_examples)} total examples\n")

        # Calculate train/val/test split
        total_examples = len(train_examples)

        if args.holdout_test:
            # HOLDOUT MODE: Reserve test set for final evaluation
            # Strategy: Always respect --test-size for final evaluation
            # If max_questions specified: use min(max_questions, available) for GEPA optimization
            # Remaining examples after test holdout go to GEPA training

            # Always hold out test_size examples for final evaluation (respect user's --test-size)
            test_size = min(args.test_size, total_examples // 2)  # At most 50% for test
            available_for_optimization = total_examples - test_size

            if args.max_questions and args.max_questions < available_for_optimization:
                # User specified max_questions, cap optimization set
                optimization_examples = args.max_questions
            else:
                # Use all available examples for optimization
                optimization_examples = available_for_optimization

            if test_size < 10 and test_size > 0:
                print(f"⚠️  Warning: Only {test_size} examples for test set")

            # Split GEPA training examples into train/val
            val_size = max(args.min_val_size, round(optimization_examples * (1 - args.train_split)))
            train_size = optimization_examples - val_size

            if val_size > optimization_examples:
                print(f"⚠️  Warning: Only {optimization_examples} examples for optimization, adjusting split")
                val_size = max(5, optimization_examples // 4)
                train_size = optimization_examples - val_size

            # Data layout: [train_examples[:train_size]] [train_examples[train_size:optimization_examples]] [train_examples[optimization_examples:]]
            # Store test examples separately to avoid data leakage
            test_examples = train_examples[optimization_examples:] if test_size > 0 else []
            optimization_train = train_examples[:train_size]
            optimization_val = train_examples[train_size:optimization_examples]

            print("📊 Train/Val/Test Split (Holdout Mode):")
            print(f"   Train (GEPA): {train_size} examples ({train_size/total_examples*100:.0f}%)")
            print(f"   Val (GEPA):   {val_size} examples ({val_size/total_examples*100:.0f}%)")
            print(f"   Test (Final): {test_size} examples ({test_size/total_examples*100:.0f}%) - HELD OUT\n")
        else:
            # DEFAULT: Use ALL data for GEPA and final eval (no holdout)
            print("📊 Using FULL DATASET for optimization and evaluation (no holdout)")
            optimization_examples = total_examples
            test_size = 0
            test_examples = []  # No holdout - final eval uses same data

            # Split all data between GEPA train/val
            val_size = max(args.min_val_size, round(total_examples * (1 - args.train_split)))
            train_size = total_examples - val_size

            optimization_train = train_examples[:train_size]
            optimization_val = train_examples[train_size:]

            print("📊 Train/Val Split:")
            print(f"   Train (GEPA): {train_size} examples ({train_size/total_examples*100:.0f}%)")
            print(f"   Val (GEPA):   {val_size} examples ({val_size/total_examples*100:.0f}%)")
            print(f"   Final Eval:   ALL {total_examples} examples (same as train+val)\n")

        # Define the zeroshot program class
        class ZeroshotProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.answerer = dspy.ChainOfThought(ZeroshotStructuredBaseline)

            def forward(self, context_documents, target_event, options):
                return self.answerer(
                    context_documents=context_documents,
                    target_event=target_event,
                    options=options
                )

        base_program = ZeroshotProgram()

        # Apply warm-start prompt if available
        if warm_start_prompt:
            print("\n🔥 Applying warm-start prompt to base program...")
            try:
                prompt_applied = False
                # For ChainOfThought: set on predict.signature (this is what actually gets used)
                if hasattr(base_program.answerer, 'predict') and hasattr(base_program.answerer.predict, 'signature'):
                    base_program.answerer.predict.signature.instructions = warm_start_prompt
                    print("   ✅ Warm-start prompt set via answerer.predict.signature.instructions")
                    prompt_applied = True
                # Also set on extended_signature if it exists
                if hasattr(base_program.answerer, 'extended_signature'):
                    base_program.answerer.extended_signature.instructions = warm_start_prompt
                    print("   ✅ Warm-start prompt also set via answerer.extended_signature.instructions")
                    prompt_applied = True
                # Also set on signature if it exists (for completeness)
                if hasattr(base_program.answerer, 'signature'):
                    base_program.answerer.signature.instructions = warm_start_prompt
                    print("   ✅ Warm-start prompt also set via answerer.signature.instructions")
                    prompt_applied = True

                if not prompt_applied:
                    print("   ⚠️  Could not apply warm-start prompt (no signature found)")
            except Exception as e:
                print(f"   ⚠️  Failed to apply warm-start prompt: {e}")

        print("Initializing GEPA optimizer...")

        # Wrap metric with cost tracking
        def tracked_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            """Metric wrapper that tracks costs after LM calls."""
            # Get cost before tracking
            pre_cost = cost_tracker.get_summary()['total_cost']

            result = dspy_metric(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace)

            # Track cost for answerer calls (uses dspy.settings.lm)
            if pred_name == "answerer":
                cost_tracker.track("gepa_answerer", dspy.settings.lm)
            elif not pred_name:  # Program-level call
                cost_tracker.track("gepa_program", dspy.settings.lm)

            # Update daily tracker with incremental cost
            post_cost = cost_tracker.get_summary()['total_cost']
            if post_cost > pre_cost:
                daily_tracker.add_cost(post_cost - pre_cost)

            return result

        # NOTE: Domain-specific context is now in the reflection_lm's system_prompt
        # GEPA expects reflection_lm to be a DSPy LM object, not a wrapper function
        # The previous custom_reflection_lm wrapper was causing GEPA to fail to propose better prompts

        gepa_kwargs = {
            "metric": tracked_metric,  # Use cost-tracked version
            "reflection_lm": reflection_lm,  # Pass LM directly (not a wrapper function!)
            "track_stats": True,
            "log_dir": str(output_dir / "gepa_checkpoints"),
            "reflection_minibatch_size": args.reflection_minibatch_size,
            "seed": args.seed,
            "num_threads": args.num_threads,
            "max_merge_invocations": args.max_merge_invocations
        }

        # Check if GEPA was already completed (skip re-running)
        # Multiple detection methods:
        # 1. checkpoint_data has completed=True in metadata
        # 2. eval_checkpoint.json exists (means GEPA finished, eval was interrupted)
        # 3. gepa_state.bin exists (GEPA saves this when it finishes)
        # 4. gepa_results.json exists (means full pipeline completed before)
        gepa_completed = False
        gepa_state_file = output_dir / "gepa_checkpoints" / "gepa_state.bin"
        gepa_results_file = output_dir / "gepa_results.json"

        if checkpoint_data and checkpoint_data.get('metadata', {}).get('completed'):
            gepa_completed = True
            print("   (detected via checkpoint metadata)")
        elif eval_checkpoint_file.exists():
            gepa_completed = True
            print("   (detected via eval_checkpoint.json)")
        elif gepa_state_file.exists():
            gepa_completed = True
            print("   (detected via gepa_state.bin)")
        elif gepa_results_file.exists():
            gepa_completed = True
            print("   (detected via gepa_results.json)")

        if gepa_completed:
            print("\n✅ GEPA optimization already completed, loading optimized program...\n")

            # Load GEPA state from checkpoint
            gepa_state_file = output_dir / "gepa_checkpoints" / "gepa_state.bin"
            if not gepa_state_file.exists():
                print(f"❌ GEPA state file not found: {gepa_state_file}")
                print("   Cannot resume, please restart from scratch")
                teardown_logging(log_file, original_stdout, original_stderr)
                return

            with open(gepa_state_file, 'rb') as f:
                gepa_state = pickle.load(f)

            # Extract best program from GEPA state
            scores = gepa_state.get('program_full_scores_val_set', [])
            candidates = gepa_state.get('program_candidates', [])

            if not scores or not candidates:
                print("❌ Could not find candidates in GEPA state")
                teardown_logging(log_file, original_stdout, original_stderr)
                return

            best_idx = scores.index(max(scores))
            best_score = scores[best_idx]

            print(f"   📊 Loaded {len(candidates)} candidates from GEPA state")
            print(f"   🏆 Best program: #{best_idx} with score {best_score:.4f}")

            # Reconstruct the program from the candidate state
            best_candidate = candidates[best_idx]

            # If candidate is already a DSPy module, use it directly
            if hasattr(best_candidate, 'forward') and callable(best_candidate):
                optimized_program = best_candidate
                print("   Using candidate as DSPy module directly")
            else:
                # Otherwise create a fresh program and load the optimized prompt
                optimized_program = ZeroshotProgram()
                if isinstance(best_candidate, dict) and 'answerer.predict' in best_candidate:
                    optimized_prompt = best_candidate['answerer.predict']
                    print(f"   Loading optimized prompt ({len(optimized_prompt)} chars)")

                    # Set the optimized instruction on the program
                    # For ChainOfThought predictors, the prompt lives in answerer.predict.signature
                    # We need to set it on ALL relevant signature objects to ensure it's used
                    try:
                        prompt_set = False
                        # For ChainOfThought: set on predict.signature (this is what actually gets used)
                        if hasattr(optimized_program.answerer, 'predict') and hasattr(optimized_program.answerer.predict, 'signature'):
                            optimized_program.answerer.predict.signature.instructions = optimized_prompt
                            print("   ✅ Prompt set via answerer.predict.signature.instructions")
                            prompt_set = True
                        # Also set on extended_signature if it exists
                        if hasattr(optimized_program.answerer, 'extended_signature'):
                            optimized_program.answerer.extended_signature.instructions = optimized_prompt
                            print("   ✅ Prompt also set via answerer.extended_signature.instructions")
                            prompt_set = True
                        # Also set on signature if it exists (for completeness)
                        if hasattr(optimized_program.answerer, 'signature'):
                            optimized_program.answerer.signature.instructions = optimized_prompt
                            print("   ✅ Prompt also set via answerer.signature.instructions")
                            prompt_set = True

                        if not prompt_set:
                            raise AttributeError("Could not find any signature attribute")
                    except Exception as e:
                        print(f"   ⚠️  Could not set prompt: {e}")
                        print(f"   Answerer type: {type(optimized_program.answerer)}")
                        if hasattr(optimized_program.answerer, '__dict__'):
                            print(f"   Answerer attributes: {list(optimized_program.answerer.__dict__.keys())}")
                else:
                    print("   ⚠️  Candidate is dict but no 'answerer.predict' key found")
                    print(f"   Available keys: {list(best_candidate.keys()) if isinstance(best_candidate, dict) else 'N/A'}")

            # Create and attach detailed_results for downstream code
            class DetailedResults:
                def __init__(self, gepa_state, best_idx):
                    self.candidates = gepa_state.get('program_candidates', [])
                    self.val_aggregate_scores = gepa_state.get('program_full_scores_val_set', [])
                    self.train_aggregate_scores = gepa_state.get('program_full_scores_train_set', [])
                    self.best_idx = best_idx
                    self.per_val_instance_best_candidates = gepa_state.get('program_at_pareto_front_valset', [])
                    self.parents = gepa_state.get('parents', [])

            optimized_program.detailed_results = DetailedResults(gepa_state, best_idx)

            # Verify and show the loaded optimized prompt
            try:
                # Try different attribute paths to get the prompt
                if hasattr(optimized_program.answerer, 'signature') and hasattr(optimized_program.answerer.signature, 'instructions'):
                    loaded_prompt = optimized_program.answerer.signature.instructions
                elif hasattr(optimized_program.answerer, 'extended_signature') and hasattr(optimized_program.answerer.extended_signature, 'instructions'):
                    loaded_prompt = optimized_program.answerer.extended_signature.instructions
                elif hasattr(optimized_program.answerer, 'predict') and hasattr(optimized_program.answerer.predict, 'signature'):
                    loaded_prompt = optimized_program.answerer.predict.signature.instructions
                else:
                    loaded_prompt = None

                if loaded_prompt:
                    print(f"   📝 Loaded prompt preview: {loaded_prompt[:200]}...")
                else:
                    print("   ⚠️  Could not extract prompt from loaded program")
            except Exception as e:
                print(f"   ⚠️  Could not extract prompt: {e}")

            print("   ✅ Optimized program loaded successfully\n")

            # Create progress_display for resume case (won't have GEPA stats)
            progress_display = ParetoProgressDisplay(output_dir)
        else:
            # Run GEPA optimization
            if args.max_rollouts:
                gepa_kwargs["max_metric_calls"] = args.max_rollouts
                print(f"Using custom max_metric_calls: {args.max_rollouts}")
            else:
                gepa_kwargs["auto"] = args.gepa_budget

            gepa = GEPA(**gepa_kwargs)

            progress_display = ParetoProgressDisplay(output_dir)

            print("Running GEPA optimization...")
            print("💡 Press Ctrl+C to save checkpoint and exit gracefully\n")

            try:
                # Wrap GEPA compilation in network retry logic
                def compile_with_network_retry():
                    return gepa.compile(
                        student=base_program,
                        trainset=optimization_train,
                        valset=optimization_val
                    )

                optimized_program = retry_with_exponential_backoff(
                    compile_with_network_retry,
                    max_retries=3,
                    initial_delay=2.0,
                    max_delay=30.0
                )

            except Exception as e:
                # Check if this is a keyboard interrupt (propagate it immediately)
                if isinstance(e, KeyboardInterrupt):
                    raise

                # Check if this is AWS daily limit
                if isinstance(e, (RuntimeError, ValueError, KeyError)):
                    error_str = str(e)
                    is_daily_limit = (
                        "Daily token limit exceeded" in error_str or
                        "Too many tokens per day" in error_str or
                        ("daily" in error_str.lower() and "limit" in error_str.lower())
                    )

                    if is_daily_limit:
                        print("\n\n❌ AWS DAILY QUOTA EXCEEDED!")
                        cost_tracker.print_report(verbose=True)

                        print("\n💾 Saving checkpoint before exit...")
                        checkpoint_mgr.save_checkpoint(
                            gepa_optimizer=gepa,
                            iteration=-1,
                            train_examples=optimization_train,
                            val_examples=optimization_val,
                            daily_tracker=daily_tracker,
                            cost_tracker=cost_tracker,
                            metadata={'interrupted': True, 'reason': 'aws_daily_limit'}
                        )

                        print("✅ Checkpoint saved. Resume tomorrow with --resume flag")
                        print(f"   Error: {error_str}")
                        teardown_logging(log_file, original_stdout, original_stderr)
                        return

                # Re-raise unknown exceptions
                raise

            except KeyboardInterrupt:
                print("\n\n⚠️  Optimization interrupted by user!")

                # Print cost breakdown
                cost_tracker.print_report(verbose=True)

                print("\n💾 Saving checkpoint...")

                checkpoint_mgr.save_checkpoint(
                    gepa_optimizer=gepa,
                    iteration=-1,
                    train_examples=optimization_train,
                    val_examples=optimization_val,
                    daily_tracker=daily_tracker,
                    cost_tracker=cost_tracker,
                    metadata={'interrupted': True}
                )

                print("✅ Checkpoint saved. Resume with --resume flag")
                teardown_logging(log_file, original_stdout, original_stderr)
                return

        # Update progress display (only if GEPA actually ran)
        if not gepa_completed and hasattr(optimized_program, 'detailed_results'):
            dr = optimized_program.detailed_results
            if dr and HAS_GEPA_UTILS and hasattr(dr, 'candidates'):
                try:
                    pareto_front = find_dominator_programs(
                        dr.per_val_instance_best_candidates,
                        dr.val_aggregate_scores
                    )
                    best_idx = dr.best_idx if hasattr(dr, 'best_idx') else 0

                    progress_display.update(
                        iteration=len(dr.candidates),
                        candidates=dr.candidates,
                        scores=dr.val_aggregate_scores,
                        pareto_indices=pareto_front,
                        best_idx=best_idx
                    )
                except Exception as e:
                    print(f"⚠️  Could not update progress display: {e}")

        # Save final checkpoint (only if GEPA actually ran)
        if not gepa_completed:
            checkpoint_mgr.save_checkpoint(
                gepa_optimizer=gepa,
                iteration=-1,
                train_examples=optimization_train,
                val_examples=optimization_val,
                daily_tracker=daily_tracker,
                cost_tracker=cost_tracker,
                metadata={'completed': True}
            )

        # Evaluate optimized program
        if args.skip_final_eval:
            print("\n⏭️  Skipping final evaluation (--skip-final-eval)")
            avg_score = 0.0
            exact_count = 0
            partial_count = 0
            superset_count = 0
            mismatch_count = 0
            total = 0
            results = []
            gepa_stats = None
            if hasattr(optimized_program, 'detailed_results'):
                gepa_stats = extract_gepa_stats(optimized_program.detailed_results)
        else:
            # Determine evaluation examples based on mode
            if args.holdout_test:
                eval_examples = test_examples  # Use held-out test set
            else:
                eval_examples = train_examples  # Default: use ALL examples

            # Check for partial evaluation checkpoint
            eval_checkpoint_file = output_dir / "eval_checkpoint.json"
            start_idx = 0
            exact_count = 0
            partial_count = 0
            superset_count = 0
            mismatch_count = 0
            total_score = 0.0
            total = 0
            results = []

            if eval_checkpoint_file.exists():
                print("\n📂 Found evaluation checkpoint, resuming...")
                with open(eval_checkpoint_file, 'r') as f:
                    eval_checkpoint = json.load(f)
                start_idx = eval_checkpoint['completed_examples']
                exact_count = eval_checkpoint['exact_count']
                partial_count = eval_checkpoint['partial_count']
                superset_count = eval_checkpoint['superset_count']
                mismatch_count = eval_checkpoint['mismatch_count']
                total_score = eval_checkpoint['total_score']
                total = eval_checkpoint['total']
                results = eval_checkpoint['results']
                print(f"   Resuming from example {start_idx}/{len(eval_examples)}")
                print(f"   Current score: {(total_score/total*100) if total > 0 else 0:.2f}%\n")

            # Print evaluation mode
            if args.holdout_test:
                print(f"\n🧪 Evaluating on HELD-OUT TEST SET ({len(eval_examples)} examples, never seen during optimization)")
            else:
                print(f"\n📊 Evaluating on FULL DATASET ({len(eval_examples)} examples)")
            print(f"   Progress: {start_idx}/{len(eval_examples)} done")
            if args.num_threads > 1:
                print(f"   Using {args.num_threads} threads for parallel evaluation\n")
            else:
                print(f"   Using sequential evaluation (1 thread)\n")

            # Show the prompt being used for evaluation
            try:
                # Try different attribute paths to get the prompt
                eval_prompt = None
                if hasattr(optimized_program.answerer, 'signature') and hasattr(optimized_program.answerer.signature, 'instructions'):
                    eval_prompt = optimized_program.answerer.signature.instructions
                elif hasattr(optimized_program.answerer, 'extended_signature') and hasattr(optimized_program.answerer.extended_signature, 'instructions'):
                    eval_prompt = optimized_program.answerer.extended_signature.instructions
                elif hasattr(optimized_program.answerer, 'predict') and hasattr(optimized_program.answerer.predict, 'signature'):
                    eval_prompt = optimized_program.answerer.predict.signature.instructions

                if eval_prompt:
                    print(f"   📝 Evaluating with prompt: {eval_prompt[:150]}...\n")
                else:
                    print("   ⚠️  Could not extract prompt from program\n")
            except Exception as e:
                print(f"   ⚠️  Could not extract prompt: {e}\n")

            # Remaining examples to evaluate
            remaining_examples = eval_examples[start_idx:]

            # Thread-safe counters for parallel evaluation
            eval_lock = threading.Lock()

            # Define metric that collects detailed results
            def eval_metric(gold, pred, trace=None):
                """Metric for parallel evaluation that collects detailed results."""
                nonlocal exact_count, partial_count, superset_count, mismatch_count, total_score

                # Track cost (cost_tracker should be thread-safe or we accept minor inaccuracy)
                cost_tracker.track("final_eval", dspy.settings.lm)

                predicted = normalize_answer(pred.answer)
                gold_normalized = normalize_answer(gold.answer)

                match_type, score = calculate_match_type(predicted, gold_normalized)

                # Thread-safe counter updates
                with eval_lock:
                    total_score += score

                    if match_type == 'exact':
                        exact_count += 1
                    elif match_type == 'partial':
                        partial_count += 1
                    elif match_type == 'superset':
                        superset_count += 1
                    else:
                        mismatch_count += 1

                    results.append({
                        "predicted": sorted(list(predicted)),
                        "gold": sorted(list(gold_normalized)),
                        "match_type": match_type,
                        "score": score
                    })

                return score

            try:
                # Run parallel evaluation with network retry logic
                def run_evaluation_with_retry():
                    evaluator = Evaluate(
                        devset=remaining_examples,
                        metric=eval_metric,
                        num_threads=args.num_threads,
                        display_progress=True,
                        display_table=False
                    )

                    pre_cost = cost_tracker.get_summary()['total_cost']
                    avg_score = evaluator(optimized_program)
                    post_cost = cost_tracker.get_summary()['total_cost']

                    # Update daily tracker
                    if post_cost > pre_cost:
                        daily_tracker.add_cost(post_cost - pre_cost)

                    return avg_score

                # Run evaluation with automatic network error retry
                avg_score = retry_with_exponential_backoff(
                    run_evaluation_with_retry,
                    max_retries=3,
                    initial_delay=2.0,
                    max_delay=30.0
                )

                total = len(remaining_examples)

            except KeyboardInterrupt:
                print("\n\n⚠️  Evaluation interrupted by user!")
                print("💾 Saving evaluation checkpoint...")
                completed = len(results)
                eval_checkpoint = {
                    'completed_examples': start_idx + completed,
                    'exact_count': exact_count,
                    'partial_count': partial_count,
                    'superset_count': superset_count,
                    'mismatch_count': mismatch_count,
                    'total_score': total_score,
                    'total': completed,
                    'results': results
                }
                with open(eval_checkpoint_file, 'w') as f:
                    json.dump(eval_checkpoint, f, indent=2)
                cost_tracker.print_report(verbose=True)
                print(f"\n✅ Evaluation checkpoint saved: {eval_checkpoint_file}")
                print(f"   Progress: {start_idx + completed}/{len(eval_examples)}")
                print("   Run again with same arguments to resume")
                teardown_logging(log_file, original_stdout, original_stderr)
                return

            # Clean up checkpoint file after successful completion
            if eval_checkpoint_file.exists():
                eval_checkpoint_file.unlink()

            # Calculate final score: total includes all examples (resumed + new)
            # Note: When resuming, total_score accumulates from checkpoint + new examples
            # and results list contains all results, so len(results) is the true total
            total = len(results)
            avg_score = (total_score / total * 100) if total > 0 else 0.0

            gepa_stats = None
            if hasattr(optimized_program, 'detailed_results'):
                gepa_stats = extract_gepa_stats(optimized_program.detailed_results)

        # Extract the optimized prompt for saving
        try:
            best_prompt = None
            if hasattr(optimized_program.answerer, 'signature') and hasattr(optimized_program.answerer.signature, 'instructions'):
                best_prompt = optimized_program.answerer.signature.instructions
            elif hasattr(optimized_program.answerer, 'extended_signature') and hasattr(optimized_program.answerer.extended_signature, 'instructions'):
                best_prompt = optimized_program.answerer.extended_signature.instructions
            elif hasattr(optimized_program.answerer, 'predict') and hasattr(optimized_program.answerer.predict, 'signature'):
                best_prompt = optimized_program.answerer.predict.signature.instructions

            if not best_prompt:
                best_prompt = "Could not extract prompt"
        except Exception as e:
            best_prompt = f"Could not extract prompt: {e}"

        gepa_result = {
            "strategy": "gepa_optimized_FIXED",
            "avg_score": avg_score,
            "exact": exact_count,
            "partial": partial_count,
            "superset": superset_count,
            "mismatch": mismatch_count,
            "total": total,
            "results": results,
            "gepa_stats": gepa_stats,
            "best_prompt": best_prompt
        }

        # Save results
        with open(output_dir / "gepa_results.json", 'w') as f:
            json.dump(gepa_result, f, indent=2)

        print("\n✅ GEPA Optimization Complete:")
        print(f"   Avg Score: {avg_score:.2f}%")
        print(f"   Exact: {exact_count}, Partial: {partial_count}, " +
              f"Superset: {superset_count}, Mismatch: {mismatch_count}")
        print(f"\n📄 Results saved to: {output_dir / 'gepa_results.json'}\n")

        # Save optimized program
        optimized_program.save(str(output_dir / "gepa_optimized_answerer.json"))
        print(f"💾 Optimized program saved to: {output_dir / 'gepa_optimized_answerer.json'}\n")

        progress_display.print_final_summary()

        # Print detailed cost breakdown
        cost_summary = cost_tracker.print_report(verbose=True)

        # Save cost breakdown to JSON
        with open(output_dir / "cost_breakdown.json", 'w') as f:
            json.dump(cost_summary, f, indent=2)
        print(f"💰 Cost breakdown saved to: {output_dir / 'cost_breakdown.json'}\n")

        # Extract Pareto front prompts
        print("\n[PARETO FRONT] Extracting and saving all Pareto prompts...")
        try:
            dr = optimized_program.detailed_results
            if dr and HAS_GEPA_UTILS and hasattr(dr, 'candidates') and hasattr(dr, 'val_aggregate_scores'):
                pareto_front = find_dominator_programs(
                    dr.per_val_instance_best_candidates,
                    dr.val_aggregate_scores
                )

                print(f"   📊 Found {len(pareto_front)} programs in Pareto front")
                print(f"   📊 Total candidates: {len(dr.candidates)}")

                pareto_dir = output_dir / "pareto_front_prompts"
                pareto_dir.mkdir(exist_ok=True)

                pareto_info = []

                for prog_idx, candidate in enumerate(dr.candidates):
                    score = dr.val_aggregate_scores[prog_idx]
                    is_pareto = prog_idx in pareto_front
                    is_best = (prog_idx == dr.best_idx) if hasattr(dr, 'best_idx') else False

                    instructions = extract_instructions_from_candidate(candidate)

                    if instructions:
                        pareto_rank = sorted(pareto_front).index(prog_idx) if is_pareto else -1

                        if is_pareto:
                            prompt_file = pareto_dir / f"pareto_rank{pareto_rank:02d}_idx{prog_idx}_score{score:.4f}.txt"
                        else:
                            prompt_file = pareto_dir / f"candidate_idx{prog_idx}_score{score:.4f}.txt"

                        with open(prompt_file, "w", encoding="utf-8") as f:
                            f.write(f"# Program Index: {prog_idx}\n")
                            f.write(f"# In Pareto Front: {'YES' if is_pareto else 'NO'}\n")
                            if is_pareto:
                                f.write(f"# Pareto Rank: {pareto_rank}\n")
                            f.write(f"# Is Best Program: {'YES' if is_best else 'NO'}\n")
                            f.write(f"# Validation Score: {score:.4f}\n")
                            f.write(f"# {'='*60}\n\n")
                            f.write(instructions)

                        pareto_info.append({
                            "rank": pareto_rank if is_pareto else None,
                            "program_idx": prog_idx,
                            "val_score": score,
                            "is_pareto": is_pareto,
                            "is_best": is_best,
                            "prompt_file": prompt_file.name
                        })

                        status = "✅ PARETO" if is_pareto else "  "
                        best_marker = " 🏆 BEST" if is_best else ""
                        print(f"   {status} idx={prog_idx}: score={score:.4f}{best_marker}")
                    else:
                        print(f"   ⚠️ idx={prog_idx}: No instructions found (score={score:.4f})")

                with open(output_dir / "pareto_front_info.json", "w") as f:
                    json.dump(pareto_info, f, indent=2)

                print(f"\n   📁 Prompts saved to: {pareto_dir}")
                print(f"   📊 Saved {len(pareto_info)} prompts ({len(pareto_front)} in Pareto front)")
            else:
                print("   ⚠️ Could not extract Pareto front (missing gepa_utils or detailed_results)")

        except Exception as e:
            print(f"   ⚠️ Pareto front extraction failed: {e}")
            import traceback
            traceback.print_exc()

        # Generate DAG visualization
        print("\n[VISUALIZATION] Generating optimization DAG...")
        try:
            dr = optimized_program.detailed_results
            if dr and HAS_GEPA_UTILS and hasattr(dr, 'parents') and hasattr(dr, 'val_aggregate_scores'):
                pareto_front = find_dominator_programs(
                    dr.per_val_instance_best_candidates,
                    dr.val_aggregate_scores
                )

                best_idx = dr.best_idx if hasattr(dr, 'best_idx') else 0

                dot_graph = dag_to_dot(
                    dr.parents,
                    pareto_front,
                    best_idx,
                    dr.val_aggregate_scores
                )

                dot_file = output_dir / "optimization_dag.dot"
                with open(dot_file, "w") as f:
                    f.write(dot_graph)
                print(f"   ✅ Saved DAG to: {dot_file}")
                print(f"   📊 Best program: {best_idx}, Pareto front size: {len(pareto_front)}")

                try:
                    import subprocess
                    svg_file = output_dir / "optimization_dag.svg"
                    subprocess.run(
                        ["dot", "-Tsvg", str(dot_file), "-o", str(svg_file)],
                        check=True,
                        capture_output=True
                    )
                    print(f"   ✅ Also saved as SVG: {svg_file}")
                except FileNotFoundError:
                    print("   ℹ️  Graphviz not installed. To generate SVG: sudo apt install graphviz")
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️ Graphviz SVG generation failed: {e}")
            else:
                print("   ⚠️ Could not generate DAG (missing parents or scores in detailed_results)")

        except (FileNotFoundError, subprocess.CalledProcessError, AttributeError) as e:
            print(f"   ⚠️ DAG visualization failed: {e}")
            import traceback
            traceback.print_exc()

    finally:
        teardown_logging(log_file, original_stdout, original_stderr)
        print("✅ Log file saved")


if __name__ == "__main__":
    main()
