"""
Cost tracking utilities for AWS Bedrock and Google Gemini API calls.

Tracks token usage and calculates costs based on model pricing.
Provides centralized cost calculation and tracking with support for:
- Multiple model pricing (Claude, Llama, DeepSeek, Kimi, MiniMax, Gemini)
- Prompt caching (cache writes and reads)
- Per-stage cost breakdown
- Thread-safe tracking for parallel operations
- Automatic model detection from LM objects for multi-model pipelines
"""

import threading
from typing import Dict, List, Any
from collections import defaultdict


class CostEstimator:
    """
    Calculates API call costs based on token counts.
    Includes pricing for all supported Bedrock models including cache rates.
    Model names aligned with configs/aws_config.py MODEL_IDS
    """

    def __init__(self, model_name: str = "claude-sonnet-4.5"):
        self.model_name = model_name
        self._fallback_warned = False

        # Pricing (as of December 2025) - AWS Bedrock rates
        # Format: input, output, cache_write, cache_read (per million tokens)
        # Model names aligned with configs/aws_config.py MODEL_IDS
        self.pricing = {
            # Claude models (aligned with aws_config.py)
            "claude-sonnet-4.0": {
                "input": 3.00,  # $0.003 per 1K = $3.00 per 1M
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30
            },
            "claude-sonnet-4.5": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30
            },
            "claude-sonnet-4.5-1m": {  # 1M context version (2x input, 1.5x output for >200K tokens)
                "input": 6.00,  # $6.00 per 1M tokens (2x standard rate)
                "output": 22.50,  # $22.50 per 1M tokens (1.5x standard rate)
                "cache_write": 7.50,  # 2x standard cache write rate
                "cache_read": 0.60  # 2x standard cache read rate
            },
            "claude-opus-4.0": {
                "input": 15.00,
                "output": 75.00,
                "cache_write": 18.75,
                "cache_read": 1.50
            },
            "claude-opus-4.1": {
                "input": 15.00,
                "output": 75.00,
                "cache_write": 18.75,
                "cache_read": 1.50
            },
            "claude-opus-4.5": {
                "input": 5.00,  # $0.005 per 1K = $5.00 per 1M
                "output": 25.00,
                "cache_write": 6.25,
                "cache_read": 0.50
            },
            "claude-haiku-3.5": {
                "input": 0.80,
                "output": 4.00,
                "cache_write": 1.00,
                "cache_read": 0.08
            },
            "claude-haiku-4.5": {
                "input": 1.00,  # $0.001 per 1K = $1.00 per 1M
                "output": 5.00,
                "cache_write": 1.25,
                "cache_read": 0.10
            },
            # Llama models
            "llama-3.3-70b": {
                "input": 0.99,
                "output": 0.99,
                "cache_write": 0.99,
                "cache_read": 0.099
            },
            # DeepSeek models
            "deepseek-r1": {
                "input": 1.35,  # $0.00135 per 1K = $1.35 per 1M
                "output": 5.40,  # $0.0054 per 1K = $5.40 per 1M
                "cache_write": 1.35,
                "cache_read": 0.135
            },
            "deepseek-v3.1": {
                "input": 0.58,  # $0.00058 per 1K = $0.58 per 1M
                "output": 1.68,  # $0.00168 per 1K = $1.68 per 1M
                "cache_write": 0.58,
                "cache_read": 0.058
            },
            # Kimi models
            "kimi-k2-thinking": {
                "input": 0.60,  # $0.00060 per 1K = $0.60 per 1M
                "output": 2.50,  # $0.00250 per 1K = $2.50 per 1M
                "cache_write": 0.0,  # Kimi doesn't support caching
                "cache_read": 0.0
            },
            # MiniMax models
            "minimax-m2": {
                "input": 0.10,  # Estimated $0.0001 per 1K = $0.10 per 1M
                "output": 0.10,
                "cache_write": 0.0,  # MiniMax doesn't support caching
                "cache_read": 0.0
            },
            # Gemini 3 Flash Preview (Google AI Studio pricing - Jan 2025)
            # Source: https://ai.google.dev/pricing
            # Standard pricing (text/image/video)
            "gemini-3-flash-preview": {
                "input": 0.50,  # $0.50 per 1M tokens (text/image/video)
                "output": 3.00,  # $3.00 per 1M tokens (including thinking tokens)
                "cache_write": 0.50,  # Same as input
                "cache_read": 0.05  # $0.05 per 1M tokens (context caching)
            },
            # Gemini 3 Flash Preview - Batch API (50% discount)
            "gemini-3-flash-preview-batch": {
                "input": 0.25,  # $0.25 per 1M tokens (50% off)
                "output": 1.50,  # $1.50 per 1M tokens (50% off)
                "cache_write": 0.25,  # Same as input
                "cache_read": 0.025  # $0.025 per 1M tokens (50% off)
            },
            # Gemini 3 Pro Preview - PLACEHOLDER (update with actual pricing when available)
            "gemini-3-pro-preview": {
                "input": 5.00,  # PLACEHOLDER - update with actual pricing
                "output": 15.00,  # PLACEHOLDER - update with actual pricing
                "cache_write": 5.00,  # PLACEHOLDER
                "cache_read": 0.50  # PLACEHOLDER
            },
            # Gemini 3 Pro Preview - Batch API (50% discount) - PLACEHOLDER
            "gemini-3-pro-preview-batch": {
                "input": 2.50,  # PLACEHOLDER - update with actual pricing
                "output": 7.50,  # PLACEHOLDER - update with actual pricing
                "cache_write": 2.50,  # PLACEHOLDER
                "cache_read": 0.25  # PLACEHOLDER
            },
            # OpenAI GPT 5.2 (estimated pricing - update with actual when available)
            "gpt-5.2": {
                "input": 5.00,  # Estimated $5.00 per 1M tokens
                "output": 15.00,  # Estimated $15.00 per 1M tokens
                "cache_write": 5.00,  # Same as input (OpenAI auto-caches)
                "cache_read": 1.25  # 75% discount on cached tokens
            },
            # OpenAI GPT 5 (estimated pricing)
            "gpt-5": {
                "input": 10.00,  # Estimated
                "output": 30.00,
                "cache_write": 10.00,
                "cache_read": 2.50
            },
            # OpenAI GPT 4.1
            "gpt-4.1": {
                "input": 2.00,
                "output": 8.00,
                "cache_write": 2.00,
                "cache_read": 0.50
            },
            # OpenAI GPT-4o
            "gpt-4o": {
                "input": 2.50,
                "output": 10.00,
                "cache_write": 2.50,
                "cache_read": 0.625
            },
            # OpenAI GPT-4o-mini
            "gpt-4o-mini": {
                "input": 0.15,
                "output": 0.60,
                "cache_write": 0.15,
                "cache_read": 0.0375
            },
            # OpenAI o1
            "o1": {
                "input": 15.00,
                "output": 60.00,
                "cache_write": 15.00,
                "cache_read": 3.75
            },
            # OpenAI o1-mini
            "o1-mini": {
                "input": 1.10,
                "output": 4.40,
                "cache_write": 1.10,
                "cache_read": 0.55
            },
            # OpenAI o3
            "o3": {
                "input": 10.00,  # Estimated
                "output": 40.00,
                "cache_write": 10.00,
                "cache_read": 2.50
            },
            # OpenAI o3-mini
            "o3-mini": {
                "input": 1.10,
                "output": 4.40,
                "cache_write": 1.10,
                "cache_read": 0.275
            },
        }

    def calculate_cost(self,
                      input_tokens: int = 0,
                      output_tokens: int = 0,
                      cache_write_tokens: int = 0,
                      cache_read_tokens: int = 0) -> Dict[str, float]:
        """
        Calculate cost for a single API call.

        Args:
            input_tokens: Total input tokens (includes cache tokens)
            output_tokens: Output tokens
            cache_write_tokens: Tokens written to cache
            cache_read_tokens: Tokens read from cache

        Returns:
            Dict with breakdown: input_cost, output_cost, cache_write_cost,
            cache_read_cost, total_cost
        """
        if self.model_name in self.pricing:
            prices = self.pricing[self.model_name]
        else:
            prices = self.pricing["claude-sonnet-4.5"]
            if not self._fallback_warned:
                print(f"[CostEstimator] Unknown model '{self.model_name}', using claude-sonnet-4.5 pricing as fallback")
                self._fallback_warned = True

        # Input tokens from API usually include cache tokens.
        # We must subtract them to avoid double counting.
        regular_input = max(0, input_tokens - cache_write_tokens - cache_read_tokens)

        # Prices are per million tokens, so divide token counts by 1M
        input_cost = (regular_input / 1_000_000) * prices['input']
        output_cost = (output_tokens / 1_000_000) * prices['output']
        cache_write_cost = (cache_write_tokens / 1_000_000) * prices['cache_write']
        cache_read_cost = (cache_read_tokens / 1_000_000) * prices['cache_read']

        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'cache_write_cost': cache_write_cost,
            'cache_read_cost': cache_read_cost,
            'total_cost': input_cost + output_cost + cache_write_cost + cache_read_cost
        }


class CostTracker:
    """
    Enhanced cost tracking with per-stage breakdown and thread-safe operations.
    Compatible with scripts that use track(stage_name, lm) pattern.
    """

    def __init__(self, model_name: str = "claude-sonnet-4.5"):
        self.estimator = CostEstimator(model_name)
        self.stages = defaultdict(lambda: {
            'input_tokens': 0,
            'output_tokens': 0,
            'cache_write_tokens': 0,
            'cache_read_tokens': 0,
            'total_cost': 0.0,
            'calls': 0
        })
        self.lock = threading.Lock()

        # Legacy support for simple tracking
        self.calls: List[Dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cost = 0.0

    def track(self, stage_name: str, lm):
        """
        Track cost for a specific stage (used by optimize_prompts.py).

        Args:
            stage_name: Name of the processing stage (e.g., "extraction", "expert_temporal")
            lm: Language model instance with last_usage attribute
        """
        if not hasattr(lm, 'last_usage'):
            return

        with self.lock:
            usage = lm.last_usage
            input_tokens = usage.get('input_tokens', 0) or 0
            output_tokens = usage.get('output_tokens', 0) or 0
            cache_write = usage.get('cache_creation_input_tokens', 0) or 0
            cache_read = usage.get('cache_read_input_tokens', 0) or 0

            if input_tokens == 0 and output_tokens == 0:
                return

            # Detect model from lm object for accurate pricing
            # If this is a different model than the tracker's default, use its pricing
            estimator_to_use = self.estimator
            if hasattr(lm, 'model_id'):
                # Gemini models have model_id attribute (e.g., "gemini-3-pro-preview")
                lm_model = lm.model_id
                if lm_model != self.estimator.model_name and lm_model in self.estimator.pricing:
                    estimator_to_use = CostEstimator(lm_model)
            elif hasattr(lm, 'model_name'):
                # Extract model name from DSPy LM (e.g., "gemini.gemini-3-pro-preview" -> "gemini-3-pro-preview")
                lm_model_name = lm.model_name
                # Handle bedrock. prefix (e.g., "bedrock.claude-sonnet-4.5" -> "claude-sonnet-4.5")
                if '.' in lm_model_name:
                    lm_model = lm_model_name.split('.', 1)[1]
                else:
                    lm_model = lm_model_name
                if lm_model != self.estimator.model_name and lm_model in self.estimator.pricing:
                    estimator_to_use = CostEstimator(lm_model)

            cost = estimator_to_use.calculate_cost(input_tokens, output_tokens, cache_write, cache_read)

            stage = self.stages[stage_name]
            stage['input_tokens'] += input_tokens
            stage['output_tokens'] += output_tokens
            stage['cache_write_tokens'] += cache_write
            stage['cache_read_tokens'] += cache_read
            stage['total_cost'] += cost['total_cost']
            stage['calls'] += 1

            # Clear usage (preserve all fields including cache tokens)
            lm.last_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }

    def add_call(self,
                 input_tokens: int = 0,
                 output_tokens: int = 0,
                 cache_creation_tokens: int = 0,
                 cache_read_tokens: int = 0,
                 metadata: Dict = None):
        """
        Record a single API call with its token usage (legacy interface).

        Args:
            input_tokens: Regular input tokens
            output_tokens: Output tokens
            cache_creation_tokens: Tokens written to cache
            cache_read_tokens: Tokens read from cache
            metadata: Optional metadata (e.g., question_id, stage)
        """
        cost = self.estimator.calculate_cost(
            input_tokens, output_tokens,
            cache_creation_tokens, cache_read_tokens
        )

        call_record = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cache_creation_tokens': cache_creation_tokens,
            'cache_read_tokens': cache_read_tokens,
            **cost,
            'metadata': metadata or {}
        }

        self.calls.append(call_record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_creation_tokens += cache_creation_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_cost += cost['total_cost']

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated cost summary across all stages."""
        with self.lock:
            # Use stage-based tracking if available, otherwise use legacy tracking
            if self.stages:
                total_cost = sum(s['total_cost'] for s in self.stages.values())
                total_input = sum(s['input_tokens'] for s in self.stages.values())
                total_output = sum(s['output_tokens'] for s in self.stages.values())
                total_cache_write = sum(s['cache_write_tokens'] for s in self.stages.values())
                total_cache_read = sum(s['cache_read_tokens'] for s in self.stages.values())

                # Order stages by pipeline execution order
                # Numbered prefixes (1-11) make execution order immediately clear
                stage_order = [
                    '1_extraction',
                    '2_expert_temporal',
                    '3_expert_discourse', 
                    '4_expert_precondition',
                    '5_expert_commonsense',
                    '6_discussion_temporal',
                    '7_discussion_discourse',
                    '8_discussion_precondition',
                    '9_discussion_commonsense',
                    '10_judge',
                    '11_answer_generation'
                ]
                
                # Build ordered stages dict - include known stages in order, then any unknown stages
                ordered_stages = {}
                for stage_name in stage_order:
                    if stage_name in self.stages:
                        ordered_stages[stage_name] = self.stages[stage_name]
                
                # Add any stages not in the predefined order (for extensibility)
                for stage_name in sorted(self.stages.keys()):
                    if stage_name not in ordered_stages:
                        ordered_stages[stage_name] = self.stages[stage_name]

                return {
                    'total_cost': total_cost,
                    'total_input_tokens': total_input,
                    'total_output_tokens': total_output,
                    'total_cache_write_tokens': total_cache_write,
                    'total_cache_read_tokens': total_cache_read,
                    'total_tokens': total_input + total_output,
                    'stages': ordered_stages
                }
            else:
                # Legacy format
                return {
                    'total_calls': len(self.calls),
                    'total_input_tokens': self.total_input_tokens,
                    'total_output_tokens': self.total_output_tokens,
                    'total_cache_creation_tokens': self.total_cache_creation_tokens,
                    'total_cache_read_tokens': self.total_cache_read_tokens,
                    'total_tokens': (self.total_input_tokens + self.total_output_tokens +
                                   self.total_cache_creation_tokens + self.total_cache_read_tokens),
                    'total_cost': self.total_cost,
                    'avg_cost_per_call': self.total_cost / len(self.calls) if self.calls else 0
                }

    def print_report(self, verbose: bool = False):
        """
        Print detailed cost breakdown with cache statistics.

        Args:
            verbose: If True, print per-stage breakdown. If False, print only total.

        Returns:
            Cost summary dictionary
        """
        summary = self.get_summary()

        if verbose and 'stages' in summary:
            # New per-stage reporting
            print(f"\n{'='*70}")
            print("COST BREAKDOWN BY STAGE")
            print(f"{'='*70}")

            # Stages are already ordered in get_summary(), no need to sort
            for stage_name, stage_data in summary['stages'].items():
                calls = f"({stage_data['calls']}x)" if stage_data['calls'] > 1 else ""

                # Build cache info string if any cache tokens exist
                cache_info = ""
                cw = stage_data['cache_write_tokens']
                cr = stage_data['cache_read_tokens']
                if cw > 0 or cr > 0:
                    cache_parts = []
                    if cw > 0:
                        cache_parts.append(f"CW:{cw:,}")
                    if cr > 0:
                        cache_parts.append(f"CR:{cr:,}")
                    cache_info = f" | {' '.join(cache_parts)}"

                print(f"  {stage_name:<30} {calls:<6} | In: {stage_data['input_tokens']:>6,} | Out: {stage_data['output_tokens']:>6,}{cache_info} | ${stage_data['total_cost']:.5f}")

            print("-" * 70)

            # Always show cache summary in verbose mode
            total_cw = summary['total_cache_write_tokens']
            total_cr = summary['total_cache_read_tokens']
            if total_cw > 0 or total_cr > 0:
                # Calculate cache hit rate (cache reads vs total tokens processed)
                total_tokens_processed = summary['total_input_tokens'] + total_cr
                cache_hit_rate = (total_cr / (total_tokens_processed + 0.001)) * 100
                print(f"  💾 CACHE: Writes={total_cw:,} | Reads={total_cr:,} ({cache_hit_rate:.1f}% cache hit rate)")
            else:
                # Model-specific threshold messages
                if 'gemini' in self.estimator.model_name.lower():
                    threshold_msg = "context below 32,768 token threshold (Gemini requirement)"
                else:
                    threshold_msg = "context may be below minimum caching threshold"
                print(f"  💾 CACHE: No cache hits ({threshold_msg})")

            print(f"  {'TOTAL':<30} {'':6} | ${summary['total_cost']:.5f}")
            print(f"{'='*70}\n")
        else:
            # Simple summary
            cache_info = ""
            if summary.get('total_cache_read_tokens', 0) > 0:
                cache_info = f" (CR:{summary['total_cache_read_tokens']:,})"
            print(f"💰 Cost: ${summary['total_cost']:.5f}{cache_info}")

        return summary

    def print_summary(self):
        """Print formatted cost summary (legacy interface)."""
        summary = self.get_summary()

        print(f"\n{'='*80}")
        print("COST SUMMARY")
        print(f"{'='*80}")

        if 'total_calls' in summary:
            print(f"Total API Calls:       {summary['total_calls']:,}")

        print(f"Input Tokens:          {summary['total_input_tokens']:,}")
        print(f"Output Tokens:         {summary['total_output_tokens']:,}")

        cache_creation = summary.get('total_cache_creation_tokens') or summary.get('total_cache_write_tokens', 0)
        cache_read = summary.get('total_cache_read_tokens', 0)

        if cache_creation > 0:
            print(f"Cache Write Tokens:    {cache_creation:,}")
        if cache_read > 0:
            print(f"Cache Read Tokens:     {cache_read:,}")
            # Calculate savings: (cache_read_tokens / 1M) * (input_price - cache_read_price)
            # Use fallback pricing if model not found
            model_name = self.estimator.model_name
            if model_name in self.estimator.pricing:
                prices = self.estimator.pricing[model_name]
            else:
                prices = self.estimator.pricing["claude-sonnet-4.5"]
                print(f"[CostTracker] Unknown model '{model_name}', using claude-sonnet-4.5 pricing as fallback")
            cache_savings = (cache_read / 1_000_000) * \
                           (prices['input'] - prices['cache_read'])
            print(f"Cache Savings:         ${cache_savings:.4f}")

        print(f"Total Tokens:          {summary['total_tokens']:,}")
        print(f"Total Cost:            ${summary['total_cost']:.4f}")

        if 'avg_cost_per_call' in summary and summary.get('total_calls', 0) > 1:
            print(f"Avg Cost/Call:         ${summary['avg_cost_per_call']:.4f}")

        print(f"{'='*80}")
