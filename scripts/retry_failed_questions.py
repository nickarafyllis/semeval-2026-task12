#!/usr/bin/env python3
"""
Retry failed questions from a previous experiment

This script identifies failed questions (empty predictions, missing analyses/thinkings)
and retries them, then merges the results back into a complete experiment.
"""

import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.aws_config import get_bedrock_client, get_model_id
from src.data.loader import load_dev_data
from src.experiments.manager import list_experiments, load_experiment_results
from src.experiments.retry import retry_failed_questions
from src.prompts.prompt_templates import get_template

# LLM clients
from src.models.llm_clients import (
    ChatClaude, ChatClaudeThinking,
    ChatLlama, ChatDeepSeek, ChatDeepSeekV31,
    ChatGemini, ChatGeminiCached
)

# Inference functions
from src.inference.claude import run_claude_inference
from src.inference.llama import run_llama_inference
from src.inference.deepseek import run_deepseek_inference
from src.inference.gemini import run_gemini_inference



def build_client(model_family: str, version: str, system_prompt: str, claude_mode: str = "optimized",
                 gemini_mode: str = "cached"):
    """Build appropriate LLM client with auto-detected region"""

    # Handle Gemini separately (uses Google API, not AWS Bedrock)
    if model_family == "gemini":
        from configs.google_config import get_gemini_client, get_gemini_model_id

        print("   → Using Google Gemini API")
        gemini_client = get_gemini_client()
        model_id = get_gemini_model_id(version)
        print(f"   Model: {model_id}")

        if gemini_mode == "cached":
            print("   → ChatGeminiCached client (context caching enabled)")
            return ChatGeminiCached(model_id, gemini_client, system_prompt)
        else:
            print("   → ChatGemini client (no caching)")
            return ChatGemini(model_id, gemini_client, system_prompt, use_caching=False)

    # For AWS Bedrock models, import AWS dependencies
    from configs.aws_config import get_bedrock_client, get_model_id

    # Auto-detect region based on model
    if model_family == "deepseek" and ("v3" in version.lower() or "v31" in version.lower()):
        region = "us-west-2"  # V3.1 only available here
        print(f"   Using region: {region} (for DeepSeek V3.1)")
    else:
        region = "us-east-1"  # Default for all others

    client = get_bedrock_client(region=region)
    model_id = get_model_id(version)

    if model_family == "claude":
        if claude_mode == "thinking":
            print("   → ChatClaudeThinking client")
            return ChatClaudeThinking(model_id, client, system_prompt)
        else:
            print("   → ChatClaude client")
            return ChatClaude(model_id, client, system_prompt)

    if model_family == "llama":
        print("   → ChatLlama client")
        return ChatLlama(model_id, client, system_prompt)

    if model_family == "deepseek":
        if "v3.1" in version.lower() or "v31" in version.lower() or "v3" in version.lower():
            print("   → ChatDeepSeekV31 client")
            return ChatDeepSeekV31(model_id, client, system_prompt)
        else:
            print("   → ChatDeepSeek client (R1)")
            return ChatDeepSeek(model_id, client, system_prompt)

    raise ValueError(f"Unknown model_family '{model_family}'")


def get_inference_function(model_family: str, version: str = None,
                          claude_mode: str = "optimized", reasoning_effort: str = "high",
                          gemini_mode: str = "cached"):
    """Get appropriate inference function with parameters"""
    if model_family == "claude":
        return lambda c, q, d, sleep_seconds=1: run_claude_inference(c, q, d, mode=claude_mode, sleep_seconds=sleep_seconds)
    elif model_family == "llama":
        return lambda c, q, d, sleep_seconds=1: run_llama_inference(c, q, d, sleep_seconds=sleep_seconds)
    elif model_family == "gemini":
        return lambda c, q, d, sleep_seconds=1: run_gemini_inference(c, q, d, mode=gemini_mode, sleep_seconds=sleep_seconds)
    else:  # deepseek
        return lambda c, q, d, sleep_seconds=1: run_deepseek_inference(
            c, q, d,
            version=version or "v3.1",
            sleep_seconds=sleep_seconds,
            reasoning_effort=reasoning_effort
        )


def main():
    parser = argparse.ArgumentParser(description="Retry failed questions from experiment")
    parser.add_argument("--pattern", type=str,
                        help="Experiment name pattern (e.g., 'claude-4-sonnet_20250113')")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    parser.add_argument("--data-dir", type=str, default="data/raw/semeval2026-task12-dataset")
    parser.add_argument("--model-family", type=str,
                        choices=["claude", "llama", "deepseek", "gemini"],
                        help="Model family to use for retry (defaults to original)")
    parser.add_argument("--version", type=str,
                        help="Model version (defaults to original)")
    parser.add_argument("--claude-mode", type=str, default="optimized",
                        choices=["simple", "optimized", "thinking"])
    parser.add_argument("--reasoning-effort", type=str, default="high",
                        choices=["off", "low", "medium", "high"],
                        help="Reasoning level for DeepSeek V3.1 (default: high)")
    parser.add_argument("--gemini-mode", type=str, default="cached",
                        choices=["simple", "cached"],
                        help="Gemini mode: 'simple' (no caching) or 'cached' (with context caching, default)")
    parser.add_argument("--sleep", type=int, default=1)
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Don't create dashboard after merge")
    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n📋 Available Experiments:")
        print("="*80)
        df = list_experiments()
        if not df.empty:
            print(df[['experiment_name', 'model_name', 'score', 'num_questions', 'timestamp']].to_string(index=False))
            print("\n💡 Use --pattern <name_pattern> to retry failed questions")
        else:
            print("No experiments found")
        return

    # Check that pattern is provided if not listing
    if not args.pattern:
        parser.error("--pattern is required (unless using --list)")

    # Load data
    print("📦 Loading data...")
    _, docs = load_dev_data(Path(args.data_dir))

    # Load original experiment to get model info if not specified
    exps = list_experiments(as_dataframe=True)
    matching = exps[exps["experiment_name"].str.contains(args.pattern, case=False, na=False)]

    if matching.empty:
        print(f"❌ No experiment found matching: {args.pattern}")
        print("\n💡 Available experiments:")
        print(exps["experiment_name"].to_string(index=False))
        return

    exp_path = matching.iloc[0]["path"]
    loaded = load_experiment_results(exp_path)
    original_metadata = loaded["metadata"]

    # Determine model settings (use args or fall back to original)
    model_family = args.model_family
    version = args.version

    if not model_family:
        # Infer from model name
        model_name = original_metadata["model_name"]
        if "claude" in model_name.lower():
            model_family = "claude"
            version = version or model_name
        elif "llama" in model_name.lower():
            model_family = "llama"
            version = version or model_name
        elif "deepseek" in model_name.lower():
            model_family = "deepseek"
            version = version or model_name
        elif "gemini" in model_name.lower():
            model_family = "gemini"
            version = version or model_name
        else:
            print(f"❌ Cannot infer model family from: {model_name}")
            print("   Please specify --model-family and --version")
            return

    if not version:
        version = original_metadata["model_name"]

        # Get system prompt from metadata
    prompt_name = original_metadata.get("prompt", "")
    system_prompt = original_metadata.get("system_prompt", "")  # Check if already saved

    if not system_prompt and prompt_name:
        # Load from template
        try:
            system_prompt = get_template(prompt_name)
            print(f"   ✓ Loaded system prompt from template: '{prompt_name}' ({len(system_prompt)} chars)")
        except (ImportError, ValueError) as e:
            print(f"   ⚠️  Could not load template '{prompt_name}': {e}")

    if not system_prompt:
        # Final fallback to hardcoded default
        # system_prompt = (
        #     "You are an expert in causal and abductive reasoning. "
        #     "Analyze the provided context carefully and determine which options "
        #     "are plausible causes of the target event. Provide concise justifications."
        # )
        print("   ⚠️  No system prompt in metadata")

    print("\n🔧 Retry Configuration:")
    print(f"   Model family: {model_family}")
    print(f"   Version: {version}")
    if model_family == "claude":
        print(f"   Claude mode: {args.claude_mode}")
    elif model_family == "gemini":
        print(f"   Gemini mode: {args.gemini_mode}")
    elif model_family == "deepseek":
        print(f"   Reasoning effort: {args.reasoning_effort}")

    # Build client and inference function
    print(f"\n🤖 Creating {model_family} client...")
    chat = build_client(model_family, version, system_prompt, args.claude_mode, args.gemini_mode)
    inference_fn = get_inference_function(
        model_family,
        version,
        args.claude_mode,
        args.reasoning_effort,
        args.gemini_mode
    )

    # Run retry
    print("\n🔄 Starting retry process...")
    print("="*80)

    result = retry_failed_questions(
        experiment_name_pattern=args.pattern,
        chat_client=chat,
        inference_function=inference_fn,
        docs=docs,
        sleep_seconds=args.sleep,
        create_dashboard_after=not args.no_dashboard
    )

    # Print summary
    if result.get("success"):
        print(f"\n{'='*80}")
        print("📊 RETRY SUMMARY")
        print(f"{'='*80}")
        
        # Check if there were actually failed questions to retry
        if result.get('failed_count', 0) == 0:
            print("✅ No failed questions found in experiment!")
            print(f"   Experiment: {result.get('experiment_name', args.pattern)}")
            # Safe score handling
            score = result.get('score', result.get('original_score', 0.0))
            if isinstance(score, (int, float)):
                print(f"   Score: {score:.4f}")
            else:
                print("   Score: N/A")

            print(f"   Total questions: {result.get('num_questions', 'N/A')}")
            print("\n💡 All questions already have valid predictions and analyses.")
        else:
            # Normal retry summary (with failed questions)
            print(f"Original experiment: {result.get('original_experiment', 'N/A')}")
            print(f"Merged experiment:   {result.get('merged_experiment', 'N/A')}")
            print(f"Failed questions:    {result.get('failed_count', 0)}")
            print(f"Original score:      {result.get('original_score', 0.0):.4f}")
            print(f"Final score:         {result.get('final_score', 0.0):.4f}")
            
            # Show improvement if available
            if 'improvement' in result and isinstance(result['improvement'], (int, float)):
                print(f"Improvement:         {result['improvement']:+.4f}")
            
            print(f"\nSaved to: {result.get('save_path', 'N/A')}")
            
            if result.get('dashboard_path'):
                print(f"Dashboard: {result['dashboard_path']}")
    else:
        print(f"\n❌ Retry failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
