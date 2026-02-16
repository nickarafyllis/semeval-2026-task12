#!/usr/bin/env python3
"""Run abductive event reasoning experiment with automatic result saving"""
import sys
from pathlib import Path
import argparse
import time
import json
from multiprocessing import cpu_count
import gc

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts.prompt_templates import get_template
from src.data.loader import load_dev_data, load_train_data, load_test_data
from src.evaluation.metrics import evaluate
from src.evaluation.analysis import analyze_predictions
from src.experiments.manager import finalize_experiment, load_experiment_results
from src.experiments.dashboard import create_dashboard
from src.experiments.retry import identify_failed_questions
from src.retrieval.graph_rag_utils import load_graph_rag_data
from src.utils.submission import create_submission_file, load_predictions_from_results


# LLM clients
from src.models.llm_clients import (
    ChatClaude, ChatClaudeThinking, ChatClaudeUncached,
    ChatLlama, ChatDeepSeek, ChatDeepSeekV31, ChatKimi,
    ChatGemini, ChatGeminiCached,
    ChatOpenAI, ChatOpenAICached
)

# Inference engines
from src.inference.claude import run_claude_inference
from src.inference.llama import run_llama_inference
from src.inference.deepseek import run_deepseek_inference
from src.inference.kimi import run_kimi_inference
from src.inference.gemini import run_gemini_inference
from src.inference.openai import run_openai_inference, run_openai_sc_inference

def build_client(model_family: str, version: str, system_prompt: str, claude_mode: str = "optimized",
                 gemini_mode: str = "cached", openai_mode: str = "cached", temperature: float = None,
                 thinking_level: str = None, reasoning_effort: str = None, key_index: int = None):
    """Build appropriate LLM client with auto-detected region."""

    # Handle OpenAI separately (uses OpenAI API)
    if model_family == "openai":
        from configs.openai_config import get_openai_client, get_openai_model_id

        print("   → Using OpenAI API")
        openai_client = get_openai_client(key_index=key_index)
        model_id = get_openai_model_id(version)
        print(f"   Model: {model_id}")

        # Set default reasoning effort to "xhigh" for GPT-5.2 if not specified
        if reasoning_effort is None and "gpt-5" in version.lower():
            reasoning_effort = "xhigh"
            print("   → Auto-enabling xhigh reasoning effort for GPT-5.2")

        # Display configuration
        print(f"   Temperature: {temperature}")
        if reasoning_effort:
            print(f"   Reasoning effort: {reasoning_effort}")

        if openai_mode == "cached":
            print("   → ChatOpenAICached client (automatic caching enabled)")
            return ChatOpenAICached(model_id, openai_client, system_prompt,
                                   temperature=temperature, reasoning_effort=reasoning_effort)
        else:
            print("   → ChatOpenAI client (no caching)")
            return ChatOpenAI(model_id, openai_client, system_prompt,
                            temperature=temperature, reasoning_effort=reasoning_effort)

    # Handle Gemini separately (uses Google API, not AWS Bedrock)
    if model_family == "gemini":
        from configs.google_config import get_gemini_client, get_gemini_model_id

        print("   → Using Google Gemini API")
        gemini_client = get_gemini_client(key_index=key_index)
        model_id = get_gemini_model_id(version)
        print(f"   Model: {model_id}")

        # Handle "off" option for thinking level (convert to None)
        actual_thinking_level = None if thinking_level == "off" else thinking_level

        # Display configuration
        print(f"   Temperature: {temperature}")
        if actual_thinking_level:
            print(f"   Thinking level: {actual_thinking_level}")
        else:
            print("   Thinking: disabled")

        if gemini_mode == "cached":
            print("   → ChatGeminiCached client (context caching enabled)")
            return ChatGeminiCached(model_id, gemini_client, system_prompt,
                                   temperature=temperature, thinking_level=actual_thinking_level)
        else:
            print("   → ChatGemini client (no caching)")
            return ChatGemini(model_id, gemini_client, system_prompt, use_caching=False,
                            temperature=temperature, thinking_level=actual_thinking_level)

    # For AWS Bedrock models, import AWS dependencies
    from configs.aws_config import get_bedrock_client, get_model_id

    # Auto-detect region based on model (for AWS Bedrock models)
    if model_family == "deepseek" and ("v3" in version.lower() or "v31" in version.lower()):
        region = "us-west-2"  # V3.1 only available here
    else:
        region = "us-east-1"  # Default for all others

    print(f"   Region: {region}")

    # Use get_bedrock_client() which has credentials configured
    client = get_bedrock_client(region=region)
    model_id = get_model_id(version)

    if model_family == "claude":
        if claude_mode == "thinking":
            print("   → ChatClaudeThinking client")
            return ChatClaudeThinking(model_id, client, system_prompt)
        elif "uncached" in version.lower():
            print("   → ChatClaudeUncached client")
            return ChatClaudeUncached(model_id, client, system_prompt)
        else:
            print("   → ChatClaude client")
            return ChatClaude(model_id, client, system_prompt)

    if model_family == "llama":
        return ChatLlama(model_id, client, system_prompt)

    if model_family == "deepseek":
        if "v3.1" in version.lower() or "v31" in version.lower() or "v3" in version.lower():
            return ChatDeepSeekV31(model_id, client, system_prompt)
        else:
            return ChatDeepSeek(model_id, client, system_prompt)

    if model_family == "kimi":
        print("   → ChatKimi client")
        return ChatKimi(model_id, client, system_prompt)

    raise ValueError(f"Unknown model_family '{model_family}'")


def main():
    parser = argparse.ArgumentParser(description="Run abductive event reasoning experiment")
    parser.add_argument("--data-dir", type=str, default="data/raw/semeval2026-task12-dataset")
    parser.add_argument("--limit", type=int, default=400, help="Number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Number of questions to skip (offset for batch processing)")
    parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "val", "test"],
                        help="Dataset to run on: 'dev' (default), 'val' (last N of train), or 'test'")
    parser.add_argument("--topic-split", type=str, default=None,
                        help="Split topics across parallel runs: 'N/M' (e.g., '1/4' for part 1 of 4). Maximizes cache efficiency when using multiple API keys.")
    parser.add_argument("--model-family", type=str, required=True, choices=["claude", "llama", "deepseek", "kimi", "gemini", "openai"])
    parser.add_argument("--version", type=str, required=True,
                        help="Model version (e.g., claude-haiku-4.5, llama-3.3-70b, deepseek-v3.1, gemini-3-flash-preview, gemini-3-pro-preview)")
    parser.add_argument("--prompt", type=str, default="zeroshot_structured",
                        help="Prompt template: zeroshot_structured | space")
    parser.add_argument("--claude-mode", type=str, default="optimized",
                        choices=["simple", "optimized", "thinking"])
    parser.add_argument("--gemini-mode", type=str, default="cached",
                        choices=["simple", "cached", "batch"],
                        help="Gemini mode: 'simple' (no caching), 'cached' (context caching, requires 32K+ tokens), or 'batch' (async batch API, 50%% cost reduction, 5-30+ min turnaround)")
    parser.add_argument("--openai-mode", type=str, default="cached",
                        choices=["simple", "cached"],
                        help="OpenAI mode: 'simple' (no caching) or 'cached' (automatic prompt caching for prompts > 1024 tokens)")
    parser.add_argument("--cache-ttl", type=int, default=2,
                        help="Cache TTL in minutes for Gemini context caching (default: 30)")
    parser.add_argument("--sleep", type=int, default=0, help="Rate limiting seconds")
    parser.add_argument("--no-save", action="store_true", help="Don't save experiment")
    parser.add_argument("--no-dashboard", action="store_true", help="Don't create dashboard after experiment")
    parser.add_argument("--name", type=str, help="Custom experiment name (optional)")
    parser.add_argument("--reasoning-effort", type=str, default="off",
                        choices=["off", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort level for DeepSeek/OpenAI models (default: off, xhigh for GPT-5.2)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Requests per minute rate limit (Gemini only, default: None = no rate limiting)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0.0-2.0, default: 1.0 for Gemini - Google recommended)")
    parser.add_argument("--thinking-level", type=str, default="high",
                        choices=["low", "medium", "high", "off"],
                        help="Thinking level for Gemini 3 models (default: high for deep reasoning, use 'off' to disable)")
    parser.add_argument("--key", type=int, default=None,
                        help="Select API key from .google/other.json by index (1-based). "
                             "Example: --key 1 uses first key, --key 2 uses second key, etc. "
                             "Useful for running multiple parallel experiments with different API keys "
                             "to avoid rate limits. If not specified, uses .google/credentials.json")

    parser.add_argument('--use-self-consistency', action='store_true',
                        help='Enable self-consistency (majority vote over multiple samples)')
    parser.add_argument('--sc-samples', type=int, default=3,
                        help='Number of samples for self-consistency (default: 3)')
    parser.add_argument('--sc-temperature', type=float, default=1.0,
                        help='Temperature for self-consistency sampling (default: 1.0)')
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Number of threads for parallel inference (default: available CPUs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a crashed experiment by providing the experiment path")

    # Graph RAG arguments (Hybrid: semantic + lexical)
    parser.add_argument("--use-graph-rag", action="store_true",
                        help="Enable Hybrid Graph RAG retrieval (semantic + lexical)")
    parser.add_argument("--graph-path", type=str, default="data/indices/doc_graph_dev.pkl",
                        help="Path to precomputed document graphs")
    parser.add_argument("--query-embeddings-path", type=str, default="data/indices/query_embeddings_dev.pkl",
                        help="Path to precomputed query embeddings")
    parser.add_argument("--graph-n-semantic-entry", type=int, default=3,
                        help="Number of semantic entry points for graph traversal")
    parser.add_argument("--graph-n-lexical-entry", type=int, default=2,
                        help="Number of lexical (BM25+) entry points for graph traversal")
    parser.add_argument("--graph-unlimited-traversal", action="store_true", default=True,
                        help="Enable unlimited graph traversal (traverse entire connected component)")
    parser.add_argument("--graph-min-cluster-size", type=int, default=1,
                        help="Minimum cluster size (1=no filtering)")

    args = parser.parse_args()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📦 Loading data...")

    data_path = Path(args.data_dir)
    limit = args.limit if args.limit > 0 else 0
    offset = args.offset

    if args.dataset == "val":
        print(f"   Mode: Loading 'val' (last {limit} from train set)")
        # Load all training data
        all_train_questions, docs = load_train_data(data_path)

        total_train = len(all_train_questions)

        # Take the last 'limit' questions
        if limit > 0 and total_train >= limit:
            questions = all_train_questions[-limit:]
            print(f"   Using last {len(questions)} questions (from index {total_train - len(questions)})")
        else:
            questions = all_train_questions
            print(f"   Using all {len(questions)} train questions (limit not applied or < total)")

    elif args.dataset == "test":
        print("   Mode: Loading 'test' set")
        questions, docs = load_test_data(data_path)

        # Apply offset/limit BEFORE resume filtering
        all_questions_for_resume = questions  # Keep full list for resume logic
        if limit > 0:
            end_idx = offset + limit
            questions = questions[offset:end_idx]
            print(f"   Loaded {len(questions)} questions (range: {offset+1}-{end_idx})")
        else:
            print(f"   Loaded {len(questions)} questions")

    else: # Default "dev" mode
        print("   Mode: Loading 'dev' set")
        questions, docs = load_dev_data(data_path)

        # Apply offset/limit BEFORE resume filtering
        all_questions_for_resume = questions  # Keep full list for resume logic
        if limit > 0:
            end_idx = offset + limit
            questions = questions[offset:end_idx]
            print(f"   Loaded {len(questions)} questions (range: {offset+1}-{end_idx})")
        else:
            print(f"   Loaded {len(questions)} questions")

    # Apply topic-based splitting for parallel runs with multiple API keys
    if args.topic_split:
        try:
            part, total = map(int, args.topic_split.split('/'))
            if part < 1 or part > total:
                raise ValueError(f"Invalid part number: {part} (must be 1-{total})")

            # Group questions by topic
            from collections import defaultdict
            topic_groups = defaultdict(list)
            for q in questions:
                topic_groups[q['topic_id']].append(q)

            # Get sorted topic list for consistent splitting
            all_topics = sorted(topic_groups.keys())
            total_topics = len(all_topics)

            # Calculate topic range for this part
            topics_per_part = total_topics // total
            remainder = total_topics % total

            # Distribute remainder across first parts
            if part <= remainder:
                start_idx = (part - 1) * (topics_per_part + 1)
                end_idx = start_idx + topics_per_part + 1
            else:
                start_idx = remainder * (topics_per_part + 1) + (part - remainder - 1) * topics_per_part
                end_idx = start_idx + topics_per_part

            # Select topics for this part
            my_topics = all_topics[start_idx:end_idx]

            # Filter questions to only include this part's topics
            questions = [q for topic in my_topics for q in topic_groups[topic]]

            print(f"\n📊 Topic-based splitting enabled:")
            print(f"   Part {part} of {total}")
            print(f"   Topics: {len(my_topics)} of {total_topics} total")
            print(f"   Questions: {len(questions)}")
            print(f"   Topics in this part: {', '.join(str(t) for t in my_topics[:5])}{'...' if len(my_topics) > 5 else ''}")
            print("   ✓ Cache efficiency: Each part processes complete topics for maximum cache reuse\n")

        except ValueError as e:
            raise ValueError(f"Invalid --topic-split format: '{args.topic_split}'. Use 'N/M' format (e.g., '1/4'). Error: {e}")

    # Load contextualized docs if provided
    if args.contextualized_docs:
        print("\n📚 Loading contextualized documents...")
        print(f"   Path: {args.contextualized_docs}")
        with open(args.contextualized_docs, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        print("   ✓ Loaded contextualized docs (with agentic chunks)")

    # Set default threads if not specified
    if args.num_threads is None:
        args.num_threads = cpu_count()
        print(f"   Using {args.num_threads} threads (auto-detected)")
    else:
        print(f"   Using {args.num_threads} threads (user-specified)")

    # Graph RAG initialization using central utility module
    graph_rag_data = None
    context_cache = None  # Preprocessed context cache (takes priority)

    if args.use_graph_rag:
        # First, try to load preprocessed topic-wide contexts (fastest, recommended)
        from src.retrieval.graph_rag_utils import PreprocessedContextCache
        # Map 'val' to 'train' for preprocessed context loading (val uses train split)
        preprocessed_split = 'train' if args.dataset == 'val' else args.dataset
        preprocessed_cache = PreprocessedContextCache(split=preprocessed_split, docs=docs)

        if preprocessed_cache.using_preprocessed:
            # Use preprocessed contexts (default when available)
            context_cache = preprocessed_cache
            print("   📦 Using preprocessed topic-wide contexts")
        else:
            # Fall back to runtime GraphRAG retrieval
            graph_rag_data = load_graph_rag_data(
                graph_path=args.graph_path,
                query_embeddings_path=args.query_embeddings_path,
                n_semantic_entry=args.graph_n_semantic_entry,
                n_lexical_entry=args.graph_n_lexical_entry,
                min_cluster_size=args.graph_min_cluster_size,
                max_docs=None,  # UNLIMITED: Retrieve ALL connected documents (no limit)
                verbose=True
            )
            if not graph_rag_data:
                raise FileNotFoundError(f"Graph file not found: {args.graph_path}. Run build_document_graph.py first.")

            # Ensure unlimited traversal is enabled
            graph_rag_data['unlimited_traversal'] = args.graph_unlimited_traversal

    # ========================================================================
    # GET PROMPT TEMPLATE
    # ========================================================================

    system_prompt = get_template(args.prompt)

    # ========================================================================
    # BUILD CLIENT
    # ========================================================================

    print(f"\n🤖 Creating {args.model_family} client ({args.version})...")

    if args.model_family == "claude":
        chat = build_client(args.model_family, args.version, system_prompt,
                           claude_mode=args.claude_mode)
    elif args.model_family == "gemini":
        chat = build_client(args.model_family, args.version, system_prompt,
                           gemini_mode=args.gemini_mode,
                           temperature=args.temperature,
                           thinking_level=args.thinking_level,
                           key_index=args.key)
    elif args.model_family == "openai":
        chat = build_client(args.model_family, args.version, system_prompt,
                           openai_mode=args.openai_mode,
                           temperature=args.temperature,
                           reasoning_effort=args.reasoning_effort,
                           key_index=args.key)
    else:
        chat = build_client(args.model_family, args.version, system_prompt)

    # ========================================================================
    # INITIALIZE INCREMENTAL SAVING (or RESUME)
    # ========================================================================

    experiment_path = None
    if not args.no_save:
        # SC resume is handled separately in the OpenAI SC inference section
        if args.resume and args.use_self_consistency and args.model_family == "openai":
            experiment_path = args.resume
            print(f"\n🔄 SC Resume mode - will handle in inference section")
            print(f"   Resume path: {experiment_path}")

        # Check if resuming an existing experiment (non-SC)
        elif args.resume:
            experiment_path = args.resume
            print("\n🔄 Resuming experiment...")
            print(f"   Path: {experiment_path}")

            # Load existing results to identify failed and unprocessed questions
            results_path = Path(experiment_path) / "results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)

                original_count = len(questions)
                processed_ids = set(existing_results.get('predictions', {}).keys())

                # Identify failed questions (Fail/Failed/N/A predictions, empty analyses, etc.)
                failed_ids = set(identify_failed_questions(existing_results, questions))

                # Find unprocessed questions (not in predictions at all)
                all_question_ids = set(q['id'] for q in questions)
                unprocessed_ids = all_question_ids - processed_ids

                # Combine: retry failed + continue with unprocessed
                ids_to_process = failed_ids | unprocessed_ids
                questions = [q for q in questions if q['id'] in ids_to_process]

                # Show summary
                successful_count = len(processed_ids) - len(failed_ids)
                print(f"   Total questions: {original_count}")
                print(f"   ✓ Successfully processed: {successful_count}")
                if failed_ids:
                    print(f"   ✗ Failed (will retry): {len(failed_ids)}")
                    print(f"      (Fail/Failed/N/A predictions, empty analyses, etc.)")
                if unprocessed_ids:
                    print(f"   ○ Not yet processed: {len(unprocessed_ids)}")
                print(f"   → To process now: {len(questions)}")

                if len(questions) == 0:
                    print("\n✅ All questions already processed!")
                    print("   Finalizing experiment...")

                    # Reload all questions for evaluation (skip for test dataset - no golden answers)
                    if args.dataset == "test":
                        score = None
                        all_questions, _ = load_test_data(data_path)
                        if limit > 0:
                            end_idx = offset + limit
                            all_questions = all_questions[offset:end_idx]
                        print("   ⚠️  Test dataset - skipping evaluation (no golden answers)")
                    else:
                        if args.dataset == "val":
                            all_questions, _ = load_train_data(data_path)
                            if limit > 0:
                                all_questions = all_questions[-limit:]
                        else:
                            all_questions, _ = load_dev_data(data_path)
                            if limit > 0:
                                end_idx = offset + limit
                                all_questions = all_questions[offset:end_idx]

                        score = evaluate(existing_results['predictions'], all_questions)

                    # Update questions.json with the full question set for dashboard
                    questions_path = Path(experiment_path) / "questions.json"
                    with open(questions_path, 'w', encoding='utf-8') as f:
                        json.dump(all_questions, f, indent=2, ensure_ascii=False)
                    print(f"   ✓ Updated questions.json with {len(all_questions)} questions")

                    # Update metadata.json with correct question count
                    metadata_path = Path(experiment_path) / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        metadata['num_questions'] = len(all_questions)
                        metadata['question_ids'] = [q['id'] for q in all_questions]
                        metadata['topics'] = list(set(q['topic_id'] for q in all_questions))
                        metadata['num_topics'] = len(metadata['topics'])
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        print(f"   ✓ Updated metadata.json")

                    finalize_experiment(
                        experiment_path=experiment_path,
                        results=existing_results,
                        score=score,
                        elapsed=0
                    )

                    # Create Codabench submission file (regenerate even if complete)
                    try:
                        predictions_dict = existing_results['predictions']
                        split_name = 'test' if args.dataset == 'test' else 'dev'
                        jsonl_path, zip_path = create_submission_file(predictions_dict, experiment_path, split=split_name)
                        print("\n📤 Created Codabench submission:")
                        print(f"   {zip_path}")
                        print("   Upload this file to https://www.codabench.org/competitions/12446/")
                    except Exception as e:
                        print(f"\n⚠️  Submission file creation failed: {e}")

                    # Create dashboard automatically (regenerate even if complete)
                    if not args.no_dashboard:
                        try:
                            print("\n📊 Regenerating dashboard...")
                            create_dashboard(experiment_path)
                        except (FileNotFoundError, ValueError, OSError) as e:
                            print(f"\n⚠️  Dashboard creation failed: {e}")
                    else:
                        print("\n💡 Dashboard skipped (use without --no-dashboard to create)")
                        print(f"   To create later: python scripts/create_dashboard.py --path {experiment_path}")

                    sys.exit(0)
            else:
                print("   ⚠️  No existing results found, initializing experiment folder...")

                # Create the experiment directory structure manually
                exp_path = Path(experiment_path)
                exp_path.mkdir(parents=True, exist_ok=True)

                # Create initial files similar to initialize_experiment_folder
                # but without creating a new timestamped folder

                # Save questions
                questions_path = exp_path / "questions.json"
                with open(questions_path, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, indent=2, ensure_ascii=False)

                # Save metadata
                from datetime import datetime
                metadata = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "experiment_name": exp_path.name,
                    "model_name": exp_path.parent.name,
                    "prompt": args.prompt,
                    "num_questions": len(questions),
                    "question_ids": [q["id"] for q in questions],
                    "topics": list(set(q["topic_id"] for q in questions)),
                    "num_topics": len(set(q["topic_id"] for q in questions)),
                    "status": "running",
                    "created_at": datetime.now().isoformat()
                }
                metadata_path = exp_path / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Save prompt
                prompt_path = exp_path / "prompt.txt"
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(system_prompt)

                # Initialize empty results file
                results_path = exp_path / "results.json"
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "predictions": {},
                        "analyses": {},
                        "thinkings": {}
                    }, f, indent=2, ensure_ascii=False)

                print(f"   ✓ Initialized experiment folder: {experiment_path}")
        else:
            # Create new experiment
            experiment_name = args.name
            if experiment_name is None:
                # Auto-generate name
                mode_suffix = f"_{args.claude_mode}" if args.model_family == "claude" else ""
                experiment_name = f"{args.version}_{args.prompt}{mode_suffix}"

            # For topic-split runs, remove the part suffix from experiment name
            # All parts should write to the same experiment directory
            if args.topic_split and experiment_name:
                import re
                # Remove _partN_of_M suffix if it exists
                experiment_name = re.sub(r'_part\d+_of_\d+$', '', experiment_name)

            print("\n💾 Initializing incremental save...")
            print(f"   Experiment: {experiment_name}")
            if args.topic_split:
                print("   ✓ All topic-split parts will write to the same experiment directory")

            # Initialize the experiment folder
            from src.experiments.manager import initialize_experiment_folder
            experiment_path = initialize_experiment_folder(
                model_name=args.version,
                experiment_name=experiment_name,
                prompt=args.prompt,
                questions=questions
            )
            print(f"   Path: {experiment_path}")

    # ========================================================================
    # RUN INFERENCE WITH INCREMENTAL SAVING
    # ========================================================================

    print("\n🔬 Running inference with incremental saving...")
    if args.use_graph_rag:
        print("   📊 Graph RAG enabled: UNLIMITED traversal (entire connected component)")

    start = time.time()

    if args.model_family == "claude":

        results = run_claude_inference(
            chat, questions, docs,
            mode=args.claude_mode,
            sleep_seconds=args.sleep,
            use_self_consistency=args.use_self_consistency,
            sc_samples=args.sc_samples,
            sc_temperature=args.sc_temperature,
            num_threads=args.num_threads,
            experiment_path=experiment_path,  # Pass save path for incremental saving
            graph_rag_data=graph_rag_data,  # Pass Graph RAG data
            context_cache=context_cache,  # Preprocessed context cache (takes priority)
        )

    elif args.model_family == "llama":
        results = run_llama_inference(
            chat, questions, docs,
            sleep_seconds=args.sleep,
            experiment_path=experiment_path
        )

    elif args.model_family == "deepseek":
        version_str = args.version.lower()
        if 'r1' in version_str:
            deepseek_version = 'r1'
        elif 'v3.1' in version_str or 'v31' in version_str:
            deepseek_version = 'v3.1'
        else:
            deepseek_version = 'v3.1'
        
        results = run_deepseek_inference(
            chat, questions, docs,
            version=deepseek_version,
            sleep_seconds=args.sleep,
            reasoning_effort=args.reasoning_effort,
            experiment_path=experiment_path
        )

    elif args.model_family == "openai":
        if args.use_self_consistency:
            from src.experiments.manager import initialize_experiment_folder

            base_name = args.name if args.name else f"{args.version}_{args.prompt}"
            all_questions = list(questions)  # Keep full list before resume filtering

            # ── SC RESUME: find existing sample dirs, identify failed questions ──
            existing_sample_paths = []
            existing_sample_results = []
            questions_to_process = questions

            if args.resume:
                import re
                resume_base = Path(args.resume)
                print(f"\n🔄 SC Resume: scanning for existing sample experiments...")

                # Extract base pattern from resume path and use it as base_name
                if resume_base.exists() and resume_base.is_dir():
                    # User passed a specific sample dir — extract base by stripping _sc_sampleN
                    parent = resume_base.parent
                    dir_name = resume_base.name
                    base_pattern = re.sub(r'_sc_sample\d+.*$', '', dir_name)
                else:
                    # Resume path doesn't exist yet — use its name as base
                    parent = resume_base.parent
                    # Strip any trailing _sc suffix if user added it
                    base_pattern = re.sub(r'_sc$', '', resume_base.name)

                # CRITICAL: Update base_name to match resume path
                # This ensures new directories use the resume name, not auto-generated name
                base_name = base_pattern

                print(f"   Base name: {base_name}")
                print(f"   Search dir: {parent}")

                # Scan for _sc_sampleN directories
                for i in range(1, args.sc_samples + 1):
                    matches = sorted(parent.glob(f"{base_name}_sc_sample{i}_*"))
                    if matches:
                        sample_dir = matches[-1]  # Use most recent
                        results_file = sample_dir / "results.json"
                        if results_file.exists():
                            with open(results_file, 'r', encoding='utf-8') as f:
                                sample_data = json.load(f)
                            existing_sample_paths.append(str(sample_dir))
                            existing_sample_results.append(sample_data)
                            print(f"   ✓ Found sample {i}: {sample_dir.name} ({len(sample_data.get('predictions', {}))} predictions)")
                        else:
                            existing_sample_paths.append(None)
                            existing_sample_results.append(None)
                            print(f"   ○ Sample {i}: directory exists but no results.json")
                    else:
                        existing_sample_paths.append(None)
                        existing_sample_results.append(None)
                        print(f"   ○ Sample {i}: not found")

                # Identify questions that need re-processing
                # A question needs re-processing if it failed in ANY sample
                all_question_ids = set(q['id'] for q in questions)
                ids_needing_retry = set()

                for sample_idx, sample_data in enumerate(existing_sample_results):
                    if sample_data is None:
                        # Entire sample missing — need to process all questions
                        ids_needing_retry = all_question_ids
                        break

                    preds = sample_data.get('predictions', {})
                    processed_ids = set(preds.keys())
                    unprocessed = all_question_ids - processed_ids

                    # Check for failed predictions
                    failed = set()
                    for qid, pred in preds.items():
                        pred_str = pred if isinstance(pred, str) else ','.join(pred) if isinstance(pred, list) else str(pred)
                        if pred_str in ('Fail', 'Failed', 'N/A', ''):
                            failed.add(qid)

                    ids_needing_retry |= unprocessed | failed

                questions_to_process = [q for q in questions if q['id'] in ids_needing_retry]

                successful = len(all_question_ids) - len(ids_needing_retry)
                print(f"\n   Total questions: {len(all_question_ids)}")
                print(f"   ✓ Successfully processed in all samples: {successful}")
                if ids_needing_retry:
                    print(f"   → To re-process now: {len(ids_needing_retry)}")
                else:
                    print(f"\n✅ All questions already processed in all samples!")
                    # Print combine command with existing paths
                    valid_paths = [p for p in existing_sample_paths if p is not None]
                    if valid_paths:
                        paths_str = " ".join(valid_paths)
                        print(f"\nTo combine with self-consistency voting, run:")
                        print(f"\n  python scripts/combine_experiments_sc.py \\")
                        print(f"    --experiments {paths_str} \\")
                        print(f"    --threshold 0.35\n")
                    sys.exit(0)

            # ── INITIALIZE EXPERIMENT DIRECTORIES FOR INCREMENTAL SAVING ──
            sc_experiment_paths = []
            print(f"\n💾 Initializing {args.sc_samples} sample experiment directories...")
            for sample_idx in range(args.sc_samples):
                if args.resume and sample_idx < len(existing_sample_paths) and existing_sample_paths[sample_idx]:
                    # Use existing directory
                    sample_path = existing_sample_paths[sample_idx]
                    print(f"   ✓ Sample {sample_idx + 1}: Using existing {Path(sample_path).name}")
                else:
                    # Create new directory
                    sample_name = f"{base_name}_sc_sample{sample_idx + 1}"
                    sample_path = initialize_experiment_folder(
                        model_name=args.version,
                        experiment_name=sample_name,
                        prompt=args.prompt,
                        questions=all_questions
                    )
                    print(f"   ✓ Sample {sample_idx + 1}: {Path(sample_path).name}")
                sc_experiment_paths.append(sample_path)

            # ── RUN SC INFERENCE WITH INCREMENTAL SAVING ──
            sample_results_list = run_openai_sc_inference(
                chat, questions_to_process, docs,
                sc_samples=args.sc_samples,
                sc_temperature=args.sc_temperature,
                mode=args.openai_mode,
                sleep_seconds=args.sleep,
                graph_rag_data=graph_rag_data,
                context_cache=context_cache,
                temperature=args.temperature,
                experiment_paths=sc_experiment_paths,
            )

            elapsed = time.time() - start

            # ── FINALIZE: Merge with existing (if resume) & create dashboards ──
            print(f"\n✅ Finalizing {args.sc_samples} sample experiments...")

            for sample_idx in range(args.sc_samples):
                new_results = sample_results_list[sample_idx]
                sample_path = sc_experiment_paths[sample_idx]

                # Merge with existing results if resuming
                if args.resume and sample_idx < len(existing_sample_results) and existing_sample_results[sample_idx] is not None:
                    merged = existing_sample_results[sample_idx]
                    merged["predictions"].update(new_results["predictions"])
                    merged["analyses"].update(new_results["analyses"])
                    if "thinkings" not in merged:
                        merged["thinkings"] = {}
                    merged["thinkings"].update(new_results.get("thinkings", {}))

                    # Merge cost tracker
                    if "cost_tracker" in new_results and new_results["cost_tracker"]:
                        new_cost = new_results["cost_tracker"]
                        if hasattr(new_cost, 'get_summary'):
                            new_cost = new_cost.get_summary()
                        existing_cost = merged.get("cost_tracker", {})
                        if existing_cost:
                            for key in ['total_cost', 'total_input_tokens', 'total_output_tokens', 'total_tokens']:
                                if key in new_cost:
                                    existing_cost[key] = existing_cost.get(key, 0) + new_cost[key]
                            merged["cost_tracker"] = existing_cost
                        else:
                            merged["cost_tracker"] = new_cost

                    final_results = merged
                else:
                    final_results = new_results
                    # Convert cost tracker to dict
                    if 'cost_tracker' in final_results and final_results['cost_tracker']:
                        if hasattr(final_results['cost_tracker'], 'get_summary'):
                            final_results['cost_tracker'] = final_results['cost_tracker'].get_summary()

                # Save results.json
                results_path = Path(sample_path) / "results.json"
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)

                # Update questions.json to full set
                questions_path = Path(sample_path) / "questions.json"
                with open(questions_path, 'w', encoding='utf-8') as f:
                    json.dump(all_questions, f, indent=2, ensure_ascii=False)

                # Evaluate this sample (if not test dataset)
                sample_score = None
                if args.dataset != "test":
                    preds_for_eval = {}
                    for qid, pred in final_results["predictions"].items():
                        if isinstance(pred, list):
                            preds_for_eval[qid] = ','.join(pred) if pred else ''
                        else:
                            preds_for_eval[qid] = pred
                    sample_score = evaluate(preds_for_eval, all_questions)

                # Finalize this sample's metadata
                finalize_experiment(
                    experiment_path=sample_path,
                    results=final_results,
                    score=sample_score,
                    elapsed=elapsed / args.sc_samples
                )

                # Create Codabench submission file for this sample
                try:
                    split_name = 'test' if args.dataset == 'test' else 'dev'
                    jsonl_path, zip_path = create_submission_file(
                        final_results['predictions'],
                        sample_path,
                        split=split_name
                    )
                except Exception as e:
                    print(f"      ⚠️  Submission creation failed: {e}")

                # Create dashboard for this sample (unless --no-dashboard)
                if not args.no_dashboard:
                    try:
                        create_dashboard(sample_path)
                    except Exception as e:
                        print(f"      ⚠️  Dashboard creation failed: {e}")

                # Print sample summary
                cost_summary = final_results.get('cost_tracker', {})
                total_cost_sample = cost_summary.get('total_cost', 0) if isinstance(cost_summary, dict) else 0
                n_preds = len(final_results.get("predictions", {}))
                if sample_score is not None:
                    print(f"   ✓ Sample {sample_idx + 1}: {n_preds} predictions, Score={sample_score:.4f}, Cost=${total_cost_sample:.4f}")
                else:
                    print(f"   ✓ Sample {sample_idx + 1}: {n_preds} predictions, Cost=${total_cost_sample:.4f}")

            # Aggregate stats
            total_cost = sum(
                (sr.get('cost_tracker', {}).get('total_cost', 0) if isinstance(sr.get('cost_tracker', {}), dict) else 0)
                for sr in sample_results_list
            )

            print(f"\n{'='*80}")
            print("SELF-CONSISTENCY: AGGREGATE SUMMARY")
            print(f"{'='*80}")
            print(f"Saved {args.sc_samples} sample experiments")
            print(f"Total time: {elapsed:.2f}s ({len(questions_to_process)/max(elapsed,1e-6):.2f} q/s)")
            print(f"Total cost (this run): ${total_cost:.4f}")

            if args.dataset != "test":
                scores = []
                for si in range(args.sc_samples):
                    fr = sample_results_list[si] if not args.resume else (
                        # Use merged results for scoring
                        json.load(open(Path(sc_experiment_paths[si]) / "results.json", 'r', encoding='utf-8'))
                    )
                    pfe = {qid: (p if isinstance(p, str) else ','.join(p))
                           for qid, p in fr["predictions"].items()}
                    scores.append(evaluate(pfe, all_questions))
                print(f"Avg score across samples: {sum(scores)/len(scores):.4f}")
                print(f"Individual scores: {', '.join(f'{s:.4f}' for s in scores)}")

            print(f"\nTo combine with self-consistency voting, run:")
            paths_str = " ".join(sc_experiment_paths)
            print(f"\n  python scripts/combine_experiments_sc.py \\")
            print(f"    --experiments {paths_str} \\")
            print(f"    --threshold 0.35\n")
            print(f"{'='*80}\n")

            sys.exit(0)
        else:
            results = run_openai_inference(
                chat, questions, docs,
                mode=args.openai_mode,
                sleep_seconds=args.sleep,
                num_threads=args.num_threads,
                experiment_path=experiment_path,
                graph_rag_data=graph_rag_data,
                context_cache=context_cache,
                temperature=args.temperature,
                )

    else:  # gemini
        if args.gemini_mode == "batch":
            # Use TRUE Batch API (async, 50% cost reduction)
            from src.inference.gemini import GeminiBatchInference
            from configs.google_config import get_gemini_client, get_gemini_model_id

            print("\n🔷 Using Gemini Batch API (50% cost reduction)")
            print("   ⚠️  This is asynchronous - results may take 5-30+ minutes")

            # Get Gemini client and model
            gemini_client = get_gemini_client(key_index=args.key)
            model_id = get_gemini_model_id(args.version)

            # Create batch inference instance with system prompt and context cache
            batch_inference = GeminiBatchInference(
                gemini_client=gemini_client,
                model_id=model_id,
                experiment_path=experiment_path,
                poll_interval=30,
                max_wait_hours=24,
                graph_rag_data=graph_rag_data,  # Pass Graph RAG data if enabled
                context_cache=context_cache,  # Pass preprocessed context cache (takes priority)
                system_prompt=system_prompt  # Pass system prompt template
            )

            # Run batch inference
            results = batch_inference.run(questions, docs)

        else:
            # Use standard real-time inference (simple or cached)
            results = run_gemini_inference(
                chat, questions, docs,
                mode=args.gemini_mode,
                sleep_seconds=args.sleep,
                use_self_consistency=args.use_self_consistency,
                sc_samples=args.sc_samples,
                sc_temperature=args.sc_temperature,
                num_threads=args.num_threads,
                rpm=args.rpm,  # Pass rate limit
                experiment_path=experiment_path,
                graph_rag_data=graph_rag_data,
                cache_ttl_minutes=args.cache_ttl,
                context_cache=context_cache,  # Preprocessed context cache (takes priority)
                temperature=args.temperature,
                thinking_level=args.thinking_level,
                )

    elapsed = time.time() - start

    # ========================================================================
    # EVALUATE
    # ========================================================================

    # Convert predictions to string format for evaluation (if they're lists)
    preds_for_eval = {}
    for qid, pred in results["predictions"].items():
        if isinstance(pred, list):
            preds_for_eval[qid] = ','.join(pred) if pred else ''
        else:
            preds_for_eval[qid] = pred

    # Evaluate only if not test dataset (test has no golden answers)
    if args.dataset == "test":
        score = None
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Dataset:   test (no evaluation - no golden answers)")
        print(f"Questions: {len(questions)}")
        print(f"Time:      {elapsed:.2f}s ({len(questions)/max(elapsed,1e-6):.2f} q/s)")
        print(f"Threads:   {args.num_threads}")
        print(f"{'='*80}")
    else:
        score = evaluate(preds_for_eval, questions)

        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Score:     {score:.4f}")
        print(f"Questions: {len(questions)}")
        print(f"Time:      {elapsed:.2f}s ({len(questions)/max(elapsed,1e-6):.2f} q/s)")
        print(f"Threads:   {args.num_threads}")
        print(f"{'='*80}")

    # Display cost summary if available
    if 'cost_tracker' in results and results['cost_tracker']:
        results['cost_tracker'].print_summary()

    # ========================================================================
    # DETAILED ANALYSIS
    # ========================================================================

    # Skip analysis for test dataset (requires golden answers)
    if args.dataset != "test":
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")
        analyze_predictions(results["predictions"], questions)
    else:
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")
        print("   ⚠️  Skipped (test dataset has no golden answers)")
        print(f"{'='*80}")

    # ========================================================================
    # FINALIZE EXPERIMENT
    # ========================================================================

    if not args.no_save and experiment_path:
        print(f"\n{'='*80}")
        print("FINALIZING EXPERIMENT")
        print(f"{'='*80}")

        # CRITICAL: Merge batch results with existing results.json when resuming
        if args.resume:
            results_path = Path(experiment_path) / "results.json"
            if results_path.exists():
                print("   Merging results with existing results.json...")
                with open(results_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)

                # Save original score for comparison if we retried failed questions
                metadata_path = Path(experiment_path) / "metadata.json"
                original_score = None
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        original_score = metadata.get("score")

                # Merge predictions, analyses, etc. (new results override)
                existing_results["predictions"].update(results["predictions"])
                existing_results["analyses"].update(results["analyses"])
                if "thinkings" in results:
                    if "thinkings" not in existing_results:
                        existing_results["thinkings"] = {}
                    existing_results["thinkings"].update(results["thinkings"])

                # Merge cost tracker (convert new CostTracker object to dict if needed)
                if "cost_tracker" in results and results["cost_tracker"]:
                    if hasattr(results["cost_tracker"], 'get_summary'):
                        # New result has CostTracker object, convert to dict
                        new_cost_summary = results["cost_tracker"].get_summary()
                        # Merge with existing cost data (existing is already a dict from JSON)
                        if "cost_tracker" in existing_results and existing_results["cost_tracker"]:
                            # Merge costs by adding them together
                            for key in ['total_cost', 'total_input_tokens', 'total_output_tokens',
                                       'total_cache_write_tokens', 'total_cache_read_tokens', 'total_tokens']:
                                if key in new_cost_summary:
                                    existing_results["cost_tracker"][key] = existing_results["cost_tracker"].get(key, 0) + new_cost_summary[key]
                            # Merge stages
                            if 'stages' in new_cost_summary:
                                if 'stages' not in existing_results["cost_tracker"]:
                                    existing_results["cost_tracker"]['stages'] = {}
                                for stage_name, stage_data in new_cost_summary['stages'].items():
                                    if stage_name in existing_results["cost_tracker"]['stages']:
                                        # Merge existing stage
                                        for key in stage_data.keys():
                                            existing_results["cost_tracker"]['stages'][stage_name][key] = \
                                                existing_results["cost_tracker"]['stages'][stage_name].get(key, 0) + stage_data[key]
                                    else:
                                        # New stage
                                        existing_results["cost_tracker"]['stages'][stage_name] = stage_data
                        else:
                            existing_results["cost_tracker"] = new_cost_summary

                # Use merged results for finalization
                results = existing_results
                print(f"   ✓ Merged: {len(results['predictions'])} total predictions")

                # Show improvement if we retried failed questions
                if original_score is not None and score is not None and isinstance(original_score, (int, float)):
                    improvement = score - original_score
                    if improvement != 0:
                        print(f"\n   📈 Retry Summary:")
                        print(f"      Original score: {original_score:.4f}")
                        print(f"      New score:      {score:.4f}")
                        print(f"      Improvement:    {improvement:+.4f}")

        # Convert CostTracker to dict before saving (not JSON serializable)
        if 'cost_tracker' in results and results['cost_tracker']:
            # Convert CostTracker object to dictionary
            if hasattr(results['cost_tracker'], 'get_summary'):
                results['cost_tracker'] = results['cost_tracker'].get_summary()

        # Save merged results.json
        results_path = Path(experiment_path) / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("   ✓ Saved results.json")

        # Finalize the experiment (save final metadata, cost tracker, etc.)
        finalize_experiment(
            experiment_path=experiment_path,
            results=results,
            score=score,
            elapsed=elapsed
        )

        # Create Codabench submission file
        try:
            predictions_dict = results['predictions']
            # Determine split name from dataset argument
            # 'val' is a subset of train for validation, so it maps to 'dev' for submission
            # 'test' uses the actual test split for final submission
            split_name = 'test' if args.dataset == 'test' else 'dev'
            jsonl_path, zip_path = create_submission_file(predictions_dict, experiment_path, split=split_name)
            print("\n📤 Created Codabench submission:")
            print(f"   {zip_path}")
            print("   Upload this file to https://www.codabench.org/competitions/12446/")
        except Exception as e:
            print(f"\n⚠️  Submission file creation failed: {e}")

        # Clean up memory
        del results
        gc.collect()

        # Create dashboard automatically
        if not args.no_dashboard:
            try:
                create_dashboard(experiment_path)
            except (FileNotFoundError, ValueError, OSError) as e:
                print(f"\n⚠️  Dashboard creation failed: {e}")
        else:
            print("\n💡 Dashboard skipped (use without --no-dashboard to create)")
            print(f"   To create later: python scripts/create_dashboard.py --path {experiment_path}")


if __name__ == "__main__":
    main()
