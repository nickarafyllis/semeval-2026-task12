"""
Retry Failed Questions Utility

Identifies failed questions in an experiment and retries them with updated settings,
then merges the results back into a new experiment.
"""

from typing import Dict, List, Any
from src.experiments.manager import (
    list_experiments,
    load_experiment_results,
    save_experiment_results
)
from src.experiments.dashboard import create_dashboard
from src.evaluation.metrics import evaluate


def identify_failed_questions(
    results: Dict[str, Any],
    questions: List[Dict]
) -> List[str]:
    """
    Identify UUIDs of questions that failed during inference.

    A question is considered failed if:
    - Prediction contains "FAIL" string
    - Prediction is not A/B/C/D or a comma-separated combination thereof
    - Analysis is empty or missing
    - Thinking is missing (if experiment has thinking mode)
    """
    failed_ids = []

    predictions = results.get("predictions", {})
    analyses = results.get("analyses", {})
    thinkings = results.get("thinkings", {})
    has_thinking_mode = bool(thinkings)

    valid_answers = {"A", "B", "C", "D"}

    for id in [q["id"] for q in questions]:
        pred = predictions.get(id)
        analysis = analyses.get(id, "")
        thinking = thinkings.get(id, "") if has_thinking_mode else ""

        # --- Handle prediction format ---
        if isinstance(pred, list):
            # flatten possible nested lists (just in case)
            pred_items = [str(p).strip() for p in pred if isinstance(p, str)]
        elif isinstance(pred, str):
            pred_items = [p.strip() for p in pred.split(",")]
        else:
            pred_items = []

        # --- Fail checks ---
        is_fail_string = any("FAIL" in p for p in pred_items)

        invalid_prediction = (
            not pred_items
            or any(p not in valid_answers for p in pred_items)
        )

        empty_analysis = (
            not analysis
            or analysis.strip() in {"", " ", "Retry failed"}
        )

        missing_thinking = has_thinking_mode and (not thinking or thinking.strip() == "")

        if is_fail_string or invalid_prediction or empty_analysis or missing_thinking:
            failed_ids.append(id)

    return failed_ids



def retry_failed_questions(
    experiment_name_pattern: str,
    chat_client: Any,
    inference_function: Any,
    docs: List[Dict],
    sleep_seconds: int = 1,
    create_dashboard_after: bool = True,
    base_folder: str = None
) -> Dict[str, Any]:
    """
    Retry failed questions from an experiment and merge results.

    Workflow:
    1. Find experiment matching pattern
    2. Load experiment data
    3. Identify failed questions
    4. Re-run inference on failed questions only
    5. Merge new results into original results
    6. Save as new merged experiment
    7. Optionally create dashboard

    Args:
        experiment_name_pattern: Pattern to find target experiment (case-insensitive)
        chat_client: LLM client instance (e.g., ChatClaude, ChatLlama)
        inference_function: Inference function to use (e.g., run_claude_inference)
        docs: Document list for context
        sleep_seconds: Sleep between API calls for rate limiting
        create_dashboard_after: Whether to create dashboard after merge
        base_folder: Experiments base folder (defaults to ./experiments)

    Returns:
        Dictionary with results, stats, and paths
    """
    # 1. Find the target experiment
    print("Finding target experiment...")
    print("="*80)

    try:
        exps = list_experiments(base_folder=base_folder, as_dataframe=True)

        if exps.empty:
            return {"error": "No experiments found"}

        # Filter by pattern (case-insensitive)
        matching = exps[
            exps["experiment_name"].str.contains(
                experiment_name_pattern,
                case=False,
                na=False
            )
        ]

        if matching.empty:
            return {"error": f"No experiment found matching pattern: {experiment_name_pattern}"}

        # Use most recent match
        exp_path = matching.iloc[0]["path"]
        exp_name = matching.iloc[0]["experiment_name"]

        print(f"Found: {exp_name}")
        print(f"   Path: {exp_path}")

    except Exception as e:
        print(f"Error finding experiment: {e}")
        return {"error": str(e)}

    # 2. Load the experiment
    print("\nLoading experiment data...")
    print("-"*80)

    try:
        loaded = load_experiment_results(exp_path)
        old_results = loaded["results"]
        all_questions = loaded["questions"]
        metadata = loaded["metadata"]
        model_name = metadata["model_name"]
        prompt = metadata.get("prompt", "")

        print(f"Loaded {len(all_questions)} questions")
        print(f"   Model: {model_name}")
        print(f"   Original score: {metadata.get('score', 'N/A')}")

    except Exception as e:
        print(f"Error loading experiment: {e}")
        return {"error": str(e)}

    # 3. Identify failed questions
    print("\nIdentifying failed questions...")
    print("-"*80)

    failed_ids = identify_failed_questions(old_results, all_questions)

    if not failed_ids:
        print("No failed questions found!")
        return {
            "success": True,
            "message": "No retries needed",
            "failed_count": 0,
            "original_score": metadata.get("score")
        }

    print(f"Found {len(failed_ids)} failed questions to retry:")
    print(f"   {', '.join(failed_ids[:5])}" + ("..." if len(failed_ids) > 5 else ""))

    # 4. Create subset of failed questions
    subset_questions = [q for q in all_questions if q["id"] in failed_ids]

    # 5. Re-run inference on failed questions
    print(f"\nRetrying {len(subset_questions)} questions...")
    print("-"*80)

    try:
        # Call the inference function with failed questions only
        new_results = inference_function(
            chat_client,
            subset_questions,
            docs,
            sleep_seconds=sleep_seconds
        )

        # Calculate score on retried subset
        retry_score = evaluate(new_results["predictions"], subset_questions)
        print(f"Retry complete - Subset score: {retry_score:.4f}")

    except Exception as e:
        print(f"Error during retry: {e}")
        return {"error": str(e)}

    # 6. Merge new results into old results
    print("\nMerging results...")
    print("-"*80)

    merge_stats = {"updated": 0, "predictions": 0, "analyses": 0, "thinkings": 0}

    new_preds = new_results.get("predictions", {})
    new_analyses = new_results.get("analyses", {})
    new_thinkings = new_results.get("thinkings", {})

    for id in failed_ids:
        if id in new_preds:
            old_results["predictions"][id] = new_preds[id]
            merge_stats["predictions"] += 1

        if id in new_analyses:
            old_results["analyses"][id] = new_analyses[id]
            merge_stats["analyses"] += 1

        if "thinkings" in old_results and id in new_thinkings:
            old_results["thinkings"][id] = new_thinkings[id]
            merge_stats["thinkings"] += 1

        merge_stats["updated"] += 1

    print(f"Merged {merge_stats['updated']} questions:")
    print(f"   Predictions: {merge_stats['predictions']}")
    print(f"   Analyses: {merge_stats['analyses']}")
    print(f"   Thinkings: {merge_stats['thinkings']}")

    # 7. Calculate final score on full dataset
    print("\nCalculating final score on full dataset...")
    print("-"*80)

    final_score = evaluate(old_results["predictions"], all_questions)
    original_score = metadata.get("score", 0)
    improvement = final_score - original_score if isinstance(original_score, (int, float)) else "N/A"

    print(f"Final merged score: {final_score:.4f}")
    if isinstance(improvement, float):
        print(f"   Improvement: {improvement:+.4f}")

    # 8. Save merged experiment
    print("\nSaving merged experiment...")
    print("-"*80)

    try:
        # Extract original experiment name without timestamp
        original_name_parts = exp_name.rsplit('_', 2)
        if len(original_name_parts) >= 3:
            # Has timestamp (name_YYYYMMDD_HHMMSS)
            base_name = original_name_parts[0]
        else:
            base_name = exp_name

        merged_experiment_name = f"{base_name}_merged"

        save_path = save_experiment_results(
            results=old_results,
            questions=all_questions,
            model_name=model_name,
            prompt=prompt,
            experiment_name=merged_experiment_name,
            base_folder=base_folder,
            score=final_score
        )

        print(f"Saved: {merged_experiment_name}")

    except Exception as e:
        print(f"Error saving: {e}")
        return {"error": str(e)}

    # 9. Create dashboard if requested
    dashboard_path = None
    if create_dashboard_after:
        print("\nCreating dashboard...")
        print("-"*80)

        try:
            dashboard_path = create_dashboard(save_path)
            print("Dashboard created")
        except Exception as e:
            print(f"Dashboard creation failed: {e}")

    # 10. Return comprehensive results
    print("\n" + "="*80)
    print("RETRY COMPLETE")
    print("="*80)

    return {
        "success": True,
        "original_experiment": exp_name,
        "merged_experiment": merged_experiment_name,
        "save_path": save_path,
        "dashboard_path": dashboard_path,
        "failed_count": len(failed_ids),
        "merge_stats": merge_stats,
        "original_score": original_score,
        "final_score": final_score,
        "improvement": improvement,
        "model_name": model_name
    }
