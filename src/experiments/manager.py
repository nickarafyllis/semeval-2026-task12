"""
Experiment management for abductive event reasoning

Experiments are organized by model, with customizable naming and timestamps.
"""

import json
import fcntl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd
from src.evaluation.analysis import analyze_predictions
from src.evaluation.metrics import evaluate


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_EXPERIMENTS_DIR = Path("experiments")


# ============================================================================
# INCREMENTAL SAVING FUNCTIONS
# ============================================================================

def initialize_experiment_folder(
    model_name: str,
    experiment_name: str,
    prompt: str,
    questions: List[Dict]
) -> str:
    """
    Initialize experiment folder structure for incremental saving.

    Creates folder and saves initial metadata and questions.
    Results will be saved incrementally as they come in.

    Args:
        model_name: Model identifier
        experiment_name: Experiment name (without timestamp)
        prompt: System prompt or template name
        questions: List of question dictionaries

    Returns:
        Path to the experiment folder
    """
    base_folder = DEFAULT_EXPERIMENTS_DIR
    base_folder = Path(base_folder)

    # Create model-specific folder
    model_folder = base_folder / model_name
    model_folder.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment name with timestamp
    full_experiment_name = f"{experiment_name}_{timestamp}"

    # Create experiment folder
    experiment_folder = model_folder / full_experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Save questions immediately
    questions_path = experiment_folder / "questions.json"
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # Save initial metadata (will be updated at the end)
    metadata = {
        "timestamp": timestamp,
        "experiment_name": full_experiment_name,
        "model_name": model_name,
        "prompt": prompt,
        "num_questions": len(questions),
        "question_ids": [q["id"] for q in questions],
        "topics": list(set(q["topic_id"] for q in questions)),
        "num_topics": len(set(q["topic_id"] for q in questions)),
        "status": "running",
        "created_at": datetime.now().isoformat()
    }

    metadata_path = experiment_folder / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save prompt
    prompt_path = experiment_folder / "prompt.txt"
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)

    # Initialize empty results file
    results_path = experiment_folder / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "predictions": {},
            "analyses": {},
            "thinkings": {}
        }, f, indent=2, ensure_ascii=False)

    return str(experiment_folder)


def save_experiment_results_incremental(
    experiment_path: str,
    question_id: str,
    prediction: List[str],
    analysis: str,
    thinking: str = None
):
    """
    Save a single result incrementally to the experiment folder.

    This appends to the results.json file immediately, ensuring results
    are preserved even if the process crashes.

    Uses file locking to prevent race conditions when multiple processes
    write to the same experiment folder simultaneously.

    IMPORTANT: Skips saving if the result represents a retriable failure:
    - Answers containing "Fail", "N/A", "FAILED" (case-insensitive)
    - Empty or whitespace-only analyses
    - Analyses starting with "ERROR::" (API errors)

    This ensures that failed questions are retried on --resume.

    Args:
        experiment_path: Path to experiment folder
        question_id: Question ID
        prediction: Predicted answer(s)
        analysis: Analysis text
        thinking: Optional thinking text
    """
    if not experiment_path:
        return

    # ========================================================================
    # VALIDATION: Skip saving invalid/retriable results
    # ========================================================================

    # Check if prediction is a retriable failure
    if prediction:
        # Normalize prediction to list if it's a string
        pred_list = prediction if isinstance(prediction, list) else [prediction]

        # Check for failure indicators (case-insensitive)
        invalid_answers = {'fail', 'failed', 'n/a', 'na', 'error'}
        for ans in pred_list:
            if str(ans).lower().strip() in invalid_answers:
                # print(f"   ⏩ Skipping save for {question_id[:8]}: Invalid answer '{ans}'")
                return  # Don't save - will be retried on resume

    # Check if analysis is empty or contains error message
    if not analysis or not analysis.strip():
        # print(f"   ⏩ Skipping save for {question_id[:8]}: Empty analysis")
        return  # Don't save - will be retried on resume

    if analysis.strip().startswith("ERROR::"):
        # print(f"   ⏩ Skipping save for {question_id[:8]}: API error in analysis")
        return  # Don't save - will be retried on resume

    # ========================================================================
    # SAVE VALID RESULT
    # ========================================================================

    experiment_path = Path(experiment_path)
    results_path = experiment_path / "results.json"
    lock_path = experiment_path / ".results.lock"

    # Acquire exclusive file lock for thread-safe writes across processes
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Load existing results (now protected by lock)
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            results = {
                "predictions": {},
                "analyses": {},
                "thinkings": {}
            }

        # Add new result
        results["predictions"][question_id] = prediction
        results["analyses"][question_id] = analysis
        if thinking is not None:
            results["thinkings"][question_id] = thinking

        # Save immediately (atomic write)
        temp_path = results_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(results_path)

    finally:
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def finalize_experiment(
    experiment_path: str,
    results: Dict[str, Any],
    score: float,
    elapsed: float
):
    """
    Finalize experiment by updating metadata with final statistics.

    Args:
        experiment_path: Path to experiment folder
        results: Full results dictionary
        score: Final accuracy score
        elapsed: Total elapsed time
    """
    if not experiment_path:
        return

    experiment_path = Path(experiment_path)
    metadata_path = experiment_path / "metadata.json"

    # Load existing metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Update question counts from actual predictions (critical for resume scenarios)
    predictions = results.get("predictions", {})
    if predictions:
        metadata["num_questions"] = len(predictions)
        metadata["question_ids"] = sorted(predictions.keys())

        # Try to extract topics from question IDs
        # Question IDs have format q-XXXX where first digit indicates topic
        try:
            topics = list(set(int(qid.split('-')[1][0]) for qid in predictions.keys() if qid.startswith('q-')))
            if topics:
                metadata["topics"] = sorted(topics)
                metadata["num_topics"] = len(topics)
        except (ValueError, IndexError):
            # If topic extraction fails, keep existing values
            pass

    # Update with final statistics
    metadata["status"] = "completed"
    metadata["score"] = score
    metadata["elapsed_seconds"] = elapsed
    metadata["has_thinkings"] = "thinkings" in results and bool(results.get("thinkings"))
    metadata["thinking_count"] = len(results.get("thinkings", {}))
    metadata["completed_at"] = datetime.now().isoformat()

    # Save cost tracker if available
    if 'cost_tracker' in results and results['cost_tracker']:
        if hasattr(results['cost_tracker'], 'get_summary'):
            metadata["cost_summary"] = results['cost_tracker'].get_summary()
        elif isinstance(results['cost_tracker'], dict):
            metadata["cost_summary"] = results['cost_tracker']

    # Save updated metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Experiment finalized")
    print(f"   Path: {experiment_path}")
    if score is not None:
        print(f"   Score: {score:.4f}")
    else:
        print(f"   Score: N/A (test dataset has no golden answers)")
    print(f"   Time: {elapsed:.2f}s")


# ============================================================================
# SAVE EXPERIMENT - ORGANIZED BY MODEL (Legacy - for backward compatibility)
# ============================================================================

def save_experiment_results(
    results: Dict[str, Any],
    questions: List[Dict],
    model_name: str,
    prompt: str,
    experiment_name: Optional[str] = None,
    base_folder: str = None,
    score: Optional[float] = None
) -> str:
    """
    Save experiment results with organized folder structure.
    
    Structure:
        experiments/
        └── {model_name}/
            └── {experiment_name}_{timestamp}/
                ├── results.json
                ├── questions.json
                ├── metadata.json
                ├── prompt.txt
                └── dashboard.html
    
    Args:
        results: Dictionary with 'predictions', 'analyses', and optionally 'thinkings'
        questions: List of question dictionaries used in the experiment
        model_name: Model identifier (e.g., 'claude-3.5-haiku', 'llama-3.3-70b')
        prompt: System prompt or template name used
        experiment_name: Optional custom name (defaults to prompt template name)
                        If provided, will be: {experiment_name}_{timestamp}
                        If not provided, will be: {prompt}_{timestamp}
        base_folder: Base experiments directory (defaults to ./experiments)
        score: Pre-computed score (if None, will try to compute from evaluate())
    
    Returns:
        Path to the saved experiment folder
    
    Example:
        # Auto-named by prompt
        save_experiment_results(
            results, questions, "claude-3.5-haiku", "simple"
        )
        → experiments/claude-3.5-haiku/simple_20250114_190125/
        
        # Custom name
        save_experiment_results(
            results, questions, "claude-3.5-haiku", "simple",
            experiment_name="my_awesome_test"
        )
        → experiments/claude-3.5-haiku/my_awesome_test_20250114_190125/
    """
    # Default base folder
    if base_folder is None:
        base_folder = DEFAULT_EXPERIMENTS_DIR
    base_folder = Path(base_folder)

    # Create model-specific folder
    model_folder = base_folder / model_name
    model_folder.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment name
    if experiment_name is None:
        # Default: use prompt name + timestamp
        experiment_name = f"{prompt}_{timestamp}"
    else:
        # Custom name + timestamp
        experiment_name = f"{experiment_name}_{timestamp}"

    # Create experiment folder inside model folder
    experiment_folder = model_folder / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Calculate score if not provided
    if score is None:
        try:
            score = evaluate(results.get('predictions', {}), questions)
        except Exception as e:
            print(f"⚠️  Could not calculate score: {e}")
            score = None

    # Check for thinking data
    has_thinkings = "thinkings" in results and results.get("thinkings")
    thinking_count = len(results.get("thinkings", {}))

    # Save results (predictions, analyses, thinkings)
    # Convert CostTracker to dict for JSON serialization
    results_to_save = results.copy()
    if 'cost_tracker' in results_to_save and results_to_save['cost_tracker']:
        if hasattr(results_to_save['cost_tracker'], 'get_summary'):
            results_to_save['cost_tracker'] = results_to_save['cost_tracker'].get_summary()

    results_path = experiment_folder / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    # Save questions
    questions_path = experiment_folder / "questions.json"
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # Save metadata - EXACT COLAB FORMAT
    metadata = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "prompt": prompt,
        "num_questions": len(questions),
        "question_ids": [q["id"] for q in questions],
        "topics": list(set(q["topic_id"] for q in questions)),
        "num_topics": len(set(q["topic_id"] for q in questions)),
        "score": score,
        "has_thinkings": has_thinkings,
        "thinking_count": thinking_count,
        "created_at": datetime.now().isoformat()
    }

    metadata_path = experiment_folder / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save prompt separately for easy viewing
    prompt_path = experiment_folder / "prompt.txt"
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print("✅ Experiment saved")
    print(f"   Path: {experiment_folder}")
    print(f"   Questions: {len(questions)}")
    print(f"   Topics: {len(set(q['topic_id'] for q in questions))}")
    if score is not None:
        print(f"   Score: {score:.4f}")
    if has_thinkings:
        print(f"   Thinking data: {thinking_count} entries")

    return str(experiment_folder)


# ============================================================================
# LOAD EXPERIMENT
# ============================================================================

def load_experiment_results(experiment_path: str, results_file: str = "results.json") -> Dict[str, Any]:
    """
    Load a saved experiment from disk.

    Args:
        experiment_path: Path to the experiment folder
                        Can be full path or relative (model/experiment)
        results_file: Name of the results file to load (default: results.json)

    Returns:
        Dictionary with 'results', 'questions', 'metadata'
    """
    experiment_path = Path(experiment_path)

    # If not absolute, try to resolve from experiments dir
    if not experiment_path.is_absolute():
        base_path = DEFAULT_EXPERIMENTS_DIR / experiment_path
        if base_path.exists():
            experiment_path = base_path

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_path}")

    # Load results
    results_path = experiment_path / results_file
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Load questions
    questions_path = experiment_path / "questions.json"
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Normalize question IDs: ensure 'id' field exists (use 'uuid' if 'id' is missing)
    for q in questions:
        if 'id' not in q and 'uuid' in q:
            q['id'] = q['uuid']

    # Load metadata
    metadata_path = experiment_path / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return {
        'results': results,
        'questions': questions,
        'metadata': metadata,
        'path': str(experiment_path)
    }


# ============================================================================
# LIST EXPERIMENTS - WITH MODEL GROUPING
# ============================================================================

def list_experiments(base_folder: str = None, model_filter: str = None, as_dataframe: bool = True):
    """
    List all experiments in the base folder, organized by model.
    
    Args:
        base_folder: Base experiments directory (defaults to ./experiments)
        model_filter: Optional filter for specific model (e.g., "claude-3.5-haiku")
        as_dataframe: If True, return pandas DataFrame; else return list of dicts
    
    Returns:
        DataFrame or list of experiment metadata, sorted by timestamp (newest first)
    
    Example:
        # All experiments
        df = list_experiments()
        
        # Only Claude experiments
        df = list_experiments(model_filter="claude-3.5-haiku")
    """
    if base_folder is None:
        base_folder = DEFAULT_EXPERIMENTS_DIR
    base_folder = Path(base_folder)

    if not base_folder.exists():
        print(f"⚠️  Experiments folder not found: {base_folder}")
        return pd.DataFrame() if as_dataframe else []

    experiments = []

    # Iterate through model folders
    for model_folder in sorted(base_folder.iterdir()):
        if not model_folder.is_dir():
            continue

        # Skip if model filter is set and doesn't match
        if model_filter and model_folder.name != model_filter:
            continue

        # Iterate through experiments in this model folder
        for exp_folder in sorted(model_folder.iterdir()):
            if not exp_folder.is_dir():
                continue

            metadata_path = exp_folder / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Add relative path for easier access
                metadata['path'] = str(exp_folder)
                metadata['relative_path'] = f"{model_folder.name}/{exp_folder.name}"
                experiments.append(metadata)
            except Exception as e:
                print(f"⚠️  Error loading {exp_folder}: {e}")

    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    if as_dataframe:
        if not experiments:
            return pd.DataFrame()
        df = pd.DataFrame(experiments)
        # Sort and return ALL columns (including 'path')
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)
        return df

    return experiments

# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def quick_analysis_from_path(experiment_path: str):
    """Quick analysis of a saved experiment with printed summary."""
    # Load experiment
    loaded = load_experiment_results(experiment_path)

    # Print metadata
    metadata = loaded['metadata']
    print(f"\n{'='*80}")
    print(f"Experiment: {metadata.get('experiment_name', 'Unknown')}")
    print(f"{'='*80}")
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Questions: {metadata.get('num_questions', 0)}")
    print(f"Topics: {metadata.get('num_topics', 0)}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")

    # Show score
    score = metadata.get('score')
    if score is not None:
        print(f"\n📊 Score: {score:.4f}")

    # Show prediction analysis
    print(f"\n{'='*80}")
    print("📈 Prediction Analysis:")
    print(f"{'='*80}")
    try:
        analyze_predictions(loaded['results']['predictions'], loaded['questions'])
    except Exception as e:
        print(f"⚠️  Error in prediction analysis: {e}")

    return loaded


def get_latest_experiment(base_folder: str = None, model_filter: str = None) -> Optional[str]:
    """
    Get the path to the most recent experiment.
    
    Args:
        base_folder: Base experiments directory
        model_filter: Optional filter for specific model
    
    Returns:
        Path to latest experiment or None
    """
    experiments = list_experiments(base_folder, model_filter, as_dataframe=True)

    if experiments.empty:
        print("⚠️  No experiments found")
        return None

    latest = experiments.iloc[0]
    print(f"🕒 Latest experiment: {latest['experiment_name']}")
    print(f"   Model: {latest['model_name']}")
    print(f"   Timestamp: {latest['timestamp']}")
    if 'score' in latest and pd.notna(latest['score']):
        print(f"   Score: {latest['score']:.4f}")

    return latest['path']

def compare_experiments(
    patterns: List[str],
    output_html: Optional[str] = None,
    output_csv: Optional[str] = None,
    base_folder: str = None,
    show_details: bool = True
) -> Dict[str, Any]:
    """
    Detailed comparison using YOUR evaluate() and analyze_predictions().
    """
    
    if base_folder is None:
        base_folder = Path("experiments")
    base_folder = Path(base_folder)
    
    # Find matching experiments
    matched_paths = []
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for model_folder in sorted(base_folder.iterdir()):
            if not model_folder.is_dir():
                continue
            for exp_folder in sorted(model_folder.iterdir()):
                if not exp_folder.is_dir():
                    continue
                exp_name = exp_folder.name.lower()
                full_path = f"{model_folder.name}/{exp_folder.name}".lower()
                if pattern_lower in exp_name or pattern_lower in full_path:
                    matched_paths.append(str(exp_folder))
                    print(f"✓ Matched '{pattern}' → {model_folder.name}/{exp_folder.name}")
    
    if not matched_paths:
        print("❌ No matching experiments found")
        return {}
    
    # Remove duplicates
    matched_paths = list(dict.fromkeys(matched_paths))
    
    print(f"\n📊 Loading {len(matched_paths)} experiments...\n")
    
    # Load all experiments
    experiments = []
    for path in matched_paths:
        try:
            exp = load_experiment_results(path)
            experiments.append(exp)
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
    
    if not experiments:
        return {}
    
    # ==================================================================
    # 1. SUMMARY COMPARISON - Use YOUR evaluate()
    # ==================================================================
    summary_data = []
    for exp in experiments:
        meta = exp['metadata']
        preds = exp['results']['predictions']
        questions = exp['questions']
        
        # Use YOUR evaluate() function directly
        score = evaluate(preds, questions)
        
        # Count predictions
        pred_counts = defaultdict(int)
        for pred_list in preds.values():
            for p in pred_list:
                pred_counts[p] += 1
        
        summary_data.append({
            'experiment': meta['experiment_name'],
            'model': meta['model_name'],
            'prompt': meta['prompt'],
            'score': score,
            'questions': meta['num_questions'],
            'pred_A': pred_counts['A'],
            'pred_B': pred_counts['B'],
            'pred_C': pred_counts['C'],
            'pred_D': pred_counts['D'],
            'has_thinking': meta.get('has_thinkings', False)
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('score', ascending=False)
    
    # ==================================================================
    # 2. PER-QUESTION COMPARISON - Use YOUR evaluate() logic
    # ==================================================================
    questions = experiments[0]['questions']
    per_question_data = []
    
    for q in questions:
        id = q['id']
        row = {
            'id': id[:12],
            'topic_id': q['topic_id'],
            'target_event': q['target_event'][:80] + '...' if len(q['target_event']) > 80 else q['target_event'],
            'golden_answer': q.get('golden_answer', '?')
        }
        
        # Golden answer
        golden = q.get('golden_answer', '')
        if isinstance(golden, str):
            golden_set = set(golden.split(',')) if ',' in golden else {golden}
        elif isinstance(golden, list):
            golden_set = set(golden)
        else:
            golden_set = {str(golden)}
        
        # Remove empty strings
        golden_set = {g.strip() for g in golden_set if g and g.strip()}
        
        # Add predictions and scores from each experiment
        for i, exp in enumerate(experiments):
            pred = exp['results']['predictions'].get(id, [])
            pred_set = set(pred)
            
            row[f'pred_{i+1}'] = ','.join(sorted(pred))
            
            # Calculate score using YOUR evaluate() logic
            # (Exact match from your evaluate function)
            row[f'correct_{i+1}'] = (pred_set == golden_set)
        
        # Check agreement
        all_preds = [tuple(sorted(exp['results']['predictions'].get(id, [])))
                     for exp in experiments]
        row['all_agree'] = len(set(all_preds)) == 1
        
        per_question_data.append(row)
    
    per_question_df = pd.DataFrame(per_question_data)
    
    # ==================================================================
    # 3. AGREEMENT ANALYSIS
    # ==================================================================
    agreement_matrix = []
    for i, exp1 in enumerate(experiments):
        row = {'experiment': exp1['metadata']['experiment_name'][:30] + '...'}
        for j, exp2 in enumerate(experiments):
            if i == j:
                row[f'exp_{j+1}'] = 1.0
            else:
                preds1 = exp1['results']['predictions']
                preds2 = exp2['results']['predictions']
                
                agree_count = sum(
                    1 for qid in preds1
                    if tuple(sorted(preds1[qid])) == tuple(sorted(preds2.get(qid, [])))
                )
                row[f'exp_{j+1}'] = round(agree_count / len(preds1), 3) if preds1 else 0.0
        
        agreement_matrix.append(row)
    
    agreement_df = pd.DataFrame(agreement_matrix)
    
    # ==================================================================
    # 4. DIFFERENCE ANALYSIS
    # ==================================================================
    differences = per_question_df[~per_question_df['all_agree']].copy()
    
    # ==================================================================
    # 5. PER-EXPERIMENT ANALYSIS - Use YOUR analyze_predictions()
    # ==================================================================
    analysis_results = []

    for i, exp in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR: {exp['metadata']['experiment_name']}")
        print(f"{'='*80}")
        
        # Call YOUR analyze_predictions function
        analyze_predictions(exp['results']['predictions'], exp['questions'])
        
        # Calculate per-question scores using YOUR evaluate() logic
        preds = exp['results']['predictions']
        questions_exp = exp['questions']
        
        perfect_count = 0    # Exact matches (score = 1.0)
        partial_count = 0    # Partial credit (0 < score < 1)
        zero_count = 0       # Completely wrong (score = 0)
        
        for q in questions_exp:
            id = q['id']
            
            # Golden answer
            golden = q.get('golden_answer', '')
            if isinstance(golden, str):
                golden_set = set(golden.split(',')) if ',' in golden else {golden}
            elif isinstance(golden, list):
                golden_set = set(golden)
            else:
                golden_set = {str(golden)}
            golden_set = {g.strip() for g in golden_set if g and g.strip()}
            
            # Prediction
            pred = preds.get(id, [])
            pred_set = set(pred)
            
            # Calculate question score (same logic as YOUR evaluate)
            if pred_set == golden_set:
                perfect_count += 1
            elif pred_set & golden_set:  # Has intersection (partial credit)
                partial_count += 1
            else:
                zero_count += 1
        
        total = len(questions_exp)
        
        analysis_results.append({
            'experiment': exp['metadata']['experiment_name'],
            'score': summary_data[i]['score'],  # YOUR overall evaluate() score
            'perfect': perfect_count,            # Exact matches
            'partial': partial_count,            # Partial credit
            'zero': zero_count,                  # Completely wrong
            'total': total
        })

    analysis_df = pd.DataFrame(analysis_results).sort_values('score', ascending=False)


    
    # ==================================================================
    # 6. SAMPLE ANALYSES COMPARISON
    # ==================================================================
    sample_comparisons = []
    sample_ids = list(per_question_df['id'])[:5]
    
    for short_uuid in sample_ids:
        # Guard against None values - use .get() instead of direct access
        full_uuid = next((q['id'] for q in questions 
                          if q.get('id') and q['id'].startswith(short_uuid)), 
                         short_uuid)
        comp = {
            'id': short_uuid,
            # Safely extract target_event with empty string fallback
            'target': next((q['target_event'][:60] for q in questions 
                           if q.get('id') and q['id'].startswith(short_uuid)), 
                          '')
        }
        
        for i, exp in enumerate(experiments):
            analysis = exp['results']['analyses'].get(full_uuid, '')
            comp[f'analysis_{i+1}'] = (analysis[:150] + '...') if len(analysis) > 150 else analysis
        
        sample_comparisons.append(comp)
    
    sample_df = pd.DataFrame(sample_comparisons)
    
    # ==================================================================
    # PRINT SUMMARY
    # ==================================================================
    if show_details:
        print(f"\n{'='*80}")
        print("📊 OVERALL SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("🤝 AGREEMENT MATRIX")
        print(f"{'='*80}")
        print(agreement_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("📈 SCORE BREAKDOWN")
        print(f"{'='*80}")
        print(analysis_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print(f"⚠️  DISAGREEMENTS: {len(differences)} / {len(per_question_df)} questions")
        print(f"{'='*80}")
        if len(differences) > 0:
            cols = ['id', 'target_event', 'golden_answer'] + [f'pred_{i+1}' for i in range(len(experiments))]
            available_cols = [c for c in cols if c in differences.columns]
            print(differences[available_cols].head(10).to_string(index=False))
    
    # ==================================================================
    # SAVE OUTPUTS
    # ==================================================================
    if output_csv:
        base_name = output_csv.replace('.csv', '')
        
        summary_df.to_csv(f"{base_name}_summary.csv", index=False)
        per_question_df.to_csv(f"{base_name}_per_question.csv", index=False)
        differences.to_csv(f"{base_name}_differences.csv", index=False)
        analysis_df.to_csv(f"{base_name}_analysis.csv", index=False)
        
        print(f"\n✅ Saved 4 CSV files: {base_name}_*.csv")
    
    if output_html:
        print(f"\n📝 Generating HTML report...")
        try:
            generate_comparison_html(
                summary_df, per_question_df, agreement_df, analysis_df,
                differences, sample_df, experiments, output_html
            )
        except Exception as e:
            print(f"❌ HTML generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    return {
        'summary': summary_df,
        'per_question': per_question_df,
        'agreement': agreement_df,
        'analysis': analysis_df,
        'differences': differences,
        'sample_analyses': sample_df,
        'experiments': experiments
    }


def generate_comparison_html(summary_df, per_question_df, agreement_df, 
                            error_df, differences, sample_df, experiments, output_path):
    """Generate detailed HTML comparison report with nice styling."""
    
    # Extract experiment names for headers
    exp_names = [exp['metadata']['experiment_name'] for exp in experiments]
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Comparison Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 5px solid #667eea;
        }}
        
        .metadata {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .metadata-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .metadata-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .metadata-value {{
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
            transition: background 0.3s ease;
        }}
        
        tbody tr:nth-child(even) {{
            background: #fafbfc;
        }}
        
        .score-high {{
            background: #d4edda;
            color: #155724;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
        }}
        
        .score-medium {{
            background: #fff3cd;
            color: #856404;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
        }}
        
        .score-low {{
            background: #f8d7da;
            color: #721c24;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
        }}
        
        .agree {{
            background: #d1ecf1;
            color: #0c5460;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        
        .disagree {{
            background: #fff3cd;
            color: #856404;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        
        .analysis-box {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.6;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .highlight {{
            background: #fffbcc;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        
        .experiment-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Experiment Comparison Report</h1>
        
        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">Generated</span>
                <span class="metadata-value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Experiments</span>
                <span class="metadata-value">{len(experiments)}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Total Questions</span>
                <span class="metadata-value">{len(per_question_df)}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Disagreements</span>
                <span class="metadata-value">{len(differences)}</span>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Summary</h2>
            {summary_df.to_html(index=False, classes='data', escape=False, border=0)}
        </div>
        
        <div class="section">
            <h2>🤝 Agreement Matrix</h2>
            <p style="color: #7f8c8d; margin-bottom: 15px;">
                Shows pairwise agreement between experiments (1.0 = perfect agreement, 0.0 = complete disagreement)
            </p>
            {agreement_df.to_html(index=False, classes='data', border=0)}
        </div>
        
        <div class="section">
            <h2>📈 Score Breakdown</h2>
            {error_df.to_html(index=False, classes='data', escape=False, border=0)}
        </div>
        
        <div class="section">
            <h2>⚠️ Disagreements ({len(differences)} questions)</h2>
            <p style="color: #7f8c8d; margin-bottom: 15px;">
                Questions where experiments gave different predictions
            </p>
            {differences.head(30).to_html(index=False, classes='data', escape=False, border=0)}
        </div>
        
        <div class="section">
            <h2>📝 Sample Analysis Comparison</h2>
            <p style="color: #7f8c8d; margin-bottom: 15px;">
                Comparison of analysis text for the first 5 questions
            </p>
            {sample_df.to_html(index=False, classes='data', escape=False, border=0)}
        </div>
        
        <div style="text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.9em;">
            <p>Generated by Abductive Event Reasoning System</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML report saved: {output_path}")

        
# def compare_experiments(
#     patterns: List[str], 
#     output_csv: Optional[str] = None,
#     base_folder: str = None
# ) -> pd.DataFrame:
#     """
#     Compare multiple experiments side by side using pattern matching.
    
#     Args:
#         patterns: List of experiment name patterns to match (e.g., ['claude', 'zeroshot'])
#                  Can be full paths or patterns like 'claude-sonnet-4.5_fewshot'
#         output_csv: Optional path to save comparison CSV
#         base_folder: Base experiments directory (defaults to ./experiments)
    
#     Returns:
#         DataFrame with comparison data
    
#     Examples:
#         # Compare by pattern
#         compare_experiments(['zeroshot', 'fewshot', 'fewshot_cot'])
        
#         # Compare specific experiments
#         compare_experiments(['claude-sonnet-4.5_zeroshot_20250119', 'claude-sonnet-4.5_fewshot_20250119'])
        
#         # Compare all experiments from a model
#         compare_experiments(['claude-sonnet-4.5'])
#     """
#     if base_folder is None:
#         base_folder = DEFAULT_EXPERIMENTS_DIR
#     base_folder = Path(base_folder)
    
#     if not base_folder.exists():
#         print(f"⚠️  Experiments folder not found: {base_folder}")
#         return pd.DataFrame()
    
#     # Find matching experiments for each pattern
#     matched_paths = []
    
#     for pattern in patterns:
#         pattern_lower = pattern.lower()
#         found = False
        
#         # Search through all model folders
#         for model_folder in sorted(base_folder.iterdir()):
#             if not model_folder.is_dir():
#                 continue
            
#             # Search through experiments in this model folder
#             for exp_folder in sorted(model_folder.iterdir()):
#                 if not exp_folder.is_dir():
#                     continue
                
#                 # Check if pattern matches experiment name or full path
#                 exp_name = exp_folder.name.lower()
#                 full_path = f"{model_folder.name}/{exp_folder.name}".lower()
                
#                 if pattern_lower in exp_name or pattern_lower in full_path:
#                     matched_paths.append(str(exp_folder))
#                     found = True
#                     print(f"✓ Matched '{pattern}' → {model_folder.name}/{exp_folder.name}")
        
#         if not found:
#             print(f"⚠️  No experiments found matching pattern: '{pattern}'")
    
#     if not matched_paths:
#         print("❌ No matching experiments found")
#         return pd.DataFrame()
    
#     print(f"\n📊 Comparing {len(matched_paths)} experiments...\n")
    
#     # Load and compare experiments
#     comparison_data = []
    
#     for path in matched_paths:
#         try:
#             exp = load_experiment_results(path)
#             metadata = exp['metadata']
            
#             # Count prediction distribution
#             predictions = exp['results']['predictions']
#             pred_counts = {}
#             for pred_list in predictions.values():
#                 for pred in pred_list:
#                     pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
#             comparison_data.append({
#                 'model_name': metadata.get('model_name', '?'),
#                 'experiment_name': metadata.get('experiment_name', '?'),
#                 'prompt': metadata.get('prompt', '?'),
#                 'score': metadata.get('score'),
#                 'num_questions': metadata.get('num_questions', 0),
#                 'num_topics': metadata.get('num_topics', 0),
#                 'timestamp': metadata.get('timestamp', '?'),
#                 'has_thinkings': metadata.get('has_thinkings', False),
#                 'pred_A': pred_counts.get('A', 0),
#                 'pred_B': pred_counts.get('B', 0),
#                 'pred_C': pred_counts.get('C', 0),
#                 'pred_D': pred_counts.get('D', 0),
#                 'path': path
#             })
#         except Exception as e:
#             print(f"⚠️  Error loading {path}: {e}")
    
#     df = pd.DataFrame(comparison_data)
    
#     if df.empty:
#         print("❌ No valid experiments loaded")
#         return df
    
#     # Sort by score (descending)
#     if 'score' in df.columns:
#         df = df.sort_values('score', ascending=False)
    
#     # Print summary table
#     print("="*100)
#     print("COMPARISON SUMMARY")
#     print("="*100)
    
#     display_cols = ['model_name', 'prompt', 'score', 'num_questions', 'pred_A', 'pred_B', 'pred_C', 'pred_D']
#     available_cols = [col for col in display_cols if col in df.columns]
#     print(df[available_cols].to_string(index=False))
#     print("="*100)
    
#     if output_csv:
#         df.to_csv(output_csv, index=False)
#         print(f"\n✅ Comparison saved to: {output_csv}")
    
#     return df
