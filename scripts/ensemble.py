#!/usr/bin/env python3
"""
Quick ensemble of two experiment results with smart aggregation.

Strategies:
- union: Take all unique options from both models
- intersection: Only options both models agree on
- vote: Options with at least 2 votes (threshold=2)
- smart: Adaptive strategy with "None" conflict resolution

Usage: python scripts/ensemble.py exp1_full_name exp2_full_name [strategy]
Example: python scripts/ensemble.py deepseek_baseline claude_baseline smart
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your existing utilities
from src.evaluation.metrics import evaluate
from src.data.loader import load_dev_data


def find_experiment_results(exp_name: str, experiments_dir: str = "experiments") -> Path:
    """Find results.json for an experiment by searching recursively."""
    exp_path = Path(experiments_dir)
    
    # Search for the experiment folder recursively
    matching_dirs = list(exp_path.rglob(exp_name))
    
    if not matching_dirs:
        # Try partial match if exact doesn't work
        matching_dirs = [p for p in exp_path.rglob("*") if exp_name in p.name and p.is_dir()]
    
    if not matching_dirs:
        raise FileNotFoundError(f"Experiment not found: {exp_name}")
    
    # Take the first match
    exp_dir = matching_dirs[0]
    
    # Find results.json in this directory
    results_file = exp_dir / "results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"No results.json found in {exp_dir}")
    
    return results_file


def load_predictions(results_path: Path) -> Dict[str, List[str]]:
    """Load predictions from results.json."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data['predictions']


def load_score_from_results(results_path: Path, dev_questions: List[Dict]) -> float:
    """Load score from results.json, or calculate it if missing."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Try to get existing score
    score = data.get('score')
    
    if score is not None and score > 0:
        return score
    
    # Score missing or 0 → calculate it
    predictions = data.get('predictions', {})
    if predictions:
        calculated_score = evaluate(predictions, dev_questions)
        print(f"    ℹ️  Calculated score from predictions: {calculated_score:.4f}")
        return calculated_score
    
    return 0.0


# ============================================================================
# ENSEMBLE STRATEGIES
# ============================================================================

def ensemble_union(pred1: List[str], pred2: List[str]) -> List[str]:
    """Union: Take all unique options from both models."""
    return sorted(list(set(pred1 + pred2)))


def ensemble_intersection(pred1: List[str], pred2: List[str]) -> List[str]:
    """Intersection: Only options both models agree on."""
    common = sorted(list(set(pred1) & set(pred2)))
    # Fallback: if no intersection, take union
    return common if common else ensemble_union(pred1, pred2)


def ensemble_vote(pred1: List[str], pred2: List[str], threshold: int = 2) -> List[str]:
    """Vote: Option needs at least 'threshold' votes to be included."""
    votes = Counter(pred1 + pred2)
    return sorted([opt for opt, count in votes.items() if count >= threshold])


def detect_none_option(question: Dict) -> Optional[str]:
    """
    Detect which option represents "None of the above".
    Returns option letter ('A', 'B', 'C', 'D') or None.
    """
    none_phrases = [
        'none of the above',
        'none of these',
        'none of the options',
        'none of them',
        'neither of',
        'neither',
        'none'
    ]
    
    for opt_letter in ['A', 'B', 'C', 'D']:
        opt_key = f'option_{opt_letter}'
        if opt_key in question:
            opt_text = question[opt_key].lower().strip()
            if any(phrase in opt_text for phrase in none_phrases):
                return opt_letter
    
    return None


def ensemble_smart(pred1: List[str], pred2: List[str], question: Dict = None) -> List[str]:
    """
    Smart ensemble strategy:
    1. Handle "None of the above" conflicts (inspired by SC)
    2. Use union when both predict multi-answer (high confidence in multi)
    3. Use intersection when one is single-answer (more conservative)
    4. Fallback to intersection for disagreements
    
    Args:
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        question: Question dict (for None detection)
    
    Returns:
        Ensemble prediction
    """
    # ========================================================================
    # STEP 1: NONE CONFLICT RESOLUTION
    # ========================================================================
    
    if question:
        none_option = detect_none_option(question)
        
        if none_option:
            none_in_pred1 = none_option in pred1
            none_in_pred2 = none_option in pred2
            
            # Conflict Case 1: One says "None", other says something else
            if none_in_pred1 and not none_in_pred2 and len(pred2) > 0:
                # Model 1 says None, Model 2 says A/B/C
                # → Trust the evidence (Model 2)
                return sorted(pred2)
            
            elif none_in_pred2 and not none_in_pred1 and len(pred1) > 0:
                # Model 2 says None, Model 1 says A/B/C
                # → Trust the evidence (Model 1)
                return sorted(pred1)
            
            # Conflict Case 2: Both include None
            elif none_in_pred1 and none_in_pred2:
                # Both agree on "None"
                if len(pred1) == 1 and len(pred2) == 1:
                    # Both say ONLY None → accept it
                    return [none_option]
                else:
                    # One says None + others (e.g., ['A', 'D']) → conflict
                    # Remove None, keep the evidence
                    pred1_clean = [opt for opt in pred1 if opt != none_option]
                    pred2_clean = [opt for opt in pred2 if opt != none_option]
                    if pred1_clean or pred2_clean:
                        # Union of non-None options
                        return sorted(list(set(pred1_clean + pred2_clean)))
                    else:
                        # Only None left → return it
                        return [none_option]
    
    # ========================================================================
    # STEP 2: SMART AGGREGATION (no None conflict)
    # ========================================================================
    
    # Case 1: Identical predictions → just return it
    if set(pred1) == set(pred2):
        return sorted(pred1)
    
    # Case 2: Both predict single-answer
    if len(pred1) == 1 and len(pred2) == 1:
        # Different single answers → intersection with fallback
        common = list(set(pred1) & set(pred2))
        if common:
            return sorted(common)
        else:
            # No overlap → take both (might be wrong, but safer than picking one)
            return ensemble_union(pred1, pred2)
    
    # Case 3: Both predict multi-answer
    if len(pred1) > 1 and len(pred2) > 1:
        # Both think it's multi-answer → likely correct
        # Use union to capture all possibilities
        return ensemble_union(pred1, pred2)
    
    # Case 4: One single, one multi (disagreement)
    # → Single-answer model might be more confident
    # → Use intersection (conservative, avoids supersets)
    return ensemble_intersection(pred1, pred2)


def create_ensemble(
    preds1: Dict[str, List[str]], 
    preds2: Dict[str, List[str]], 
    strategy: str = 'union',
    questions: List[Dict] = None
) -> Dict[str, List[str]]:
    """
    Combine predictions from two models using specified strategy.
    
    Args:
        preds1: Predictions from model 1
        preds2: Predictions from model 2
        strategy: 'union', 'intersection', 'vote', or 'smart'
        questions: List of questions (needed for 'smart' strategy)
    
    Returns:
        Ensemble predictions
    """
    ensemble_preds = {}
    
    # Create question lookup for smart strategy
    question_dict = {}
    if questions and strategy == 'smart':
        question_dict = {q['id']: q for q in questions}
    
    for id in preds1.keys():
        if id not in preds2:
            print(f"Warning: {id} not in model 2, using model 1")
            ensemble_preds[id] = preds1[id]
            continue
        
        p1 = preds1[id]
        p2 = preds2[id]
        
        if strategy == 'union':
            ensemble_preds[id] = ensemble_union(p1, p2)
        elif strategy == 'intersection':
            ensemble_preds[id] = ensemble_intersection(p1, p2)
        elif strategy == 'vote':
            ensemble_preds[id] = ensemble_vote(p1, p2, threshold=2)
        elif strategy == 'smart':
            question = question_dict.get(id)
            ensemble_preds[id] = ensemble_smart(p1, p2, question)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use: union, intersection, vote, or smart")
    
    return ensemble_preds


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/ensemble.py <exp1_full_name> <exp2_full_name> [strategy]")
        print("\nExamples:")
        print("  python scripts/ensemble.py \\")
        print("    deepseek_v3.1_thinking_baseline_merged \\")
        print("    claude_sonnet_4.5_thinking_baseline_20251015_214640 \\")
        print("    smart")
        print("\nStrategies:")
        print("  - union: Take all unique options (high recall)")
        print("  - intersection: Only common options (high precision)")
        print("  - vote: Options with ≥2 votes (balanced)")
        print("  - smart: Adaptive with None handling (recommended)")
        sys.exit(1)
    
    exp1_name = sys.argv[1]
    exp2_name = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'smart'
    
    print(f"Ensemble Strategy: {strategy}")
    print(f"\nFinding experiment results...")
    try:
        results1_path = find_experiment_results(exp1_name)
        results2_path = find_experiment_results(exp2_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable experiments:")
        exp_path = Path("experiments")
        for model_dir in sorted(exp_path.iterdir()):
            if model_dir.is_dir():
                print(f"\n  {model_dir.name}/")
                for exp_dir in sorted(model_dir.iterdir()):
                    if exp_dir.is_dir() and (exp_dir / "results.json").exists():
                        print(f"    - {exp_dir.name}")
        sys.exit(1)
    
    print(f"  Model 1: {results1_path}")
    print(f"  Model 2: {results2_path}")
    
    print(f"\nLoading predictions...")
    preds1 = load_predictions(results1_path)
    preds2 = load_predictions(results2_path)
    
    print(f"  Model 1: {len(preds1)} predictions")
    print(f"  Model 2: {len(preds2)} predictions")
    
    # Load ground truth (needed for scoring and smart strategy)
    print(f"\nLoading ground truth...")
    dev_questions, _ = load_dev_data()
    
    print(f"\nCreating ensemble with '{strategy}' strategy...")
    ensemble_preds = create_ensemble(
        preds1, 
        preds2, 
        strategy, 
        questions=dev_questions if strategy == 'smart' else None
    )
    
    # Score using your evaluate function
    score = evaluate(ensemble_preds, dev_questions)
    
    # Load individual scores (with fallback calculation)
    print(f"\nLoading individual scores...")
    score1 = load_score_from_results(results1_path, dev_questions)
    score2 = load_score_from_results(results2_path, dev_questions)
    
    gain = score - max(score1, score2)
    gain_pct = (gain / max(score1, score2)) * 100 if max(score1, score2) > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Model 1: {score1:.4f}")
    print(f"  Model 2: {score2:.4f}")
    print(f"  Ensemble ({strategy}): {score:.4f}")
    print(f"  Gain: {gain:+.4f} ({gain_pct:+.2f}%)")
    print(f"{'='*60}\n")
    
    # Save ensemble results
    output_dir = Path("experiments") / "ensembles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a short name for the output
    exp1_short = exp1_name.split('_')[-1]  # timestamp or last part
    exp2_short = exp2_name.split('_')[-1]
    output_file = output_dir / f"ensemble_{exp1_short}_{exp2_short}_{strategy}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'predictions': ensemble_preds,
            'score': score,
            'strategy': strategy,
            'exp1_name': exp1_name,
            'exp2_name': exp2_name,
            'exp1_path': str(results1_path),
            'exp2_path': str(results2_path),
            'exp1_score': score1,
            'exp2_score': score2,
            'gain': gain,
            'gain_percentage': gain_pct
        }, f, indent=2)
    
    print(f"Saved ensemble to: {output_file}")
    
    # Prediction distribution analysis
    single_count = sum(1 for p in ensemble_preds.values() if len(p) == 1)
    multi_count = sum(1 for p in ensemble_preds.values() if len(p) > 1)
    
    # Ground truth distribution
    gt_single = sum(1 for q in dev_questions if ',' not in q['golden_answer'])
    gt_multi = sum(1 for q in dev_questions if ',' in q['golden_answer'])
    
    print(f"\nPrediction distribution:")
    print(f"  Single answer: {single_count}/{len(ensemble_preds)} ({single_count/len(ensemble_preds)*100:.1f}%) [GT: {gt_single}]")
    print(f"  Multi-answer:  {multi_count}/{len(ensemble_preds)} ({multi_count/len(ensemble_preds)*100:.1f}%) [GT: {gt_multi}]")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i, (qid, pred) in enumerate(list(ensemble_preds.items())[:3]):
        p1 = preds1.get(qid, [])
        p2 = preds2.get(qid, [])

        # Find ground truth
        gt = next((q['golden_answer'] for q in dev_questions if q['id'] == qid), "?")

        print(f"\n{qid}:")
        print(f"  Model 1:  {p1}")
        print(f"  Model 2:  {p2}")
        print(f"  Ensemble: {pred}")
        print(f"  GT:       {gt}")


if __name__ == '__main__':
    main()
