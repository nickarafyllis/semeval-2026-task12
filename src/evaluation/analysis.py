"""
Model performance analysis utilities
"""
from typing import Dict, List, Optional
from .metrics import evaluate


def analyze_predictions(preds: Dict[str, list],
                       questions: List[Dict],
                       verbose: bool = True) -> Dict:
    """
    Analyze multi-answer prediction patterns with clear, grouped results.

    Args:
        preds: Dictionary mapping question ID to predicted answers
        questions: List of question dictionaries
        verbose: If True, print detailed analysis (default: True)

    Returns:
        Dictionary containing performance statistics:
        {
            "overall": {"total_questions": int, "score": float},
            "single_answer": {"total": int, "exact": int, "more_predictions": int, "exact_rate": float},
            "multi_answer": {"total": int, "exact": int, "partial": int, "more": int, "less": int, "exact_rate": float},
            "none_sufficient": {"total": int, "exact": int, "exact_rate": float}
        }
    """
    total_questions = len(questions)
    score = evaluate(preds, questions)

    # Initialize counters
    multi_total = multi_more = multi_less = multi_exact = multi_partial = 0
    none_total = none_exact = 0
    single_total = single_exact = single_more = 0

    for q in questions:
        gold = set(a.strip() for a in q["golden_answer"].split(",") if a.strip())
        pred = set(preds.get(q["id"], []))

        # Multi-answer questions (gold size > 1)
        if len(gold) > 1:
            multi_total += 1
            if pred == gold:
                multi_exact += 1
            elif len(pred) > len(gold):
                multi_more += 1
            elif len(pred) < len(gold):
                multi_less += 1
            if pred.issubset(gold) and len(pred & gold) > 0 and pred != gold:
                multi_partial += 1
        # Single-answer questions (gold size = 1)
        elif len(gold) == 1:
            single_total += 1
            if pred == gold:
                single_exact += 1
            elif len(pred) > 1:
                single_more += 1

        # Questions with option text "None of the others are correct causes."
        key = q["golden_answer"].strip()
        option_key = f"option_{key}"
        if q.get(option_key, "").strip() == "None of the others are correct causes.":
            none_total += 1
            if pred == gold:
                none_exact += 1

    # Helper for percentage
    def percent(count, base):
        return round(100.0 * count / base, 2) if base else 0.0

    # Build statistics dictionary
    stats = {
        "overall": {
            "total_questions": total_questions,
            "score": float(score)
        },
        "single_answer": {
            "total": single_total,
            "exact": single_exact,
            "more_predictions": single_more,
            "exact_rate": percent(single_exact, single_total) / 100.0 if single_total else 0.0
        },
        "multi_answer": {
            "total": multi_total,
            "exact": multi_exact,
            "partial": multi_partial,
            "more": multi_more,
            "less": multi_less,
            "exact_rate": percent(multi_exact, multi_total) / 100.0 if multi_total else 0.0
        },
        "none_sufficient": {
            "total": none_total,
            "exact": none_exact,
            "exact_rate": percent(none_exact, none_total) / 100.0 if none_total else 0.0
        }
    }

    # Print analysis if verbose
    if verbose:
        print("=== PREDICTION ANALYSIS ===")

        print("\n--- Overall ---")
        print(f"Total Questions: {total_questions}")
        print(f"Score: {score:.3f}")

        print("\n--- Single Answer ---")
        print(f"Total Single Answer Questions: {single_total}")
        print(f"Exact Single Answer Matches: {single_exact} ({percent(single_exact, single_total)}%)")
        print(f"More Predictions: {single_more} ({percent(single_more, single_total)}%)")

        print("\n--- Multi Answer ---")
        print(f"Total Multi Answer Questions: {multi_total}")
        print(f"Exact Matches: {multi_exact} ({percent(multi_exact, multi_total)}%)")
        print(f"Partial Matches: {multi_partial} ({percent(multi_partial, multi_total)}%)")
        print(f"More Predictions: {multi_more} ({percent(multi_more, multi_total)}%)")
        print(f"Less Predictions: {multi_less} ({percent(multi_less, multi_total)}%)")

        print("\n--- None Sufficient ---")
        print(f"Total 'None Sufficient' Questions: {none_total}")
        print(f"Exact 'None Sufficient' Matches: {none_exact} ({percent(none_exact, none_total)}%)")

    return stats


def compare_predictions(experiments: List[Dict], names: List[str], verbose: bool = True) -> Dict:
    """
    Compare performance across multiple experiments.

    Args:
        experiments: List of stats dictionaries from analyze_predictions()
        names: List of experiment names
        verbose: If True, print comparison table

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "experiments": names,
        "overall_scores": [],
        "single_answer_rates": [],
        "multi_answer_rates": [],
        "none_sufficient_rates": []
    }

    for stats in experiments:
        comparison["overall_scores"].append(stats["overall"]["score"])
        comparison["single_answer_rates"].append(stats["single_answer"]["exact_rate"])
        comparison["multi_answer_rates"].append(stats["multi_answer"]["exact_rate"])
        comparison["none_sufficient_rates"].append(stats["none_sufficient"]["exact_rate"])

    if verbose:
        print("\n=== EXPERIMENT COMPARISON ===")
        print(f"{'Experiment':<30} {'Overall':>8} {'Single':>8} {'Multi':>8} {'None':>8}")
        print("-" * 70)
        for i, name in enumerate(names):
            print(f"{name:<30} {comparison['overall_scores'][i]:>8.3f} "
                  f"{comparison['single_answer_rates'][i]:>8.1%} "
                  f"{comparison['multi_answer_rates'][i]:>8.1%} "
                  f"{comparison['none_sufficient_rates'][i]:>8.1%}")

    return comparison
