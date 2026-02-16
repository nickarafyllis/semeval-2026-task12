"""
Evaluation metrics module.

This module now uses centralized scoring from src.evaluation.scoring.
Kept for backward compatibility with existing scripts.
"""

from .scoring import normalize_answer, calculate_match_type, evaluate_batch


def evaluate(preds, questions):
    """
    Evaluate predictions against gold answers.

    Args:
        preds: Dict mapping question ID -> predicted answer string
        questions: List of question dicts with "id" and "golden_answer"

    Returns:
        Average score (0.0 to 1.0) with partial credit (0.5 for subsets)

    Note:
        Uses centralized scoring from src.evaluation.scoring
    """
    results = evaluate_batch(preds, questions, question_id_key="id")
    return results['avg_score']


def calculate_question_score(prediction, ground_truth):
    """
    Calculate individual question score.

    Args:
        prediction: Predicted answer (string, list, or set)
        ground_truth: Gold answer (string, list, or set)

    Returns:
        Tuple of (score, status_text, status_class)

    Note:
        Uses centralized scoring from src.evaluation.scoring
    """
    # Normalize inputs
    pred_set = normalize_answer(prediction)
    gold_set = normalize_answer(ground_truth)

    # Calculate match type and score
    match_type, score = calculate_match_type(pred_set, gold_set)

    # Map to legacy output format
    if match_type == 'exact':
        return 1, "✅ Correct", "correct"
    if match_type == 'partial':
        return 0.5, "🟡 Partial", "partial"
    return 0, "❌ Incorrect", "incorrect"
