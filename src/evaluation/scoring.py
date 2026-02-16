"""
Centralized Scoring Utilities for Abductive Event Reasoning

Located in src/evaluation/ alongside metrics.py and analysis.py.

This module provides unified scoring functions used across all pipeline scripts:
- run_experiment.py
- optimize_prompts.py

Scoring Logic:
    - Exact match (pred == gold): 1.0 (100%)
    - Partial match (pred ⊂ gold, non-empty): 0.5 (50%)
    - Superset (pred ⊃ gold): 0.0 (0%)
    - Mismatch (pred ∩ gold = ∅ or other): 0.0 (0%)

Usage:
    from src.evaluation.scoring import (
        normalize_answer,
        calculate_match_type,
        calculate_score,
        dspy_metric
    )

    # Normalize answers
    pred_set = normalize_answer("A,B")  # {'A', 'B'}
    gold_set = normalize_answer("A, B, C")  # {'A', 'B', 'C'}

    # Calculate match type and score
    match_type, score = calculate_match_type(pred_set, gold_set)
    # ('partial', 0.5)

    # Use in DSPy evaluation
    metric_result = dspy_metric(gold_example, pred_example)
"""

from typing import Set, Tuple, Union, Dict, Any

try:
    import dspy
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    # Create a dummy Prediction class for type hints when DSPy not available
    class dspy:
        class Example:
            pass
        class Prediction:
            pass


# =====================================================================
# ANSWER NORMALIZATION
# =====================================================================

def normalize_answer(answer: Union[str, Set[str], list]) -> Set[str]:
    """
    Normalize answer to a set of uppercase letters {A, B, C, D}.

    Args:
        answer: Answer in various formats:
            - String: "A", "A,B", "A, B, C"
            - Set: {'A', 'B'}
            - List: ['A', 'B']

    Returns:
        Set of uppercase letter strings: {'A', 'B', 'C', 'D'}

    Examples:
        >>> normalize_answer("a,b")
        {'A', 'B'}
        >>> normalize_answer("A, B, C")
        {'A', 'B', 'C'}
        >>> normalize_answer(['a', 'b'])
        {'A', 'B'}
        >>> normalize_answer("none")  # Invalid
        set()
    """
    if isinstance(answer, set):
        # Already a set, just normalize to uppercase
        return {x.strip().upper() for x in answer if x and x.strip()}

    if isinstance(answer, list):
        # Convert list to set
        return {x.strip().upper() for x in answer if x and x.strip()}

    # String format
    answer = str(answer).strip().upper()

    # Handle empty or invalid answers
    if not answer:
        return set()

    # Extract valid letters A, B, C, D
    letters = set()
    for letter in ['A', 'B', 'C', 'D']:
        if letter in answer:
            letters.add(letter)

    return letters


# =====================================================================
# MATCH TYPE CALCULATION
# =====================================================================

def calculate_match_type(predicted: Set[str], gold: Set[str]) -> Tuple[str, float]:
    """
    Calculate match type and score based on set comparison.

    Args:
        predicted: Predicted answer set (e.g., {'A', 'B'})
        gold: Gold answer set (e.g., {'A', 'B', 'C'})

    Returns:
        Tuple of (match_type, score):
            - ('exact', 1.0): Perfect match
            - ('partial', 0.5): Subset match (non-empty intersection)
            - ('superset', 0.0): Predicted more than gold
            - ('mismatch', 0.0): No overlap or other mismatch

    Examples:
        >>> calculate_match_type({'A', 'B'}, {'A', 'B'})
        ('exact', 1.0)
        >>> calculate_match_type({'A'}, {'A', 'B', 'C'})
        ('partial', 0.5)
        >>> calculate_match_type({'A', 'B', 'C'}, {'A', 'B'})
        ('superset', 0.0)
        >>> calculate_match_type({'D'}, {'A', 'B'})
        ('mismatch', 0.0)
    """
    # Handle empty sets
    if not predicted and not gold:
        return 'exact', 1.0  # Both empty = exact match

    if not predicted or not gold:
        return 'mismatch', 0.0

    # Exact match
    if predicted == gold:
        return 'exact', 1.0

    # Partial match: predicted is non-empty subset of gold
    if predicted < gold:  # Proper subset (strictly less than)
        return 'partial', 0.5

    # Superset: predicted includes all gold + extra
    if predicted > gold:  # Proper superset
        return 'superset', 0.0

    # Mismatch: some overlap but neither subset nor superset
    return 'mismatch', 0.0


def calculate_score(predicted: Union[str, Set[str], list],
                   gold: Union[str, Set[str], list]) -> float:
    """
    Calculate score from answers in any format.

    Convenience wrapper around normalize_answer() + calculate_match_type().

    Args:
        predicted: Predicted answer (string, set, or list)
        gold: Gold answer (string, set, or list)

    Returns:
        Score: 1.0 (exact), 0.5 (partial), or 0.0 (incorrect)

    Examples:
        >>> calculate_score("A,B", "A,B,C")
        0.5
        >>> calculate_score("A, B", "A, B")
        1.0
        >>> calculate_score("D", "A,B,C")
        0.0
    """
    pred_set = normalize_answer(predicted)
    gold_set = normalize_answer(gold)
    _, score = calculate_match_type(pred_set, gold_set)
    return score


# =====================================================================
# DETAILED SCORING WITH ANALYSIS
# =====================================================================

def calculate_detailed_score(predicted: Union[str, Set[str], list],
                            gold: Union[str, Set[str], list]) -> Dict[str, Any]:
    """
    Calculate score with detailed analysis of what went wrong/right.

    Args:
        predicted: Predicted answer
        gold: Gold answer

    Returns:
        Dictionary with:
            - score: float (0.0, 0.5, or 1.0)
            - match_type: str ('exact', 'partial', 'superset', 'mismatch')
            - predicted_set: set of predicted letters
            - gold_set: set of gold letters
            - correct_hits: set of correctly predicted letters
            - false_positives: set of incorrectly predicted letters
            - false_negatives: set of missed letters
            - precision: float (0.0 to 1.0)
            - recall: float (0.0 to 1.0)

    Examples:
        >>> calculate_detailed_score("A,B", "A,B,C")
        {
            'score': 0.5,
            'match_type': 'partial',
            'predicted_set': {'A', 'B'},
            'gold_set': {'A', 'B', 'C'},
            'correct_hits': {'A', 'B'},
            'false_positives': set(),
            'false_negatives': {'C'},
            'precision': 1.0,
            'recall': 0.667
        }
    """
    pred_set = normalize_answer(predicted)
    gold_set = normalize_answer(gold)
    match_type, score = calculate_match_type(pred_set, gold_set)

    # Calculate precision/recall metrics
    correct_hits = pred_set & gold_set
    false_positives = pred_set - gold_set
    false_negatives = gold_set - pred_set

    precision = len(correct_hits) / len(pred_set) if pred_set else 0.0
    recall = len(correct_hits) / len(gold_set) if gold_set else 0.0

    return {
        'score': score,
        'match_type': match_type,
        'predicted_set': pred_set,
        'gold_set': gold_set,
        'correct_hits': correct_hits,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall
    }


# =====================================================================
# DSPY METRIC (for optimization with GEPA/MIPROv2)
# =====================================================================

def dspy_metric(gold: dspy.Example,
                pred: dspy.Prediction,
                trace=None) -> Union[float, dspy.Prediction]:
    """
    DSPy-compatible evaluation metric for answer scoring.

    This is the standard metric used in:
    - optimize_prompts.py (GEPA optimization)
    - Any DSPy Evaluate() calls

    Args:
        gold: DSPy Example with gold.answer
        pred: DSPy Prediction with pred.answer
        trace: Optional trace (unused, for DSPy compatibility)

    Returns:
        Float score (0.0, 0.5, or 1.0) for simple evaluation
        OR dspy.Prediction with score for optimization (GEPA/MIPRO)

    Examples:
        >>> gold = dspy.Example(answer="A,B,C").with_inputs()
        >>> pred = dspy.Prediction(answer="A,B")
        >>> dspy_metric(gold, pred)
        0.5
    """
    # Extract answers
    pred_answer = getattr(pred, "answer", "") or ""
    gold_answer = getattr(gold, "answer", "") or ""

    # Handle empty answers
    if not pred_answer or not gold_answer:
        return 0.0

    # Normalize and calculate
    try:
        pred_set = normalize_answer(pred_answer)
        gold_set = normalize_answer(gold_answer)
        _, score = calculate_match_type(pred_set, gold_set)
        return score
    except Exception:
        return 0.0


def dspy_metric_with_feedback(gold: dspy.Example,
                              pred: dspy.Prediction,
                              trace=None,
                              pred_name: str = None,
                              pred_trace=None) -> dspy.Prediction:
    """
    DSPy metric with detailed feedback for GEPA optimization.

    This version returns a dspy.Prediction with both score and feedback,
    which is used by GEPA's reflection mechanism to improve prompts.

    Args:
        gold: DSPy Example with gold.answer
        pred: DSPy Prediction with pred.answer
        trace: Optional trace
        pred_name: Name of the predictor (e.g., "answerer")
        pred_trace: Optional prediction trace

    Returns:
        dspy.Prediction with:
            - score: float (0.0, 0.5, or 1.0)
            - feedback: str (detailed explanation of match quality)

    Examples:
        >>> gold = dspy.Example(answer="A,B,C").with_inputs()
        >>> pred = dspy.Prediction(answer="A,B,D")
        >>> result = dspy_metric_with_feedback(gold, pred)
        >>> result.score
        0.0
        >>> "FALSE_POSITIVES" in result.feedback
        True
    """
    pred_answer = getattr(pred, "answer", "") or ""
    gold_answer = getattr(gold, "answer", "") or ""

    # Handle empty answers
    if not pred_answer or not gold_answer:
        return dspy.Prediction(
            score=0.0,
            feedback="[EMPTY_OUTPUT] Model failed to produce answer. "
                    "Check: Is input properly formatted? Are options clearly presented?"
        )

    try:
        # Get detailed analysis
        analysis = calculate_detailed_score(pred_answer, gold_answer)

        # Build feedback message
        feedback_parts = [
            f"[{analysis['match_type'].upper()}]",
            f"Score: {analysis['score']:.1f}",
        ]

        if analysis['predicted_set']:
            feedback_parts.append(
                f"Precision: {analysis['precision']:.0%} "
                f"({len(analysis['correct_hits'])}/{len(analysis['predicted_set'])})"
            )
        else:
            feedback_parts.append("Precision: N/A (no predictions)")

        if analysis['gold_set']:
            feedback_parts.append(
                f"Recall: {analysis['recall']:.0%} "
                f"({len(analysis['correct_hits'])}/{len(analysis['gold_set'])})"
            )
        else:
            feedback_parts.append("Recall: N/A (no gold answers)")

        # Add error details
        if analysis['false_positives']:
            feedback_parts.append(
                f"FALSE_POSITIVES: {sorted(analysis['false_positives'])} incorrectly selected"
            )
        if analysis['false_negatives']:
            feedback_parts.append(
                f"MISSED: {sorted(analysis['false_negatives'])} not selected when they should be"
            )

        # Add actionable guidance based on match type
        if analysis['match_type'] == 'superset':
            feedback_parts.append(
                "ERROR_TYPE: Over-selection - included non-causal options"
            )
            feedback_parts.append(
                "FIX: Apply necessity test - 'Would target occur without this candidate?'"
            )
        elif analysis['match_type'] == 'partial':
            feedback_parts.append(
                "ERROR_TYPE: Under-selection - missed some true causes"
            )
            feedback_parts.append(
                "FIX: Analyze ALL options exhaustively before deciding"
            )
        elif analysis['match_type'] == 'mismatch':
            feedback_parts.append(
                "ERROR_TYPE: Wrong selection - chose incorrect options"
            )
            feedback_parts.append(
                "FIX: Trace causal paths more carefully from each candidate to target"
            )

        return dspy.Prediction(
            score=analysis['score'],
            feedback=" | ".join(feedback_parts)
        )

    except Exception as e:
        return dspy.Prediction(
            score=0.0,
            feedback=f"[PARSE_ERROR] {str(e)[:100]}"
        )


# =====================================================================
# BATCH EVALUATION
# =====================================================================

def evaluate_batch(predictions: Dict[str, str],
                  questions: list,
                  question_id_key: str = "id") -> Dict[str, Any]:
    """
    Evaluate a batch of predictions against gold answers.

    Args:
        predictions: Dict mapping question_id -> predicted_answer
        questions: List of question dicts with gold answers
        question_id_key: Key to use for question ID (default: "id")

    Returns:
        Dictionary with:
            - accuracy: float (exact match accuracy)
            - avg_score: float (average score with partial credit)
            - num_total: int
            - num_exact: int
            - num_partial: int
            - num_incorrect: int
            - details: list of per-question results

    Examples:
        >>> predictions = {"q1": "A,B", "q2": "C"}
        >>> questions = [
        ...     {"id": "q1", "golden_answer": "A,B,C"},
        ...     {"id": "q2", "golden_answer": "C"}
        ... ]
        >>> results = evaluate_batch(predictions, questions)
        >>> results['accuracy']
        0.5  # 1 exact match out of 2
        >>> results['avg_score']
        0.75  # (0.5 + 1.0) / 2
    """
    num_exact = 0
    num_partial = 0
    num_incorrect = 0
    total_score = 0.0
    details = []

    for question in questions:
        question_id = question.get(question_id_key, "unknown")
        gold_answer = question.get("golden_answer", "")

        # Get prediction or empty string
        pred_answer = predictions.get(question_id, "")

        # Calculate score
        analysis = calculate_detailed_score(pred_answer, gold_answer)

        # Track counts
        if analysis['match_type'] == 'exact':
            num_exact += 1
        elif analysis['match_type'] == 'partial':
            num_partial += 1
        else:
            num_incorrect += 1

        total_score += analysis['score']

        # Store details
        details.append({
            'question_id': question_id,
            'predicted': sorted(list(analysis['predicted_set'])),
            'gold': sorted(list(analysis['gold_set'])),
            'match_type': analysis['match_type'],
            'score': analysis['score'],
            'precision': analysis['precision'],
            'recall': analysis['recall']
        })

    num_total = len(questions)
    accuracy = num_exact / num_total if num_total > 0 else 0.0
    avg_score = total_score / num_total if num_total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'avg_score': avg_score,
        'num_total': num_total,
        'num_exact': num_exact,
        'num_partial': num_partial,
        'num_incorrect': num_incorrect,
        'details': details
    }


# =====================================================================
# LEGACY COMPATIBILITY
# =====================================================================

def evaluate(preds: Dict[str, str], questions: list) -> float:
    """
    Legacy evaluation function for backward compatibility.

    This matches the original src/evaluation/metrics.py::evaluate() function.

    Args:
        preds: Dict mapping question ID -> predicted answer
        questions: List of question dicts with "id" and "golden_answer"

    Returns:
        Average score (0.0 to 1.0) with partial credit

    Examples:
        >>> preds = {"q1": "A,B"}
        >>> questions = [{"id": "q1", "golden_answer": "A,B,C"}]
        >>> evaluate(preds, questions)
        0.5
    """
    results = evaluate_batch(preds, questions, question_id_key="id")
    return results['avg_score']
