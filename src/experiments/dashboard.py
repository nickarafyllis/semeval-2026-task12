"""
HTML Dashboard Generator - EXACT COPY from Colab Cells 12-14

This includes ALL helper functions needed by create_dashboard:
- Cell 12: calculate_question_score()
- Cell 13: create_analysis_section_html()
- Cell 14: create_dashboard()

Preserved exactly as in Colab with no modifications to HTML/CSS/JS.
"""

import os
import json
from pathlib import Path
from src.experiments.manager import load_experiment_results
#from src.evaluation.metrics import evaluate

# ============================================================================
# CELL 12 - Helper Function
# ============================================================================

def calculate_question_score(prediction, ground_truth):
    """
    Calculate individual question score based on the same logic as evaluate()
    Returns: (score, status_text, status_class)

    For test datasets without ground_truth, returns N/A status.
    """
    # Handle missing ground truth (test dataset)
    if ground_truth == "N/A" or ground_truth is None:
        return None, "📝 Predicted", "predicted"

    if isinstance(prediction, list):
        pred_set = set(prediction)
    elif isinstance(prediction, str) and "," in prediction:
        pred_set = set(prediction.split(","))
    else:
        pred_set = set([prediction]) if prediction else set()

    if isinstance(ground_truth, str) and "," in ground_truth:
        gold_set = set(ground_truth.split(","))
    else:
        gold_set = set([ground_truth]) if ground_truth else set()

    if pred_set == gold_set:
        return 1, "✅ Correct", "correct"
    elif pred_set.issubset(gold_set) and len(pred_set & gold_set) > 0:
        return 0.5, "🟡 Partial", "partial"
    else:
        return 0, "❌ Incorrect", "incorrect"


# ============================================================================
# CELL 13 - Analysis Section HTML Generator
# ============================================================================

def create_analysis_section_html(preds, questions):
    """
    Create HTML section for prediction analysis - ADVANCED GLASS DESIGN
    """
    # Check if questions have golden answers (test dataset doesn't have them)
    has_golden_answers = any("golden_answer" in q for q in questions)

    if not has_golden_answers:
        # Return empty string - no analysis section for test dataset
        return ""

    # total_questions = len(questions)
    # score = evaluate(preds, questions)

    # Initialize counters (same logic as analyze_predictions function)
    multi_total = multi_more = multi_less = multi_exact = multi_partial = 0
    none_total = none_exact = none_selected_correct = none_selected_incorrect = 0
    single_total = single_exact = single_more = 0
    duplicate_total = duplicate_exact = duplicate_partial = 0

    for q in questions:
        gold = set(a.strip() for a in q["golden_answer"].split(",") if a.strip())
        pred = set(preds.get(q["id"], []))

        # Check if this is a "None" question (has a "None of the others" option)
        is_none_question = False
        none_option_letter = None
        for letter in ['A', 'B', 'C', 'D']:
            option_text = q.get(f"option_{letter}", "").strip()
            if option_text == "None of the others are correct causes.":
                is_none_question = True
                none_option_letter = letter
                break

        # Check if this is a duplicate options question
        is_duplicate_question = False
        option_texts = [q.get(f"option_{letter}", "").lower().strip() for letter in ['A', 'B', 'C', 'D']]
        if len(option_texts) != len(set(option_texts)):
            is_duplicate_question = True

        # None question analysis
        if is_none_question:
            none_total += 1
            # Check if model correctly predicted when None is the answer
            if none_option_letter in gold:
                if pred == gold:
                    none_exact += 1
                    none_selected_correct += 1
                elif none_option_letter in pred:
                    none_selected_correct += 1
                else:
                    pass  # Model didn't select None when it should have
            # Check if model incorrectly selected None when it's not the answer
            elif none_option_letter in pred:
                none_selected_incorrect += 1

        # Duplicate question analysis
        if is_duplicate_question:
            duplicate_total += 1
            if pred == gold:
                duplicate_exact += 1
            elif len(pred & gold) > 0 and pred.issubset(gold):
                duplicate_partial += 1

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

    # Helper for percentage
    def percent(count, base):
        return round(100.0 * count / base, 2) if base else 0.0

    # Advanced Glass Design - 4 Column Layout with Duplicate Analysis
    html_section = f"""
    <div class="glass-card mb-4">
        <div class="gradient-header">
            <h4><i class="fas fa-chart-bar icon"></i>📊 Detailed Prediction Analysis</h4>
            <p class="mb-0 opacity-90">Comprehensive breakdown of model performance patterns</p>
        </div>
        <div class="p-4">
            <!-- Clean 4 Column Layout -->
            <div class="row">
                <!-- Single Answer Details -->
                <div class="col-md-3">
                    <div class="analysis-detail-card">
                        <div class="card-header bg-success bg-opacity-20 analysis-card-header">
                            <h6 class="mb-0 analysis-card-title">
                                <i class="fas fa-check-circle icon text-success"></i>
                                Single Answer
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-list-alt icon text-muted"></i>
                                    Total:
                                </div>
                                <div class="stat-value text-info">{single_total}</div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-bullseye icon text-success"></i>
                                    Exact:
                                </div>
                                <div class="stat-value text-success">
                                    {single_exact} ({percent(single_exact, single_total)}%)
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-arrow-up icon text-warning"></i>
                                    Over:
                                </div>
                                <div class="stat-value text-warning">
                                    {single_more} ({percent(single_more, single_total)}%)
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Multi Answer Details -->
                <div class="col-md-3">
                    <div class="analysis-detail-card">
                        <div class="card-header bg-info bg-opacity-20 analysis-card-header">
                            <h6 class="mb-0 analysis-card-title">
                                <i class="fas fa-layer-group icon text-info"></i>
                                Multi Answer
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-list-alt icon text-muted"></i>
                                    Total:
                                </div>
                                <div class="stat-value text-info">{multi_total}</div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-bullseye icon text-success"></i>
                                    Exact:
                                </div>
                                <div class="stat-value text-success">
                                    {multi_exact} ({percent(multi_exact, multi_total)}%)
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-adjust icon text-warning"></i>
                                    Partial:
                                </div>
                                <div class="stat-value text-warning">
                                    {multi_partial} ({percent(multi_partial, multi_total)}%)
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-arrow-up icon text-danger"></i>
                                    Over:
                                </div>
                                <div class="stat-value text-danger">
                                    {multi_more} ({percent(multi_more, multi_total)}%)
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-arrow-down icon text-secondary"></i>
                                    Under:
                                </div>
                                <div class="stat-value text-secondary">
                                    {multi_less} ({percent(multi_less, multi_total)}%)
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- None Sufficient Details -->
                <div class="col-md-3">
                    <div class="analysis-detail-card">
                        <div class="card-header bg-warning bg-opacity-20 analysis-card-header">
                            <h6 class="mb-0 analysis-card-title">
                                <i class="fas fa-ban icon text-warning"></i>
                                None Sufficient
                            </h6>
                        </div>
                        <div class="card-body">
                            {f'''<div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-list-alt icon text-muted"></i>
                                    Total:
                                </div>
                                <div class="stat-value text-info">{none_total}</div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-bullseye icon text-success"></i>
                                    Exact:
                                </div>
                                <div class="stat-value text-success">
                                    {none_exact} ({percent(none_exact, none_total)}%)
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-check icon text-success"></i>
                                    Selected (Correct):
                                </div>
                                <div class="stat-value text-success">
                                    {none_selected_correct}
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-times icon text-danger"></i>
                                    Selected (Wrong):
                                </div>
                                <div class="stat-value text-danger">
                                    {none_selected_incorrect}
                                </div>
                            </div>
                            <div class="alert alert-info mt-2" role="alert" style="font-size: 0.75rem; padding: 0.5rem;">
                                <i class="fas fa-info-circle icon"></i>
                                <small>Questions with "None of the others" option</small>
                            </div>''' if none_total > 0 else '''<div class="text-center text-muted py-4">
                                <i class="fas fa-check-circle fa-2x opacity-50 mb-2"></i>
                                <p class="mb-0" style="font-size: 0.85rem;">No "None" questions</p>
                            </div>'''}
                        </div>
                    </div>
                </div>

                <!-- Duplicate Options Details -->
                <div class="col-md-3">
                    <div class="analysis-detail-card">
                        <div class="card-header bg-purple bg-opacity-20 analysis-card-header">
                            <h6 class="mb-0 analysis-card-title">
                                <i class="fas fa-clone icon text-purple"></i>
                                Duplicate Options
                            </h6>
                        </div>
                        <div class="card-body">
                            {f'''<div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-list-alt icon text-muted"></i>
                                    Total:
                                </div>
                                <div class="stat-value text-info">{duplicate_total}</div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-bullseye icon text-success"></i>
                                    Exact:
                                </div>
                                <div class="stat-value text-success">
                                    {duplicate_exact} ({percent(duplicate_exact, duplicate_total)}%)
                                </div>
                            </div>
                            <div class="stat-row">
                                <div class="stat-label">
                                    <i class="fas fa-adjust icon text-warning"></i>
                                    Partial:
                                </div>
                                <div class="stat-value text-warning">
                                    {duplicate_partial} ({percent(duplicate_partial, duplicate_total)}%)
                                </div>
                            </div>
                            <div class="alert alert-info mt-2" role="alert" style="font-size: 0.75rem; padding: 0.5rem;">
                                <i class="fas fa-info-circle icon"></i>
                                <small>Questions with duplicate option texts</small>
                            </div>''' if duplicate_total > 0 else '''<div class="text-center text-muted py-4">
                                <i class="fas fa-check-circle fa-2x opacity-50 mb-2"></i>
                                <p class="mb-0" style="font-size: 0.85rem;">No duplicate questions</p>
                            </div>'''}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <style>
        .analysis-detail-card {{
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .analysis-detail-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-blue);
        }}

        /* Analysis card headers */
        .analysis-card-header {{
            border-bottom: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 12px 12px 0 0;
        }}

        .analysis-card-title {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        /* Light mode specific fixes */
        [data-bs-theme="light"] .analysis-card-header {{
            background: var(--bg-secondary) !important;
            border-bottom: 1px solid var(--border-color);
        }}

        [data-bs-theme="light"] .analysis-card-title {{
            color: var(--text-primary) !important;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
        }}

        .stat-row:last-child {{
            border-bottom: none;
        }}

        .stat-label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}

        .stat-value {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
    </style>
    """

    return html_section


# ============================================================================
# CELL 14 - Main Dashboard Function
# ============================================================================

def create_dashboard(experiment_path, output_file=None, results_file="results.json"):
    """
    Create complete dashboard with light/dark mode toggle and Show Thinking functionality

    Args:
        experiment_path: Path to the experiment folder
        output_file: Optional output path for dashboard HTML
        results_file: Name of the results file to load (default: results.json)
    """

    # Load experiment data
    exp_data = load_experiment_results(experiment_path, results_file=results_file)
    results = exp_data['results']
    questions = exp_data['questions']
    metadata = exp_data['metadata']

    # Check if this is a pipeline cache experiment and load pipeline_summary.json
    experiment_path_obj = Path(experiment_path)
    pipeline_summary_path = experiment_path_obj / "pipeline_summary.json"
    pipeline_summary = None
    if pipeline_summary_path.exists():
        with open(pipeline_summary_path, 'r', encoding='utf-8') as f:
            pipeline_summary = json.load(f)

    if output_file is None:
        output_file = os.path.join(experiment_path, "dashboard.html")

    # Check if thinking data exists
    has_thinkings = "thinkings" in results and results["thinkings"]

    # Check if this is a test dataset (no golden answers)
    has_golden_answers = any("golden_answer" in q for q in questions)

    # Calculate detailed scoring stats
    total_score = 0
    correct_count = 0
    partial_count = 0
    incorrect_count = 0

    question_data = []
    for i, q in enumerate(questions):
        id = q["id"]
        prediction = results['predictions'].get(id, ["N/A"])
        analysis = results['analyses'].get(id, "No analysis available")
        thinking = results.get('thinkings', {}).get(id, "") if has_thinkings else ""
        ground_truth = q.get("golden_answer", "N/A")
        score, status_text, status_class = calculate_question_score(prediction, ground_truth)

        pred_str = ",".join(prediction) if isinstance(prediction, list) else str(prediction)

        topic_id = q.get('topic_id', 'Unknown')
        topic_display = f"Topic {topic_id}"

        question_data.append({
            'id': i + 1,
            'question_id': id,
            'topic_id': topic_id,
            'topic_name': topic_display,
            'topic_short': topic_display,
            'target_event': q['target_event'],
            'prediction': pred_str,
            'ground_truth': ground_truth,
            'status': status_text,
            'status_class': status_class,
            'analysis': analysis.replace('\n', '<br>'),
            'thinking': thinking.replace('\n', '<br>') if thinking else "",
            'has_thinking': bool(thinking),
            'option_A': q['option_A'],
            'option_B': q['option_B'],
            'option_C': q['option_C'],
            'option_D': q['option_D']
        })

        # Only accumulate scores if we have golden answers
        if score is not None:
            total_score += score
            if score == 1:
                correct_count += 1
            elif score == 0.5:
                partial_count += 1
            else:
                incorrect_count += 1

    avg_score = total_score / len(questions) if questions and has_golden_answers else None

    # Get prediction analysis HTML
    analysis_section = create_analysis_section_html(results['predictions'], questions)

    # Convert question data to JavaScript
    questions_json = json.dumps(question_data)

    # Prepare score display for templates (handle None for test datasets)
    score_display = f"{avg_score:.3f}" if avg_score is not None else "N/A"
    share_text_score = f"Score {avg_score:.3f}" if avg_score is not None else f"{len(questions)} questions processed"

    # Complete HTML template with Show Thinking functionality
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en" data-bs-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Experiment Dashboard - {metadata['experiment_name']}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            /* DARK MODE VARIABLES */
            :root[data-bs-theme="dark"] {{
                --bg-primary: #0d1117;
                --bg-secondary: #161b22;
                --bg-tertiary: #21262d;
                --border-color: #30363d;
                --text-primary: #f0f6fc;
                --text-secondary: #8b949e;
                --accent-blue: #58a6ff;
                --accent-green: #3fb950;
                --accent-yellow: #d29922;
                --accent-red: #f85149;
                --accent-purple: #a5a5ff;
                --accent-thinking: #7c3aed;
                --shadow: 0 8px 24px rgba(0,0,0,0.4);
                --shadow-hover: 0 12px 32px rgba(0,0,0,0.6);
                --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --gradient-success: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                --gradient-warning: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                --gradient-danger: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
                --gradient-thinking: linear-gradient(135deg, #a855f7 0%, #8b5cf6 100%);
                --glass-bg: rgba(22, 27, 34, 0.8);
                --glass-border: rgba(48, 54, 61, 0.8);
            }}

            /* LIGHT MODE VARIABLES - Professional & Clean */
            :root[data-bs-theme="light"] {{
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #f1f3f4;
                --border-color: #dee2e6;
                --text-primary: #212529;
                --text-secondary: #6c757d;
                --accent-blue: #0d6efd;
                --accent-green: #198754;
                --accent-yellow: #ffc107;
                --accent-red: #dc3545;
                --accent-purple: #6f42c1;
                --accent-thinking: #7c3aed;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                --shadow-hover: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
                --gradient: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
                --gradient-success: linear-gradient(135deg, #198754 0%, #20c997 100%);
                --gradient-warning: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
                --gradient-danger: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
                --gradient-thinking: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
                --glass-bg: rgba(255, 255, 255, 0.95);
                --glass-border: rgba(222, 226, 230, 0.8);
            }}

            /* Default to dark theme */
            :root {{
                --bg-primary: #0d1117;
                --bg-secondary: #161b22;
                --bg-tertiary: #21262d;
                --border-color: #30363d;
                --text-primary: #f0f6fc;
                --text-secondary: #8b949e;
                --accent-blue: #58a6ff;
                --accent-green: #3fb950;
                --accent-yellow: #d29922;
                --accent-red: #f85149;
                --accent-purple: #a5a5ff;
                --accent-thinking: #7c3aed;
                --shadow: 0 8px 24px rgba(0,0,0,0.4);
                --shadow-hover: 0 12px 32px rgba(0,0,0,0.6);
                --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --gradient-success: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                --gradient-warning: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                --gradient-danger: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
                --gradient-thinking: linear-gradient(135deg, #a855f7 0%, #8b5cf6 100%);
                --glass-bg: rgba(22, 27, 34, 0.8);
                --glass-border: rgba(48, 54, 61, 0.8);
            }}

            body {{
                background: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                transition: background-color 0.3s ease, color 0.3s ease;
            }}

            /* Light mode specific body styling */
            [data-bs-theme="light"] body {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }}

            ::-webkit-scrollbar {{ width: 8px; }}
            ::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
            ::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 4px; }}

            .glass-card {{
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                box-shadow: var(--shadow);
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.3s ease;
            }}

            .glass-card:hover {{
                transform: translateY(-2px);
                box-shadow: var(--shadow-hover);
                border-color: var(--accent-blue);
            }}

            /* Light mode specific glass card */
            [data-bs-theme="light"] .glass-card {{
                background: var(--glass-bg);
                backdrop-filter: none;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow);
            }}

            [data-bs-theme="light"] .glass-card:hover {{
                box-shadow: var(--shadow-hover);
                border-color: var(--accent-blue);
            }}

            .gradient-header {{
                background: var(--gradient);
                border-radius: 16px 16px 0 0;
                padding: 1.5rem;
                color: white;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}

            /* Light mode professional header */
            [data-bs-theme="light"] .gradient-header {{
                background: var(--gradient);
                text-shadow: none;
                color: white;
            }}

            .analysis-text {{
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-left: 4px solid var(--accent-blue);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
                line-height: 1.8;
                transition: background-color 0.3s ease, border-color 0.3s ease;
            }}

            /* NEW: Thinking text styling */
            .thinking-text {{
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-left: 4px solid var(--accent-thinking);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
                line-height: 1.8;
                transition: background-color 0.3s ease, border-color 0.3s ease;
                background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%);
            }}

            [data-bs-theme="light"] .thinking-text {{
                background: rgba(124, 58, 237, 0.08);
                border-left-color: var(--accent-thinking);
            }}

            /* NEW: Show Thinking button */
            .show-thinking-btn {{
                background: var(--gradient-thinking);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-size: 0.85rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                margin-top: 0.5rem;
            }}

            .show-thinking-btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(124, 58, 237, 0.3);
            }}

            [data-bs-theme="light"] .show-thinking-btn {{
                background: var(--gradient-thinking);
                box-shadow: 0 2px 4px rgba(124, 58, 237, 0.2);
            }}

            [data-bs-theme="light"] .show-thinking-btn:hover {{
                box-shadow: 0 4px 8px rgba(124, 58, 237, 0.3);
            }}

            /* Thinking section */
            .thinking-section {{
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border-color);
            }}

            .option-box {{
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                transition: transform 0.3s ease, border-color 0.3s ease, background-color 0.3s ease;
                font-weight: 500;
            }}

            .option-box:hover {{
                border-color: var(--accent-blue);
                transform: translateX(4px);
            }}

            [data-bs-theme="light"] .option-box:hover {{
                background: var(--bg-secondary);
                transform: translateX(2px);
            }}

            .prediction-box {{
                background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
                border: 1px solid rgba(255, 193, 7, 0.3);
                border-radius: 12px;
                padding: 1rem;
                color: var(--accent-yellow);
                font-weight: 600;
                text-align: center;
            }}

            [data-bs-theme="light"] .prediction-box {{
                background: rgba(255, 193, 7, 0.1);
                border: 2px solid var(--accent-yellow);
                color: #856404;
            }}

            .ground-truth-box {{
                background: linear-gradient(135deg, rgba(63, 185, 80, 0.1) 0%, rgba(63, 185, 80, 0.05) 100%);
                border: 1px solid rgba(63, 185, 80, 0.3);
                border-radius: 12px;
                padding: 1rem;
                color: var(--accent-green);
                font-weight: 600;
                text-align: center;
            }}

            [data-bs-theme="light"] .ground-truth-box {{
                background: rgba(25, 135, 84, 0.1);
                border: 2px solid var(--accent-green);
                color: #0f5132;
            }}

            .status-correct {{
                background: var(--gradient-success);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: 700;
                text-align: center;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }}

            [data-bs-theme="light"] .status-correct {{
                background: var(--accent-green);
                color: white;
                text-shadow: none;
            }}

            .status-partial {{
                background: var(--gradient-warning);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: 700;
                text-align: center;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }}

            [data-bs-theme="light"] .status-partial {{
                background: var(--accent-yellow);
                color: white;
                text-shadow: none;
            }}

            .status-incorrect {{
                background: var(--gradient-danger);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: 700;
                text-align: center;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }}

            [data-bs-theme="light"] .status-incorrect {{
                background: var(--accent-red);
                color: white;
                text-shadow: none;
            }}

            .status-predicted {{
                background: var(--gradient);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: 700;
                box-shadow: var(--shadow);
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}

            [data-bs-theme="light"] .status-predicted {{
                background: var(--accent-blue);
                color: white;
                text-shadow: none;
            }}

            .question-card {{
                margin-bottom: 2rem;
                transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                border-radius: 20px;
                overflow: hidden;
                animation: fadeInUp 0.6s ease-out;
                animation-fill-mode: both;
            }}

            .question-header {{
                background: linear-gradient(135deg, rgba(88, 166, 255, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%);
                border-bottom: 1px solid var(--border-color);
                padding: 1.5rem;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}

            [data-bs-theme="light"] .question-header {{
                background: rgba(13, 110, 253, 0.05);
            }}

            [data-bs-theme="light"] .question-header:hover {{
                background: rgba(13, 110, 253, 0.1);
            }}

            .sidebar {{ position: sticky; top: 20px; height: fit-content; max-height: calc(100vh - 40px); overflow-y: auto; }}
            .filter-btn {{ margin: 0.25rem; border-radius: 20px; padding: 0.5rem 1rem; font-weight: 600; transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease; }}

            .form-control, .form-select {{
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                color: var(--text-primary);
                padding: 0.75rem;
                transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            }}

            .form-control:focus, .form-select:focus {{
                background: var(--bg-secondary);
                border-color: var(--accent-blue);
                color: var(--text-primary);
                box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
            }}

            [data-bs-theme="light"] .form-control,
            [data-bs-theme="light"] .form-select {{
                background: white;
                border: 1px solid var(--border-color);
            }}

            [data-bs-theme="light"] .form-control:focus,
            [data-bs-theme="light"] .form-select:focus {{
                background: white;
                border-color: var(--accent-blue);
                box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
            }}

            .topic-badge {{
                background: linear-gradient(135deg, var(--accent-purple), #845ec2);
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            }}

            [data-bs-theme="light"] .topic-badge {{
                background: var(--accent-purple);
                color: white;
            }}

            .topic-full {{
                background: rgba(165, 165, 255, 0.1);
                border: 1px solid rgba(165, 165, 255, 0.3);
                color: var(--accent-purple);
                padding: 0.75rem 1rem;
                border-radius: 12px;
                font-size: 0.9rem;
                font-weight: 500;
                margin-bottom: 1rem;
            }}

            [data-bs-theme="light"] .topic-full {{
                background: rgba(111, 66, 193, 0.1);
                border: 1px solid rgba(111, 66, 193, 0.3);
                color: var(--accent-purple);
            }}

            .main-header {{
                background: var(--gradient);
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 2rem;
                color: white;
            }}

            .header-title {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }}

            .export-btn {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                margin: 0.25rem;
                font-weight: 600;
                transition: background-color 0.3s ease;
            }}

            .export-btn:hover {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
            }}

            /* Theme toggle button */
            .theme-toggle {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                border-radius: 12px;
                padding: 0.75rem 1rem;
                margin: 0.25rem;
                font-weight: 600;
                transition: background-color 0.3s ease;
                cursor: pointer;
            }}

            .theme-toggle:hover {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
            }}

            .expand-badge {{
                background: rgba(88, 166, 255, 0.1);
                color: var(--accent-blue);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            }}

            [data-bs-theme="light"] .expand-badge {{
                background: rgba(13, 110, 253, 0.1);
                color: var(--accent-blue);
                border: 1px solid rgba(13, 110, 253, 0.2);
            }}

            .icon {{ margin-right: 0.5rem; }}

            /* Stats card styling for light mode */
            .stats-card {{
                cursor: pointer;
                transition: transform 0.4s ease, box-shadow 0.4s ease;
                border-radius: 16px;
                background: var(--gradient);
                color: white;
                border: none !important;
                box-shadow: var(--shadow);
            }}

            .stats-card:hover {{
                transform: scale(1.05) translateY(-4px);
                box-shadow: var(--shadow-hover);
            }}

            [data-bs-theme="light"] .stats-card {{
                background: var(--gradient) !important;
                color: white !important;
                border: 2px solid var(--accent-blue) !important;
                box-shadow: var(--shadow-hover) !important;
            }}

            [data-bs-theme="light"] .stats-card h3,
            [data-bs-theme="light"] .stats-card small,
            [data-bs-theme="light"] .stats-card div {{
                color: white !important;
            }}

            [data-bs-theme="light"] .stats-card:hover {{
                border-color: var(--accent-purple) !important;
                box-shadow: 0 0.75rem 1.5rem rgba(0, 0, 0, 0.2) !important;
            }}

            .progress {{ height: 12px; border-radius: 10px; background: var(--bg-tertiary); }}

            [data-bs-theme="light"] .progress {{
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
            }}

            @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}

            /* Print Styles */
            @media print {{
                body {{ background: white !important; color: black !important; }}
                .glass-card {{ background: white !important; border: 1px solid #ddd !important; }}
                .sidebar {{ display: none; }}
                .theme-toggle {{ display: none; }}
                .export-btn {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid mt-4">
            <!-- Header with Theme Toggle -->
            <div class="main-header">
                <div class="row align-items-center">
                    <div class="col-md-6">
                        <h1 class="header-title">
                            <i class="fas fa-brain icon"></i>Experiment Dashboard
                        </h1>
                        <p class="mb-0 fs-5 opacity-90">
                            <i class="fas fa-flask icon"></i>{metadata['experiment_name']}
                        </p>
                    </div>
                    <div class="col-md-6 text-end">
                        <!-- Theme Toggle Button -->
                        <button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">
                            <i class="fas fa-sun icon"></i>Light Mode
                        </button>
                        <button class="export-btn" onclick="exportResults()">
                            <i class="fas fa-download icon"></i>Export CSV
                        </button>
                        <button class="export-btn" onclick="shareResults()">
                            <i class="fas fa-share icon"></i>Share
                        </button>
                        <button class="export-btn" onclick="printDashboard()">
                            <i class="fas fa-print icon"></i>Print
                        </button>
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Sidebar -->
                <div class="col-md-3">
                    <div class="sidebar">
                        <div class="glass-card mb-4">
                            <div class="gradient-header">
                                <h6><i class="fas fa-search icon"></i>Search & Filter</h6>
                            </div>
                            <div class="p-3">
                                <div class="mb-3">
                                    <label class="form-label fw-semibold">
                                        <i class="fas fa-magnifying-glass icon"></i>Search:
                                    </label>
                                    <input type="text" class="form-control" id="searchBox"
                                           placeholder="Search question IDs, events, analysis..." onkeyup="searchQuestions()">
                                </div>

                                <div class="mb-3">
                                    <label class="form-label fw-semibold">
                                        <i class="fas fa-filter icon"></i>Status Filter:
                                    </label><br>
                                    <button class="btn btn-success filter-btn" onclick="filterByStatus('correct')">
                                        <i class="fas fa-check icon"></i>Correct ({correct_count})
                                    </button>
                                    <button class="btn btn-warning filter-btn" onclick="filterByStatus('partial')">
                                        <i class="fas fa-minus icon"></i>Partial ({partial_count})
                                    </button>
                                    <button class="btn btn-danger filter-btn" onclick="filterByStatus('incorrect')">
                                        <i class="fas fa-times icon"></i>Incorrect ({incorrect_count})
                                    </button>
                                    <button class="btn btn-secondary filter-btn" onclick="filterByStatus('all')">
                                        <i class="fas fa-list icon"></i>Show All
                                    </button>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label fw-semibold">
                                        <i class="fas fa-tags icon"></i>Topic Filter:
                                    </label>
                                    <select class="form-select" id="topicFilter" onchange="filterByTopic()">
                                        <option value="all">All Topics</option>
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label fw-semibold">
                                        <i class="fas fa-sort icon"></i>Sort by:
                                    </label>
                                    <select class="form-select" id="sortBy" onchange="sortQuestions()">
                                        <option value="id">Question Order</option>
                                        <option value="status">Status</option>
                                        <option value="topic">Topic</option>
                                        <option value="length">Event Length</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="glass-card">
                            <div class="gradient-header">
                                <h6><i class="fas fa-chart-pie icon"></i>Performance Overview</h6>
                            </div>
                            <div class="p-3">
                                {f'''
                                <div class="progress mb-3">
                                    <div class="progress-bar bg-success" style="width: {correct_count/len(questions)*100:.0f}%"></div>
                                    <div class="progress-bar bg-warning" style="width: {partial_count/len(questions)*100:.0f}%"></div>
                                    <div class="progress-bar bg-danger" style="width: {incorrect_count/len(questions)*100:.0f}%"></div>
                                </div>
                                <div class="text-center mb-3">
                                    <span class="fs-4 fw-bold text-info">{score_display}</span>
                                    <br><small class="text-secondary">Overall Score</small>
                                </div>

                                <div style="position: relative; height: 200px; margin-top: 1rem;">
                                    <canvas id="scoreChart"></canvas>
                                </div>
                                ''' if has_golden_answers else '''
                                <div class="text-center mb-3">
                                    <span class="fs-5 fw-bold text-warning">⚠️ Test Dataset</span>
                                    <br><small class="text-secondary">No ground truth available</small>
                                    <p class="mt-3 text-secondary">Predictions generated for {len(questions)} questions</p>
                                </div>
                                '''}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Main Content -->
                <div class="col-md-9">
                    <!-- Experiment Overview -->
                    <div class="glass-card mb-4">
                        <div class="p-4">
                            <h5><i class="fas fa-info-circle icon"></i>Experiment Overview</h5>
                            <div class="row mt-3">
                                <div class="col-md-8">
                                    <div class="row">
                                        <div class="col-sm-6">
                                            {f'''<p><i class="fas fa-robot icon text-info"></i><strong>Models:</strong></p>
                                            <ul style="margin-left: 1.5rem; font-size: 0.9rem;">
                                                <li><strong>Extraction/Experts:</strong> {pipeline_summary['model']}</li>
                                                <li><strong>Judge:</strong> {pipeline_summary.get('judge_model') or pipeline_summary['model']}</li>
                                                <li><strong>Answerer:</strong> {pipeline_summary.get('answerer_model') or pipeline_summary['model']}</li>
                                            </ul>''' if pipeline_summary else f'<p><i class="fas fa-robot icon text-info"></i><strong>Model:</strong> {metadata["model_name"]}</p>'}
                                            <p><i class="fas fa-clock icon text-info"></i><strong>Timestamp:</strong> {metadata['timestamp']}</p>
                                        </div>
                                        <div class="col-sm-6">
                                            <p><i class="fas fa-question-circle icon text-info"></i><strong>Questions:</strong> {metadata['num_questions']}</p>
                                            <p><i class="fas fa-folder icon text-info"></i><strong>Topics:</strong> {metadata['num_topics']}</p>
                                            {'<p><i class="fas fa-brain icon text-purple"></i><strong>Thinkings Available:</strong> Yes</p>' if has_thinkings else ''}
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="stats-card glass-card p-3 h-100 d-flex align-items-center justify-content-center"
                                         onclick="filterByStatus('correct')">
                                        <div class="text-center">
                                            <h3 class="mb-1">{correct_count}</h3>
                                            <small>Correct ({correct_count/len(questions)*100:.1f}%)</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Analysis Section -->
                    {analysis_section}

                    <!-- Questions -->
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-list-check icon"></i>Questions <span id="questionCount" class="text-info">({len(questions)} total)</span></h2>
                        <div>
                            <button class="btn btn-outline-light filter-btn" onclick="collapseAll()">
                                <i class="fas fa-compress icon"></i>Collapse All
                            </button>
                            <button class="btn btn-outline-light filter-btn" onclick="expandAll()">
                                <i class="fas fa-expand icon"></i>Expand All
                            </button>
                            {'<button class="btn btn-outline-light filter-btn" onclick="toggleThinkingForAll()" id="thinkingToggle"><i class="fas fa-brain icon"></i>Show All Thinking</button>' if has_thinkings else ''}
                        </div>
                    </div>

                    <div id="questionsContainer">
                        <!-- Questions populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Theme Management
            let currentTheme = localStorage.getItem('theme') || 'dark';
            let thinkingVisible = false;
            const hasThinkings = {str(has_thinkings).lower()};

            function initializeTheme() {{
                document.documentElement.setAttribute('data-bs-theme', currentTheme);
                updateThemeButton();
            }}

            function toggleTheme() {{
                currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-bs-theme', currentTheme);
                localStorage.setItem('theme', currentTheme);
                updateThemeButton();

                // Recreate chart with new theme colors
                createScoreChart();
            }}

            function updateThemeButton() {{
                const button = document.getElementById('themeToggle');
                if (currentTheme === 'dark') {{
                    button.innerHTML = '<i class="fas fa-sun icon"></i>Light Mode';
                }} else {{
                    button.innerHTML = '<i class="fas fa-moon icon"></i>Dark Mode';
                }}
            }}

            // Thinking Management
            function toggleThinking(index) {{
                const thinkingSection = document.getElementById(`thinking${{index}}`);
                const button = document.getElementById(`thinkingBtn${{index}}`);

                if (thinkingSection.style.display === 'none' || thinkingSection.style.display === '') {{
                    thinkingSection.style.display = 'block';
                    button.innerHTML = '<i class="fas fa-eye-slash icon"></i>Hide Thinking';
                }} else {{
                    thinkingSection.style.display = 'none';
                    button.innerHTML = '<i class="fas fa-brain icon"></i>Show Thinking';
                }}
            }}

            function toggleThinkingForAll() {{
                thinkingVisible = !thinkingVisible;
                const toggleButton = document.getElementById('thinkingToggle');

                document.querySelectorAll('[id^="thinking"]').forEach(section => {{
                    if (section.id.startsWith('thinking') && !section.id.includes('Btn')) {{
                        section.style.display = thinkingVisible ? 'block' : 'none';
                    }}
                }});

                document.querySelectorAll('[id^="thinkingBtn"]').forEach(button => {{
                    button.innerHTML = thinkingVisible ?
                        '<i class="fas fa-eye-slash icon"></i>Hide Thinking' :
                        '<i class="fas fa-brain icon"></i>Show Thinking';
                }});

                toggleButton.innerHTML = thinkingVisible ?
                    '<i class="fas fa-eye-slash icon"></i>Hide All Thinking' :
                    '<i class="fas fa-brain icon"></i>Show All Thinking';
            }}

            // Questions Management
            const questions = {questions_json};
            const hasGoldenAnswers = {str(has_golden_answers).lower()};
            let filteredQuestions = [...questions];

            document.addEventListener('DOMContentLoaded', function() {{
                initializeTheme();
                populateTopicFilter();
                renderQuestions();
                createScoreChart();
            }});

            function populateTopicFilter() {{
                const topics = [...new Set(questions.map(q => q.topic_name))];
                const topicFilter = document.getElementById('topicFilter');
                topics.forEach(topic => {{
                    const option = document.createElement('option');
                    option.value = topic;
                    option.textContent = topic;
                    topicFilter.appendChild(option);
                }});
            }}

            function renderQuestions() {{
                const container = document.getElementById('questionsContainer');
                container.innerHTML = '';

                filteredQuestions.forEach((q, index) => {{
                    const statusClass = q.status_class === 'correct' ? 'status-correct' :
                                       q.status_class === 'partial' ? 'status-partial' :
                                       q.status_class === 'predicted' ? 'status-predicted' : 'status-incorrect';

                    // NEW: Check if thinking is available and create thinking section
                    const thinkingSection = q.has_thinking ? `
                        <div class="thinking-section" id="thinking${{index}}" style="display: none;">
                            <h6><i class="fas fa-brain icon" style="color: var(--accent-thinking);"></i>Model Thinking Process:</h6>
                            <div class="thinking-text">${{q.thinking}}</div>
                        </div>
                    ` : '';

                    const showThinkingBtn = q.has_thinking ? `
                        <button class="show-thinking-btn" onclick="toggleThinking(${{index}})" id="thinkingBtn${{index}}">
                            <i class="fas fa-brain icon"></i>Show Thinking
                        </button>
                    ` : '';

                    const card = document.createElement('div');
                    card.className = 'glass-card question-card';
                    card.setAttribute('data-status', q.status_class);
                    card.setAttribute('data-topic', q.topic_name);
                    card.style.animationDelay = `${{Math.min(index * 0.05, 1)}}s`;
                    card.innerHTML = `
                        <div class="question-header" onclick="toggleQuestion(${{index}})">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-2">
                                        <i class="fas fa-question-circle icon"></i>${{q.question_id}}
                                    </h5>
                                    <div class="topic-badge">
                                        ${{q.topic_name}}
                                    </div>
                                </div>
                                <span class="expand-badge">
                                    <i class="fas fa-chevron-down icon"></i>Click to expand
                                </span>
                            </div>
                        </div>
                        <div class="collapse" id="question${{index}}">
                            <div class="p-4">
                                <div class="topic-full">
                                    <i class="fas fa-folder-open icon"></i>
                                    <strong>Topic:</strong> ${{q.topic_name}}
                                </div>

                                <div class="row">
                                    <div class="col-md-8">
                                        <h6><i class="fas fa-bullseye icon text-warning"></i>Target Event:</h6>
                                        <p class="lead mb-4">${{q.target_event}}</p>

                                        <h6><i class="fas fa-list-ul icon text-info"></i>Options:</h6>
                                        <div class="row mt-3">
                                            <div class="col-md-6">
                                                <div class="option-box">
                                                    <strong>A)</strong> ${{q.option_A}}
                                                </div>
                                                <div class="option-box">
                                                    <strong>B)</strong> ${{q.option_B}}
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="option-box">
                                                    <strong>C)</strong> ${{q.option_C}}
                                                </div>
                                                <div class="option-box">
                                                    <strong>D)</strong> ${{q.option_D}}
                                                </div>
                                            </div>
                                        </div>

                                        <h6 class="mt-4"><i class="fas fa-microscope icon text-purple"></i>Model Analysis:</h6>
                                        <div class="analysis-text">${{q.analysis}}</div>

                                        ${{showThinkingBtn}}
                                        ${{thinkingSection}}
                                    </div>

                                    <div class="col-md-4">
                                        <div class="glass-card p-3">
                                            <h6><i class="fas fa-robot icon text-warning"></i>Model Prediction:</h6>
                                            <div class="prediction-box mb-3">
                                                <strong>${{q.prediction}}</strong>
                                            </div>

                                            ${{hasGoldenAnswers ? `
                                            <h6><i class="fas fa-check-circle icon text-success"></i>Ground Truth:</h6>
                                            <div class="ground-truth-box mb-3">
                                                <strong>${{q.ground_truth}}</strong>
                                            </div>

                                            <div class="text-center">
                                                <div class="${{statusClass}}">
                                                    <strong>${{q.status}}</strong>
                                                </div>
                                            </div>
                                            ` : `
                                            <div class="text-center mt-3">
                                                <div class="status-predicted">
                                                    <strong>${{q.status}}</strong>
                                                </div>
                                            </div>
                                            `}}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    container.appendChild(card);
                }});

                updateQuestionCount();
            }}

            function toggleQuestion(index) {{
                const element = document.getElementById(`question${{index}}`);
                const header = element.previousElementSibling.querySelector('.fa-chevron-down');

                if (element.classList.contains('show')) {{
                    element.classList.remove('show');
                    header.style.transform = 'rotate(0deg)';
                }} else {{
                    element.classList.add('show');
                    header.style.transform = 'rotate(180deg)';
                }}
            }}

            function collapseAll() {{
                document.querySelectorAll('.collapse').forEach(el => {{
                    el.classList.remove('show');
                    const header = el.previousElementSibling.querySelector('.fa-chevron-down');
                    if (header) header.style.transform = 'rotate(0deg)';
                }});
            }}

            function expandAll() {{
                document.querySelectorAll('.collapse').forEach(el => {{
                    el.classList.add('show');
                    const header = el.previousElementSibling.querySelector('.fa-chevron-down');
                    if (header) header.style.transform = 'rotate(180deg)';
                }});
            }}

            function filterByStatus(status) {{
                if (status === 'all') {{
                    filteredQuestions = [...questions];
                }} else {{
                    filteredQuestions = questions.filter(q => q.status_class === status);
                }}
                renderQuestions();
            }}

            function filterByTopic() {{
                const topic = document.getElementById('topicFilter').value;
                if (topic === 'all') {{
                    filteredQuestions = [...questions];
                }} else {{
                    filteredQuestions = questions.filter(q => q.topic_name === topic);
                }}
                renderQuestions();
            }}

            function searchQuestions() {{
                const searchTerm = document.getElementById('searchBox').value.toLowerCase();
                if (!searchTerm) {{
                    filteredQuestions = [...questions];
                }} else {{
                    filteredQuestions = questions.filter(q =>
                        q.question_id.toLowerCase().includes(searchTerm) ||
                        q.target_event.toLowerCase().includes(searchTerm) ||
                        q.analysis.toLowerCase().includes(searchTerm) ||
                        q.thinking.toLowerCase().includes(searchTerm) ||
                        q.topic_name.toLowerCase().includes(searchTerm) ||
                        q.option_A.toLowerCase().includes(searchTerm) ||
                        q.option_B.toLowerCase().includes(searchTerm) ||
                        q.option_C.toLowerCase().includes(searchTerm) ||
                        q.option_D.toLowerCase().includes(searchTerm)
                    );
                }}
                renderQuestions();
            }}

            function sortQuestions() {{
                const sortBy = document.getElementById('sortBy').value;
                filteredQuestions.sort((a, b) => {{
                    switch(sortBy) {{
                        case 'status': return a.status_class.localeCompare(b.status_class);
                        case 'topic': return a.topic_name.localeCompare(b.topic_name);
                        case 'length': return a.target_event.length - b.target_event.length;
                        default: return a.id - b.id;
                    }}
                }});
                renderQuestions();
            }}

            function updateQuestionCount() {{
                document.getElementById('questionCount').innerHTML =
                    `(${{filteredQuestions.length}} of ${{questions.length}} shown)`;
            }}

            function createScoreChart() {{
                // Skip chart creation if no golden answers (test dataset)
                if (!hasGoldenAnswers) {{
                    return;
                }}

                const ctx = document.getElementById('scoreChart').getContext('2d');

                // Clear existing chart
                if (window.scoreChartInstance) {{
                    window.scoreChartInstance.destroy();
                }}

                // Theme-appropriate colors
                const isLight = currentTheme === 'light';
                const textColor = isLight ? '#212529' : '#f0f6fc';
                const successColor = isLight ? '#198754' : '#3fb950';
                const warningColor = isLight ? '#ffc107' : '#d29922';
                const dangerColor = isLight ? '#dc3545' : '#f85149';

                window.scoreChartInstance = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Correct', 'Partial', 'Incorrect'],
                        datasets: [{{
                            data: [{correct_count}, {partial_count}, {incorrect_count}],
                            backgroundColor: [successColor, warningColor, dangerColor],
                            borderWidth: 0,
                            cutout: '70%'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{
                                    color: textColor,
                                    font: {{ size: 12 }}
                                }}
                            }}
                        }}
                    }}
                }});
            }}

            function exportResults() {{
                const csvContent = generateCSV();
                const blob = new Blob([csvContent], {{ type: 'text/csv' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'experiment_results.csv';
                a.click();
            }}

            function generateCSV() {{
                const headers = hasGoldenAnswers ?
                    ['Question', 'Topic', 'Prediction', 'Ground Truth', 'Status', 'Target Event', 'Has Thinking'] :
                    ['Question', 'Topic', 'Prediction', 'Target Event', 'Has Thinking'];
                const rows = questions.map(q => hasGoldenAnswers ? [
                    q.id, `"${{q.topic_name.replace(/"/g, '""')}}"`, q.prediction, q.ground_truth, q.status,
                    `"${{q.target_event.replace(/"/g, '""')}}"`, q.has_thinking ? 'Yes' : 'No'
                ] : [
                    q.id, `"${{q.topic_name.replace(/"/g, '""')}}"`, q.prediction,
                    `"${{q.target_event.replace(/"/g, '""')}}"`, q.has_thinking ? 'Yes' : 'No'
                ]);
                return [headers, ...rows].map(row => row.join(',')).join('\\n');
            }}

            function shareResults() {{
                if (navigator.share) {{
                    const shareText = hasGoldenAnswers ?
                        `Results from {metadata['experiment_name']}: {share_text_score}` :
                        `Predictions from {metadata['experiment_name']}: ${{questions.length}} questions processed`;
                    navigator.share({{
                        title: 'Experiment Results',
                        text: shareText,
                        url: window.location.href
                    }});
                }} else {{
                    navigator.clipboard.writeText(window.location.href);
                    alert('Dashboard URL copied to clipboard!');
                }}
            }}

            function printDashboard() {{ window.print(); }}
        </script>
    </body>
    </html>
    """

    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    return output_file
