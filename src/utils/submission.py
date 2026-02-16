"""Utility functions for creating Codabench submission files."""
import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple


def create_submission_file(predictions_dict: Dict[str, str],
                          output_dir: str,
                          split: str = 'dev') -> Tuple[Path, Path]:
    """Create Codabench-compatible submission files.

    Creates:
    1. submission_{split}.jsonl - JSONL format with one prediction per line
    2. submission_{split}.zip - ZIP file containing submission.jsonl

    Args:
        predictions_dict: Dict mapping question_id -> predicted answer
                         Answer format: 'A' for single, 'A,B' for multiple (comma-separated)
        output_dir: Directory to save submission files
        split: Data split name (default: 'dev', also: 'test', 'train')

    Returns:
        Tuple of (jsonl_path, zip_path)

    Example:
        predictions = {
            'q-2020': 'C',
            'q-2021': 'A,B',
            'q-2022': 'D'
        }
        jsonl_path, zip_path = create_submission_file(predictions, 'experiments/my_run')
        # Upload submission_dev.zip to Codabench
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create JSONL format (required by Codabench scoring script)
    # Format: {"id": "q-2020", "answer": "C"} per line
    jsonl_path = output_dir / 'submission.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for qid in sorted(predictions_dict.keys()):
            answer = predictions_dict[qid]
            # Convert list format to comma-separated string if needed
            if isinstance(answer, list):
                answer = ','.join(answer) if answer else ''
            entry = {'id': qid, 'answer': answer}
            f.write(json.dumps(entry) + '\n')

    # Create ZIP file (for upload to Codabench)
    zip_path = output_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(jsonl_path, arcname='submission.jsonl')

    return jsonl_path, zip_path


def load_predictions_from_results(results_file: str) -> Dict[str, str]:
    """Load predictions dict from experiment results.json file.

    Args:
        results_file: Path to results.json file from experiment

    Returns:
        Dict mapping question_id -> predicted answer string
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = results.get('predictions', {})

    # Convert list format to comma-separated string if needed
    formatted_predictions = {}
    for qid, pred in predictions.items():
        if isinstance(pred, list):
            # Join list elements with comma
            formatted_predictions[qid] = ','.join(sorted(pred))
        else:
            formatted_predictions[qid] = pred

    return formatted_predictions
