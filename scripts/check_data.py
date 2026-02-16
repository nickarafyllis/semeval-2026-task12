#!/usr/bin/env python3
"""
Verify data loading for all available splits.

Quick sanity check that the dataset is accessible and loads correctly.

Usage:
    python scripts/check_data.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_train_data, load_dev_data, load_test_data


def main():
    data_path = PROJECT_ROOT / "data" / "raw" / "semeval2026-task12-dataset"

    if not data_path.exists():
        # Try sample data
        sample_path = PROJECT_ROOT / "data" / "sample"
        if sample_path.exists():
            print(f"Full dataset not found at {data_path}")
            print(f"Sample data available at {sample_path}")
            print(f"\nTo use sample data, run experiments with: --dataset sample")
            return
        else:
            print(f"No data found. Expected dataset at: {data_path}")
            print("Download the SemEval 2026 Task 12 dataset and place it there.")
            return

    print("=" * 60)
    print("DATA LOADING CHECK")
    print("=" * 60)

    splits = {
        "train": load_train_data,
        "dev": load_dev_data,
        "test": load_test_data,
    }

    for split_name, loader in splits.items():
        try:
            questions, docs = loader(data_path)
            num_topics = len(docs)
            total_docs = sum(len(t.get('docs', [])) for t in docs)
            print(f"\n  {split_name:>5}: {len(questions):>4} questions, {num_topics:>3} topics, {total_docs:>5} documents")
        except FileNotFoundError:
            print(f"\n  {split_name:>5}: not found (skipped)")
        except Exception as e:
            print(f"\n  {split_name:>5}: ERROR - {e}")

    # Check sample data
    sample_path = PROJECT_ROOT / "data" / "sample"
    if sample_path.exists():
        import json
        q_file = sample_path / "questions.jsonl"
        d_file = sample_path / "docs.json"
        if q_file.exists() and d_file.exists():
            with open(q_file) as f:
                sample_q = sum(1 for _ in f)
            with open(d_file) as f:
                sample_d = json.load(f)
            total_sample_docs = sum(len(t.get('docs', [])) for t in sample_d)
            print(f"\n  sample: {sample_q:>4} questions, {len(sample_d):>3} topics, {total_sample_docs:>5} documents")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
