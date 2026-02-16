"""Data loading utilities for SemEval 2026 Task 12"""
import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_questions(path: Path) -> List[Dict]:
    """Load questions from JSONL file"""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def load_docs(path: Path) -> List[Dict]:
    """
    Load documents from JSON file, filtering to only essential fields.

    Keeps only: content, title, snippet, id
    Removes: imageUrl, link, source (not needed for LLM reasoning)

    This ensures no unnecessary metadata is passed to LLM contexts or stored in logs.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    
    # Filter to only essential fields
    filtered_docs = []
    for entry in raw_docs:
        filtered_entry = {
            "topic_id": entry.get("topic_id"),
            "docs": [
                {
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "snippet": doc.get("snippet", ""),
                    "id": doc.get("id", "")
                }
                for doc in entry.get("docs", [])
            ]
        }
        filtered_docs.append(filtered_entry)
    
    return filtered_docs


# NEW: Higher-level loading functions
def load_train_data(data_dir: Path = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load training questions and documents
    
    Args:
        data_dir: Path to semeval2026-task12-dataset folder
                  Defaults to "data/raw/semeval2026-task12-dataset"
    
    Returns:
        Tuple of (train_questions, train_docs)
    """
    if data_dir is None:
        data_dir = Path("data/raw/semeval2026-task12-dataset")
    
    train_questions_path = data_dir / "train_data" / "questions.jsonl"
    train_docs_path = data_dir / "train_data" / "docs.json"
    
    train_questions = load_questions(train_questions_path)
    train_docs = load_docs(train_docs_path)
    
    print(f"✓ Loaded {len(train_questions)} training questions")
    
    return train_questions, train_docs


def load_dev_data(data_dir: Path = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load development/validation questions and documents
    
    Args:
        data_dir: Path to semeval2026-task12-dataset folder
                  Defaults to "data/raw/semeval2026-task12-dataset"
    
    Returns:
        Tuple of (dev_questions, dev_docs)
    """
    if data_dir is None:
        data_dir = Path("data/raw/semeval2026-task12-dataset")
    
    dev_questions_path = data_dir / "dev_data" / "questions.jsonl"
    dev_docs_path = data_dir / "dev_data" / "docs.json"
    
    dev_questions = load_questions(dev_questions_path)
    dev_docs = load_docs(dev_docs_path)
    
    print(f"✓ Loaded {len(dev_questions)} dev questions")
    
    return dev_questions, dev_docs


def load_test_data(data_dir: Path = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load test questions and documents

    Args:
        data_dir: Path to semeval2026-task12-dataset folder
                  Defaults to "data/raw/semeval2026-task12-dataset"

    Returns:
        Tuple of (test_questions, test_docs)
    """
    if data_dir is None:
        data_dir = Path("data/raw/semeval2026-task12-dataset")

    test_questions_path = data_dir / "test_data" / "questions.jsonl"
    test_docs_path = data_dir / "test_data" / "docs.json"

    test_questions = load_questions(test_questions_path)
    test_docs = load_docs(test_docs_path)

    print(f"✓ Loaded {len(test_questions)} test questions")

    return test_questions, test_docs


def load_sample_data(data_dir: Path = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load sample questions and documents

    Args:
        data_dir: Path to semeval2026-task12-dataset folder
                  Defaults to "data/raw/semeval2026-task12-dataset"

    Returns:
        Tuple of (sample_questions, sample_docs)
    """
    if data_dir is None:
        data_dir = Path("data/raw/semeval2026-task12-dataset")

    sample_questions_path = data_dir / "sample_data" / "questions.jsonl"
    sample_docs_path = data_dir / "sample_data" / "docs.json"

    sample_questions = load_questions(sample_questions_path)
    sample_docs = load_docs(sample_docs_path)

    print(f"✓ Loaded {len(sample_questions)} sample questions")

    return sample_questions, sample_docs


def load_all_data(data_dir: Path = None) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Load both training and dev data

    Args:
        data_dir: Path to semeval2026-task12-dataset folder

    Returns:
        Tuple of (train_questions, train_docs, dev_questions, dev_docs)
    """
    train_questions, train_docs = load_train_data(data_dir)
    dev_questions, dev_docs = load_dev_data(data_dir)

    return train_questions, train_docs, dev_questions, dev_docs
