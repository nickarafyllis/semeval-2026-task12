#!/usr/bin/env python3
"""
Preprocess and save topic-wide contexts for all data splits.

This script:
1. Loads GraphRAG indices and questions for each split
2. Builds topic-wide contexts by aggregating retrievals across all questions per topic
3. Computes detailed statistics showing document reduction and cache efficiency
4. Saves preprocessed contexts for direct use in run_experiment.py

Usage:
    # Process all splits
    python scripts/preprocess_topic_wide_contexts.py --all

    # Process specific split
    python scripts/preprocess_topic_wide_contexts.py --split dev

    # Show statistics only (no save)
    python scripts/preprocess_topic_wide_contexts.py --split dev --stats-only
"""

import sys
from pathlib import Path
import argparse
import pickle
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dev_data, load_train_data, load_test_data, load_sample_data
from src.retrieval.graph_rag_utils import load_graph_rag_data, retrieve_with_graph_rag


def build_topic_wide_contexts_with_stats(questions, graph_rag_data, docs, verbose=True):
    """
    Build topic-wide contexts and return detailed statistics.

    Args:
        questions: List of question dicts
        graph_rag_data: GraphRAG data for retrieval
        docs: Original docs data (to count total docs per topic)
        verbose: Print progress

    Returns:
        tuple: (topic_contexts, stats_dict)
    """
    topic_doc_indices = defaultdict(set)
    topic_doc_objects = defaultdict(dict)

    # Count original docs per topic
    topic_original_docs = {}
    for entry in docs:
        tid = entry.get("topic_id")
        doc_list = entry.get("docs", [])
        if tid:
            topic_original_docs[tid] = len(doc_list)

    # Group questions by topic
    topic_questions = defaultdict(list)
    for q in questions:
        topic_id = q.get('topic_id')
        if topic_id:
            topic_questions[topic_id].append(q)

    # Track per-question retrievals for comparison
    per_question_retrievals = []  # List of (topic_id, question_id, doc_indices)

    if verbose:
        print(f"\n   Processing {len(topic_questions)} topics...")

    # For each topic, aggregate all retrievals
    for topic_id, topic_q_list in sorted(topic_questions.items()):
        for q in topic_q_list:
            retrieved = retrieve_with_graph_rag(q, graph_rag_data, verbose=False)

            q_doc_indices = set()
            for doc in retrieved:
                doc_idx = doc.get('_doc_idx')
                if doc_idx is not None:
                    topic_doc_indices[topic_id].add(doc_idx)
                    topic_doc_objects[topic_id][doc_idx] = doc
                    q_doc_indices.add(doc_idx)

            per_question_retrievals.append((topic_id, q.get('id'), q_doc_indices))

    # Build final contexts (sorted by doc index for cache consistency)
    topic_contexts = {}
    for topic_id in topic_doc_indices:
        sorted_indices = sorted(topic_doc_indices[topic_id])
        topic_contexts[topic_id] = [
            topic_doc_objects[topic_id][idx] for idx in sorted_indices
        ]

    # Compute statistics
    stats = compute_detailed_stats(
        questions, topic_questions, topic_contexts, per_question_retrievals, topic_original_docs
    )

    return topic_contexts, stats


def compute_detailed_stats(questions, topic_questions, topic_contexts, per_question_retrievals, topic_original_docs):
    """Compute detailed statistics comparing approaches."""

    # Per-question stats
    per_q_doc_counts = [len(indices) for _, _, indices in per_question_retrievals]
    total_per_q_docs = sum(per_q_doc_counts)
    avg_per_q_docs = total_per_q_docs / len(per_q_doc_counts) if per_q_doc_counts else 0

    # Topic-wide stats
    total_unique_docs = sum(len(docs) for docs in topic_contexts.values())
    avg_docs_per_topic = total_unique_docs / len(topic_contexts) if topic_contexts else 0

    # Deduplication ratio
    # This shows what % of per-question retrievals were duplicates
    if total_per_q_docs > 0:
        dedup_savings_pct = (total_per_q_docs - total_unique_docs) / total_per_q_docs * 100
    else:
        dedup_savings_pct = 0

    # Per-topic breakdown
    topic_stats = {}
    for topic_id, topic_q_list in topic_questions.items():
        # Original docs in this topic
        original_docs = topic_original_docs.get(topic_id, 0)

        # Per-question docs for this topic
        topic_per_q_docs = [
            len(indices) for t_id, _, indices in per_question_retrievals if t_id == topic_id
        ]
        total_topic_per_q = sum(topic_per_q_docs)

        # Topic-wide unique docs (selected by GraphRAG)
        topic_unique = len(topic_contexts.get(topic_id, []))

        # Filtering ratio: what % of original docs were selected
        if original_docs > 0:
            filtering_pct = (topic_unique / original_docs) * 100
        else:
            filtering_pct = 0

        # Deduplication ratio (across questions)
        if total_topic_per_q > 0:
            dedup_pct = (total_topic_per_q - topic_unique) / total_topic_per_q * 100
        else:
            dedup_pct = 0

        topic_stats[topic_id] = {
            'n_questions': len(topic_q_list),
            'original_docs': original_docs,
            'per_q_total': total_topic_per_q,
            'per_q_avg': total_topic_per_q / len(topic_q_list) if topic_q_list else 0,
            'unique_docs': topic_unique,
            'filtering_pct': filtering_pct,
            'dedup_pct': dedup_pct
        }

    # Total original docs
    total_original_docs = sum(topic_original_docs.values())

    # Cache efficiency (assuming prompt caching)
    total_questions = len(questions)
    total_topics = len(topic_contexts)
    cache_creates = total_topics  # First question per topic creates cache
    cache_reads = total_questions - total_topics  # Subsequent questions read from cache
    cache_hit_rate = (cache_reads / total_questions * 100) if total_questions > 0 else 0

    # Overall filtering (from original docs)
    if total_original_docs > 0:
        overall_filtering_pct = (total_unique_docs / total_original_docs) * 100
    else:
        overall_filtering_pct = 0

    return {
        'total_questions': total_questions,
        'total_topics': total_topics,
        'avg_questions_per_topic': total_questions / total_topics if total_topics > 0 else 0,

        # Original docs
        'total_original_docs': total_original_docs,

        # Per-question approach
        'per_q_total_docs': total_per_q_docs,
        'per_q_avg_docs': avg_per_q_docs,

        # Topic-wide approach
        'topic_wide_total_docs': total_unique_docs,
        'topic_wide_avg_docs': avg_docs_per_topic,

        # Reduction/Filtering
        'dedup_savings_pct': dedup_savings_pct,
        'docs_eliminated': total_per_q_docs - total_unique_docs,
        'overall_filtering_pct': overall_filtering_pct,

        # Cache efficiency
        'cache_creates': cache_creates,
        'cache_reads': cache_reads,
        'cache_hit_rate_pct': cache_hit_rate,

        # Per-topic breakdown
        'topic_stats': topic_stats
    }


def print_statistics(stats, split_name):
    """Print formatted statistics."""

    print(f"\n{'='*80}")
    print(f"STATISTICS FOR {split_name.upper()} SPLIT")
    print(f"{'='*80}")

    print(f"\n📊 OVERVIEW:")
    print(f"   Questions: {stats['total_questions']}")
    print(f"   Topics: {stats['total_topics']}")
    print(f"   Avg questions/topic: {stats['avg_questions_per_topic']:.1f}")
    print(f"   Total original documents: {stats['total_original_docs']}")

    print(f"\n📊 DOCUMENT FILTERING (GraphRAG selection):")
    print(f"   Original docs: {stats['total_original_docs']}")
    print(f"   Selected docs: {stats['topic_wide_total_docs']}")
    print(f"   Filtering: {stats['overall_filtering_pct']:.1f}% of original docs selected")
    print(f"   Removed: {stats['total_original_docs'] - stats['topic_wide_total_docs']} distractor docs filtered out")

    print(f"\n📊 DEDUPLICATION (across questions in topic):")
    print(f"   Per-question total retrievals: {stats['per_q_total_docs']} (with duplicates)")
    print(f"   After deduplication: {stats['topic_wide_total_docs']} unique")
    print(f"   Dedup savings: {stats['dedup_savings_pct']:.1f}% duplicates eliminated")

    print(f"\n💡 CACHE EFFICIENCY (with prompt caching):")
    print(f"   Cache creations: {stats['cache_creates']} (first question per topic)")
    print(f"   Cache reads: {stats['cache_reads']} (subsequent questions, ~90% cheaper)")
    print(f"   Cache hit rate: {stats['cache_hit_rate_pct']:.1f}%")

    # Estimate cost savings
    # Assume: cache creation = 100 units, cache read = 10 units (90% discount)
    baseline_cost = stats['total_questions'] * 100
    topic_wide_cost = stats['cache_creates'] * 100 + stats['cache_reads'] * 10
    if baseline_cost > 0:
        cost_savings = (baseline_cost - topic_wide_cost) / baseline_cost * 100
        print(f"\n💰 ESTIMATED COST SAVINGS:")
        print(f"   Baseline: {baseline_cost} cost units")
        print(f"   Topic-wide: {topic_wide_cost} cost units")
        print(f"   Savings: {cost_savings:.1f}%")

    # Per-topic breakdown
    print(f"\n📋 PER-TOPIC BREAKDOWN:")
    print(f"   {'Topic':<8} {'Qs':<5} {'Original':<10} {'Selected':<10} {'Filter%':<10} {'Dedup%':<10}")
    print(f"   {'-'*8} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for topic_id, ts in sorted(stats['topic_stats'].items()):
        print(f"   {topic_id:<8} {ts['n_questions']:<5} {ts['original_docs']:<10} {ts['unique_docs']:<10} {ts['filtering_pct']:<10.1f} {ts['dedup_pct']:<10.1f}")

    print(f"\n   Legend:")
    print(f"   - Original: Total docs available in topic")
    print(f"   - Selected: Docs selected by GraphRAG (union across all questions)")
    print(f"   - Filter%: % of original docs selected (lower = more filtering)")
    print(f"   - Dedup%: % of per-question retrievals that were duplicates")

    print(f"\n{'='*80}")


def process_split(split_name, data_path, output_dir, stats_only=False):
    """Process a single data split."""

    print(f"\n{'#'*80}")
    print(f"# PROCESSING {split_name.upper()} SPLIT")
    print(f"{'#'*80}")

    # Determine paths based on split
    if split_name == "test":
        graph_path = data_path / "indices" / "doc_graph_test.pkl"
        query_embeddings_path = data_path / "indices" / "query_embeddings_test.pkl"
    elif split_name == "sample":
        graph_path = data_path / "indices" / "doc_graph_sample.pkl"
        query_embeddings_path = data_path / "indices" / "query_embeddings_sample.pkl"
    elif split_name == "train":
        graph_path = data_path / "indices" / "doc_graph_train.pkl"
        query_embeddings_path = data_path / "indices" / "query_embeddings_train.pkl"
    else:  # dev
        graph_path = data_path / "indices" / "doc_graph_dev.pkl"
        query_embeddings_path = data_path / "indices" / "query_embeddings_dev.pkl"

    # Check if graph files exist
    if not graph_path.exists():
        print(f"   ⚠️  Graph file not found: {graph_path}")
        print(f"      Run: python scripts/build_document_graph.py --split {split_name}")
        return None, None

    # Load data
    print(f"\n📦 Loading {split_name} data...")
    raw_data_path = data_path / "raw" / "semeval2026-task12-dataset"

    if split_name == "dev":
        questions, docs = load_dev_data(raw_data_path)
    elif split_name == "train":
        questions, docs = load_train_data(raw_data_path)
    elif split_name == "test":
        questions, docs = load_test_data(raw_data_path)
    elif split_name == "sample":
        questions, docs = load_sample_data(raw_data_path)
    else:
        print(f"   ❌ Unknown split: {split_name}")
        return None, None

    print(f"   Loaded {len(questions)} questions")

    # Load GraphRAG data
    print(f"\n📊 Loading GraphRAG indices...")
    print(f"   Graph: {graph_path}")
    print(f"   Embeddings: {query_embeddings_path}")

    graph_rag_data = load_graph_rag_data(
        graph_path=str(graph_path),
        query_embeddings_path=str(query_embeddings_path),
        n_semantic_entry=3,
        n_lexical_entry=2,
        min_cluster_size=1,
        max_docs=None,  # Unlimited
        verbose=True
    )

    if not graph_rag_data:
        print(f"   ❌ Failed to load GraphRAG data")
        return None, None

    # Build topic-wide contexts
    print(f"\n🔄 Building topic-wide contexts...")
    topic_contexts, stats = build_topic_wide_contexts_with_stats(
        questions, graph_rag_data, docs, verbose=True
    )

    # Print statistics
    print_statistics(stats, split_name)

    if stats_only:
        print(f"\n   📊 Stats-only mode, not saving")
        return topic_contexts, stats

    # Save preprocessed contexts
    output_path = output_dir / f"topic_wide_contexts_{split_name}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving preprocessed contexts...")
    print(f"   Output: {output_path}")

    # Format contexts as strings (ready for use)
    context_strings = {}
    for topic_id, docs_list in topic_contexts.items():
        context_parts = []
        for doc in docs_list:
            content = doc.get('content', doc.get('text', ''))
            if content:
                context_parts.append(content)
        context_strings[topic_id] = "\n\n---\n\n".join(context_parts)

    preprocessed_data = {
        'topic_contexts': topic_contexts,  # Raw doc objects
        'context_strings': context_strings,  # Formatted strings ready for use
        'statistics': stats,
        'metadata': {
            'split': split_name,
            'n_questions': len(questions),
            'n_topics': len(topic_contexts),
            'graph_path': str(graph_path),
            'query_embeddings_path': str(query_embeddings_path)
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   ✅ Saved to {output_path}")

    return topic_contexts, stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess topic-wide contexts for all splits")
    parser.add_argument("--split", type=str, choices=["dev", "train", "test", "sample"],
                       help="Process specific split")
    parser.add_argument("--all", action="store_true",
                       help="Process all splits (dev, train, test, sample)")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Base data directory")
    parser.add_argument("--output-dir", type=str, default="data/indices",
                       help="Output directory for preprocessed files")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics, don't save")

    args = parser.parse_args()

    if not args.split and not args.all:
        print("Error: Specify --split <name> or --all")
        print("Usage:")
        print("  python scripts/preprocess_topic_wide_contexts.py --all")
        print("  python scripts/preprocess_topic_wide_contexts.py --split dev")
        sys.exit(1)

    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("TOPIC-WIDE CONTEXT PREPROCESSING")
    print("="*80)

    if args.all:
        splits = ["sample", "dev", "train", "test"]
    else:
        splits = [args.split]

    results = {}
    for split_name in splits:
        try:
            topic_contexts, stats = process_split(
                split_name, data_path, output_dir, args.stats_only
            )
            if stats:
                results[split_name] = stats
        except Exception as e:
            print(f"\n   ❌ Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL SPLITS")
        print(f"{'='*80}")
        print(f"\n{'Split':<10} {'Questions':<12} {'Topics':<10} {'Dedup %':<12} {'Cache Hit %':<12}")
        print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

        for split_name, stats in results.items():
            print(f"{split_name:<10} {stats['total_questions']:<12} {stats['total_topics']:<10} "
                  f"{stats['dedup_savings_pct']:<12.1f} {stats['cache_hit_rate_pct']:<12.1f}")

    print(f"\n✅ Preprocessing complete!")
    print(f"\n💡 To use preprocessed contexts:")
    print(f"   python scripts/run_experiment.py --use-graph-rag --model-family gemini --version gemini-3-flash-preview")
    print(f"\n   Preprocessed contexts will be loaded automatically from {output_dir}/")


if __name__ == "__main__":
    main()
