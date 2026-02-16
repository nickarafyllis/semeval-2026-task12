#!/usr/bin/env python3
"""
Build Document-Level HYBRID Graph for RAG

This script:
1. Loads documents from docs.json
2. Generates document-level embeddings using Bedrock
3. Computes BM25+ lexical similarity with entity boosting
4. Builds HYBRID graph (semantic + lexical edges)
5. Saves embeddings and graph for later use

Hybrid Formula:
    edge_weight = alpha × semantic_similarity + (1 - alpha) × lexical_similarity
    Default alpha = 0.7 (semantic-dominant)

Usage:
    python scripts/build_document_graph.py --split dev
    python scripts/build_document_graph.py --split train --alpha 0.7 --k-neighbors 5
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.document_graph_rag import (
    BedrockDocumentEmbedder,
    HybridDocumentGraph,
    DocumentSemanticGraph,
    build_document_embeddings
)


def main():
    parser = argparse.ArgumentParser(description='Build hybrid document graph for RAG')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train', 'test', 'sample'],
                        help='Data split to process')
    parser.add_argument('--data-dir', type=str, default='data/raw/semeval2026-task12-dataset',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='data/indices',
                        help='Output directory for embeddings and graph')
    parser.add_argument('--model', type=str, default='cohere.embed-v4:0',
                        help='Embedding model ID')
    parser.add_argument('--dimensions', type=int, default=1024,
                        help='Embedding dimensions')
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help='Number of k-nearest neighbors per document')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Minimum hybrid similarity for edges')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for semantic similarity (0.7 = 70%% semantic, 30%% lexical)')
    parser.add_argument('--entity-boost', type=float, default=3.0,
                        help='Entity weight multiplier for BM25+ (default 3.0)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation, use existing file')
    parser.add_argument('--semantic-only', action='store_true',
                        help='Build semantic-only graph (no lexical, for comparison)')

    args = parser.parse_args()

    # Paths
    docs_path = Path(args.data_dir) / f'{args.split}_data' / 'docs.json'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / f'doc_embeddings_{args.split}.pkl'
    
    # Use different filename for hybrid vs semantic-only
    if args.semantic_only:
        graph_path = output_dir / f'doc_graph_{args.split}_semantic.pkl'
    else:
        graph_path = output_dir / f'doc_graph_{args.split}.pkl'

    print("=" * 80)
    print("HYBRID DOCUMENT GRAPH BUILDER")
    print("=" * 80)
    print(f"\n   Split: {args.split}")
    print(f"   Docs path: {docs_path}")
    print(f"   Model: {args.model}")
    print(f"   k-neighbors: {args.k_neighbors}")
    print(f"   Threshold: {args.threshold}")
    
    if args.semantic_only:
        print(f"   Mode: SEMANTIC ONLY (no lexical)")
    else:
        print(f"   Alpha: {args.alpha} (semantic={args.alpha:.0%}, lexical={1-args.alpha:.0%})")
        print(f"   Entity boost: {args.entity_boost}")

    # Load docs
    print("\n Loading documents...")
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    total_docs = sum(len(topic.get('docs', [])) for topic in docs)
    print(f"   Topics: {len(docs)}")
    print(f"   Total documents: {total_docs}")

    # Build embeddings
    if not args.skip_embeddings:
        print("\n Generating document embeddings...")
        start_time = time.time()

        build_document_embeddings(
            docs_path=str(docs_path),
            output_path=str(embeddings_path),
            model_id=args.model,
            dimensions=args.dimensions
        )

        elapsed = time.time() - start_time
        print(f"\n   Embedding time: {elapsed:.1f}s")
    else:
        print(f"\n   Skipping embeddings, using: {embeddings_path}")

    # Load embeddings
    print(f"\n Loading embeddings...")
    with open(embeddings_path, 'rb') as f:
        all_embeddings = pickle.load(f)

    # Build graphs per topic
    print(f"\n{'='*60}")
    if args.semantic_only:
        print("BUILDING SEMANTIC-ONLY GRAPHS")
    else:
        print("BUILDING HYBRID GRAPHS (Semantic + Lexical)")
    print(f"{'='*60}")
    print(f"   k_neighbors: {args.k_neighbors}")
    print(f"   similarity_threshold: {args.threshold}")
    if not args.semantic_only:
        print(f"   alpha: {args.alpha}")
        print(f"   entity_boost: {args.entity_boost}")
    
    start_time = time.time()

    all_graphs = {}
    total_edges = 0
    total_topics = len([t for t in docs if t.get('docs') and t['topic_id'] in all_embeddings])

    for i, topic in enumerate(docs):
        topic_id = topic['topic_id']
        topic_docs = topic.get('docs', [])

        if not topic_docs or topic_id not in all_embeddings:
            continue

        print(f"\n[Graph {i+1}/{total_topics}] Topic {topic_id}: {topic['topic'][:50]}...", flush=True)
        print(f"   Docs: {len(topic_docs)}", flush=True)

        # Build graph for this topic
        if args.semantic_only:
            # Semantic-only graph (original behavior)
            graph = DocumentSemanticGraph(
                k_neighbors=args.k_neighbors,
                similarity_threshold=args.threshold,
                max_hops=2
            )
            graph.build_graph(
                docs=topic_docs,
                embedder=None,
                precomputed_embeddings=all_embeddings[topic_id]
            )
        else:
            # Hybrid graph (new default)
            graph = HybridDocumentGraph(
                k_neighbors=args.k_neighbors,
                similarity_threshold=args.threshold,
                max_hops=2,
                alpha=args.alpha,
                entity_boost=args.entity_boost
            )
            graph.build_graph(
                docs=topic_docs,
                embedder=None,
                precomputed_embeddings=all_embeddings[topic_id]
            )

        edges = sum(len(v) for v in graph.adjacency_list.values()) // 2
        degrees = [len(v) for v in graph.adjacency_list.values()]
        components = graph.get_connected_components()

        print(f"   Edges: {edges}")
        print(f"   Degree: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.1f}")
        print(f"   Components: {len(components)}", flush=True)

        all_graphs[topic_id] = graph
        total_edges += edges

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("GRAPH BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"   Graphs: {len(all_graphs)}")
    print(f"   Total edges: {total_edges}")
    print(f"   Build time: {elapsed:.1f}s", flush=True)

    # Save graphs
    print(f"\n Saving graphs to {graph_path}...")

    # Serialize graphs
    graph_data = {}
    for topic_id, graph in all_graphs.items():
        # Get topic name for this topic_id
        topic_name = next((t['topic'] for t in docs if t['topic_id'] == topic_id), f'Topic {topic_id}')
        
        if args.semantic_only:
            # Semantic-only format
            graph_data[topic_id] = {
                'topic_name': topic_name,
                'doc_embeddings': graph.doc_embeddings,
                'doc_metadata': graph.doc_metadata,
                'adjacency_list': {k: list(v) for k, v in graph.adjacency_list.items()},
                'similarity_matrix': graph.similarity_matrix,
                'k_neighbors': graph.k_neighbors,
                'similarity_threshold': graph.similarity_threshold,
                'max_hops': graph.max_hops,
                'graph_type': 'semantic'
            }
        else:
            # Hybrid format (includes lexical data + tokenized docs)
            graph_data[topic_id] = {
                'topic_name': topic_name,
                'doc_embeddings': graph.doc_embeddings,
                'doc_metadata': graph.doc_metadata,
                'doc_texts': graph.doc_texts,
                'adjacency_list': {k: list(v) for k, v in graph.adjacency_list.items()},
                'semantic_similarity': graph.semantic_similarity,
                'lexical_similarity': graph.lexical_similarity,
                'hybrid_similarity': graph.hybrid_similarity,
                'k_neighbors': graph.k_neighbors,
                'similarity_threshold': graph.similarity_threshold,
                'max_hops': graph.max_hops,
                'alpha': graph.alpha,
                'entity_boost': graph.entity_boost,
                'graph_type': 'hybrid',
                # Save tokenized docs to avoid re-tokenization at load time
                'tokenized_docs': graph.bm25_model.tokenized_docs if graph.bm25_model else None
            }

    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

    print("   Saved!")

    # Print graph statistics
    print(f"\n{'='*80}")
    print("GRAPH STATISTICS")
    print(f"{'='*80}")

    for topic_id, graph in all_graphs.items():
        topic_name = next(t['topic'] for t in docs if t['topic_id'] == topic_id)
        n_docs = len(graph.doc_metadata)
        n_edges = sum(len(v) for v in graph.adjacency_list.values()) // 2
        degrees = [len(v) for v in graph.adjacency_list.values()]
        components = graph.get_connected_components()

        print(f"\n   Topic {topic_id}: {topic_name[:40]}...")
        print(f"      Docs: {n_docs}, Edges: {n_edges}")
        print(f"      Degree: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.1f}")
        print(f"      Components: {len(components)} (sizes: {sorted([len(c) for c in components], reverse=True)[:5]})")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"\n   Embeddings: {embeddings_path}")
    print(f"   Graphs: {graph_path}")
    print("\n   Next steps:")
    print("   1. Run experiment with graph RAG:")
    print(f"      python scripts/run_experiment.py --use-graph-rag --graph-path {graph_path}")


if __name__ == '__main__':
    main()
