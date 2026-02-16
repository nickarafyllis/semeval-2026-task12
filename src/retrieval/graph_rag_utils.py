"""
Graph RAG Utilities - Centralized functions for loading and using Hybrid Graph RAG

This module provides shared utilities for Hybrid Graph RAG that can be used across:
- run_experiment.py
- optimize_prompts.py

HYBRID RETRIEVAL STRATEGY:
1. Load hybrid graph (semantic + lexical edges)
2. Find entry points using BOTH semantic AND lexical similarity
3. Traverse graph to get ALL connected documents
4. Return all relevant documents (no artificial limit)

Usage:
    from src.retrieval.graph_rag_utils import load_graph_rag_data, retrieve_with_graph_rag

    # Load Graph RAG components
    graph_rag_data = load_graph_rag_data(
        graph_path='data/indices/doc_graph_dev.pkl',
        query_embeddings_path='data/indices/query_embeddings_dev.pkl'
    )

    # Retrieve ALL connected documents for a question
    docs = retrieve_with_graph_rag(question, graph_rag_data)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import deque

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.document_graph_rag import (
    BedrockDocumentEmbedder, 
    DocumentSemanticGraph,
    HybridDocumentGraph,
    BM25EntitySimilarity
)


def load_graph_rag_data(
    graph_path: str = 'data/indices/doc_graph_dev.pkl',
    query_embeddings_path: str = 'data/indices/query_embeddings_dev.pkl',
    n_semantic_entry: int = 3,
    n_lexical_entry: int = 2,
    min_cluster_size: int = 1,
    max_docs: Optional[int] = None,  # None = no limit, retrieve all connected
    embedding_model: str = 'cohere.embed-v4:0',
    embedding_dims: int = 1024,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Load Hybrid Graph RAG components from precomputed files.

    Args:
        graph_path: Path to precomputed document graphs pickle
        query_embeddings_path: Path to precomputed query embeddings pickle
        n_semantic_entry: Number of semantic entry points (default 3)
        n_lexical_entry: Number of lexical entry points (default 2)
        min_cluster_size: Minimum cluster size for filtering (1=no filtering)
        max_docs: Maximum documents to return (None = no limit)
        embedding_model: Bedrock embedding model for fallback
        embedding_dims: Embedding dimensions
        verbose: Print loading progress

    Returns:
        Dictionary with Graph RAG data or None if files not found
    """
    graph_path = Path(graph_path)
    query_emb_path = Path(query_embeddings_path)

    if not graph_path.exists():
        if verbose:
            print(f"   Graph file not found: {graph_path}")
        return None

    if verbose:
        print(f"\n   Loading Hybrid Graph RAG components...")

    # Load document graphs
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)

    # Detect graph type and reconstruct
    graphs = {}
    graph_type = 'unknown'
    
    for topic_id, data in graph_data.items():
        graph_type = data.get('graph_type', 'semantic')
        
        if graph_type == 'hybrid':
            # Reconstruct HybridDocumentGraph
            graph = HybridDocumentGraph(
                k_neighbors=data['k_neighbors'],
                similarity_threshold=data['similarity_threshold'],
                max_hops=data['max_hops'],
                alpha=data.get('alpha', 0.7),
                entity_boost=data.get('entity_boost', 3.0)
            )
            graph.doc_embeddings = data['doc_embeddings']
            graph.doc_metadata = data['doc_metadata']
            graph.doc_texts = data.get('doc_texts', [])
            graph.adjacency_list = {int(k): set(v) for k, v in data['adjacency_list'].items()}
            graph.semantic_similarity = data.get('semantic_similarity')
            graph.lexical_similarity = data.get('lexical_similarity')
            graph.hybrid_similarity = data.get('hybrid_similarity')
            
            # Reconstruct BM25 model from saved tokenized docs (avoids re-tokenization)
            if 'tokenized_docs' in data and data['tokenized_docs'] is not None:
                from rank_bm25 import BM25Plus
                graph.bm25_model = BM25EntitySimilarity(entity_boost=graph.entity_boost)
                graph.bm25_model.tokenized_docs = data['tokenized_docs']
                graph.bm25_model.bm25 = BM25Plus(graph.bm25_model.tokenized_docs)
            elif graph.doc_texts:
                # Fallback: re-tokenize if tokenized_docs not saved (old format)
                graph.bm25_model = BM25EntitySimilarity(entity_boost=graph.entity_boost)
                graph.bm25_model.fit(graph.doc_texts)
        else:
            # Reconstruct DocumentSemanticGraph (legacy)
            graph = DocumentSemanticGraph(
                k_neighbors=data['k_neighbors'],
                similarity_threshold=data['similarity_threshold'],
                max_hops=data['max_hops']
            )
            graph.doc_embeddings = data['doc_embeddings']
            graph.doc_metadata = data['doc_metadata']
            graph.adjacency_list = {int(k): set(v) for k, v in data['adjacency_list'].items()}
            graph.similarity_matrix = data.get('similarity_matrix')
        
        graphs[int(topic_id)] = graph

    if verbose:
        print(f"   Loaded {len(graphs)} topic graphs (type: {graph_type})")

    # Load query embeddings
    query_embeddings = None
    if query_emb_path.exists():
        with open(query_emb_path, 'rb') as f:
            query_embeddings = pickle.load(f)
        if verbose:
            print(f"   Loaded {len(query_embeddings)} precomputed query embeddings")
    else:
        if verbose:
            print(f"   No precomputed query embeddings found, will compute at runtime")

    # Initialize embedder (for fallback when query embedding not precomputed)
    embedder = BedrockDocumentEmbedder(model_id=embedding_model, dimensions=embedding_dims)

    graph_rag_data = {
        'graphs': graphs,
        'graph_type': graph_type,
        'query_embeddings': query_embeddings,
        'embedder': embedder,
        'n_semantic_entry': n_semantic_entry,
        'n_lexical_entry': n_lexical_entry,
        'min_cluster_size': min_cluster_size,
        'max_docs': max_docs
    }

    if verbose:
        max_docs_str = str(max_docs) if max_docs else "unlimited"
        print(f"   Entry points: {n_semantic_entry} semantic + {n_lexical_entry} lexical")
        print(f"   Traversal: UNLIMITED (entire connected component)")
        print(f"   max_docs={max_docs_str}")

    return graph_rag_data


def retrieve_with_graph_rag(
    question: Dict[str, Any],
    graph_rag_data: Dict[str, Any],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using Hybrid Graph RAG.

    RETRIEVAL STRATEGY:
    1. Find entry points via BOTH semantic AND lexical similarity
    2. Traverse hybrid graph to find ALL connected documents (UNLIMITED by default)
    3. Return all relevant documents (no artificial limit)

    Args:
        question: Question dict with keys: id, topic_id, target_event, option_A/B/C/D
        graph_rag_data: Dict from load_graph_rag_data()
        verbose: Whether to print selection progress

    Returns:
        List of document dicts with 'content' and 'query_similarity' keys
    """
    if not graph_rag_data:
        return []

    graphs = graph_rag_data['graphs']
    graph_type = graph_rag_data.get('graph_type', 'semantic')
    query_embeddings = graph_rag_data.get('query_embeddings')
    embedder = graph_rag_data['embedder']
    n_semantic_entry = graph_rag_data.get('n_semantic_entry', 3)
    n_lexical_entry = graph_rag_data.get('n_lexical_entry', 2)
    min_cluster_size = graph_rag_data.get('min_cluster_size', 1)
    max_docs = graph_rag_data.get('max_docs')  # None = no limit
    unlimited_traversal = graph_rag_data.get('unlimited_traversal', True)  # Default: UNLIMITED

    topic_id = question.get('topic_id')
    id = question.get('id')

    if topic_id not in graphs:
        return []

    graph = graphs[topic_id]

    # Build query text
    query_parts = [question.get('target_event', '')]
    for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
        if opt in question:
            query_parts.append(question[opt])
    query_text = ' '.join(query_parts)

    # Get query embedding (precomputed or compute now)
    query_embedding = None
    if query_embeddings and id in query_embeddings:
        query_embedding = query_embeddings[id]
    else:
        # Try to compute embedding, but fallback gracefully if embedder unavailable
        try:
            query_embedding = embedder.encode([query_text], input_type='search_query', verbose=False)[0]
        except Exception as e:
            if verbose:
                print(f"   Warning: Could not compute embedding for question {id}: {e}")
            # Will use lexical-only retrieval below
            pass

    # === STEP 1: Find entry points (hybrid: semantic + lexical) ===
    entry_types = {}
    if graph_type == 'hybrid' and hasattr(graph, 'get_hybrid_entry_points_detailed'):
        # Use hybrid entry points with detailed type tracking
        entry_indices, entry_types = graph.get_hybrid_entry_points_detailed(
            query_embedding=query_embedding,
            query_text=query_text,
            n_semantic=n_semantic_entry,
            n_lexical=n_lexical_entry
        )
    elif graph_type == 'hybrid' and hasattr(graph, 'get_hybrid_entry_points'):
        # Fallback to basic hybrid entry points (no type tracking)
        entry_indices = graph.get_hybrid_entry_points(
            query_embedding=query_embedding,
            query_text=query_text,
            n_semantic=n_semantic_entry,
            n_lexical=n_lexical_entry
        )
    else:
        # Fallback: semantic-only entry points
        similarities = cosine_similarity([query_embedding], graph.doc_embeddings)[0]
        entry_indices = np.argsort(similarities)[-(n_semantic_entry + n_lexical_entry):][::-1].tolist()

    # === STEP 2: Expand via graph traversal (UNLIMITED by default = entire connected component) ===
    expanded_indices = graph.traverse(list(entry_indices), unlimited=unlimited_traversal)

    # === STEP 3: Filter by cluster size if needed ===
    if min_cluster_size > 1:
        subgraph_adj = {
            i: graph.adjacency_list[i] & expanded_indices
            for i in expanded_indices
        }
        filtered_indices = set()
        visited = set()

        for start in expanded_indices:
            if start in visited:
                continue
            cluster = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node not in visited and node in expanded_indices:
                    visited.add(node)
                    cluster.add(node)
                    queue.extend(subgraph_adj.get(node, set()) - visited)

            if len(cluster) >= min_cluster_size:
                filtered_indices.update(cluster)

        expanded_indices = filtered_indices if filtered_indices else expanded_indices

    # === STEP 4: Build retrieved docs with metadata ===
    # Print selection progress
    if verbose:
        total_docs_in_topic = len(graph.doc_metadata)
        selected_docs = len(expanded_indices)
        print(f"   {selected_docs}/{total_docs_in_topic} documents of topic {topic_id} selected")
    
    # Compute similarities for ranking
    all_similarities = cosine_similarity([query_embedding], graph.doc_embeddings)[0]
    
    retrieved_docs = []
    for idx in expanded_indices:
        doc = graph.doc_metadata[idx].copy()
        doc['query_similarity'] = float(all_similarities[idx])
        doc['is_entry_point'] = idx in entry_indices
        doc['entry_type'] = entry_types.get(idx, 'traversal')  # lexical/semantic/hybrid/traversal
        doc['from_graph'] = True
        doc['_doc_idx'] = idx  # Store original index for canonical ordering
        retrieved_docs.append(doc)

    # === CANONICAL ORDERING FOR CACHE EFFICIENCY ===
    # Sort by document index (stable) to ensure consistent ordering across questions.
    # This maximizes cache prefix matches when questions retrieve overlapping documents.
    # Documents appear in the same order regardless of query-specific similarity scores.
    retrieved_docs.sort(key=lambda x: x['_doc_idx'])

    # Apply max_docs limit only if specified
    if max_docs is not None:
        retrieved_docs = retrieved_docs[:max_docs]

    return retrieved_docs


def format_graph_rag_context(
    retrieved_docs: List[Dict[str, Any]],
    max_docs: Optional[int] = None,
    max_doc_length: int = None
) -> str:
    """
    Format retrieved documents into a context string for prompts.

    Args:
        retrieved_docs: List of document dicts from retrieve_with_graph_rag()
        max_docs: Maximum documents to include (None = include all)
        max_doc_length: DEPRECATED - no longer truncates documents (None = no limit)

    Returns:
        Formatted context string with XML tags
    """
    if not retrieved_docs:
        return ""

    docs_to_format = retrieved_docs if max_docs is None else retrieved_docs[:max_docs]

    context_parts = []
    for i, doc in enumerate(docs_to_format, 1):
        title = doc.get('title', f'Document {i}')
        content = doc.get('content', doc.get('text', ''))

        # No truncation - use full document content
        # We have 50K token output limit and exponential retry scaling, so we can handle full docs

        context_parts.append(f"<document_{i}>\nTitle: {title}\n\n{content}\n</document_{i}>")

    return "<context_documents>\n" + "\n\n".join(context_parts) + "\n</context_documents>"


def get_graph_rag_stats(graph_rag_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about loaded Graph RAG data.

    Args:
        graph_rag_data: Dict from load_graph_rag_data()

    Returns:
        Dictionary with statistics
    """
    if not graph_rag_data:
        return {'loaded': False}

    graphs = graph_rag_data.get('graphs', {})
    query_embeddings = graph_rag_data.get('query_embeddings', {})

    total_docs = sum(len(g.doc_metadata) for g in graphs.values())
    total_edges = sum(sum(len(adj) for adj in g.adjacency_list.values()) // 2 for g in graphs.values())

    return {
        'loaded': True,
        'graph_type': graph_rag_data.get('graph_type', 'unknown'),
        'n_topics': len(graphs),
        'n_documents': total_docs,
        'n_edges': total_edges,
        'n_query_embeddings': len(query_embeddings) if query_embeddings else 0,
        'n_semantic_entry': graph_rag_data.get('n_semantic_entry', 3),
        'n_lexical_entry': graph_rag_data.get('n_lexical_entry', 2),
        'n_hops': graph_rag_data.get('n_hops', 3),
        'max_docs': graph_rag_data.get('max_docs')
    }


def build_topic_wide_context(
    questions: List[Dict],
    graph_rag_data: Dict[str, Any],
    verbose: bool = True
) -> Dict[int, List[Dict]]:
    """
    Build topic-wide contexts by aggregating GraphRAG retrievals across all questions in each topic.

    This creates a shared, comprehensive context for each topic that can be cached once
    and reused across all questions, dramatically improving cache efficiency.

    Strategy:
    1. Group questions by topic_id
    2. For each topic, retrieve docs for ALL questions
    3. Merge into unique set (union) using document indices
    4. Sort by document index for canonical ordering (cache efficiency)
    5. Return unlimited documents (no filtering)

    Args:
        questions: List of question dicts with topic_id
        graph_rag_data: Dict from load_graph_rag_data()
        verbose: Print progress

    Returns:
        Dict mapping topic_id (int) -> list of unique retrieved docs (sorted by doc index)

    Example:
        >>> questions = load_dev_data()[0]
        >>> graph_rag_data = load_graph_rag_data(...)
        >>> topic_contexts = build_topic_wide_context(questions, graph_rag_data)
        >>> context_for_topic_1 = format_graph_rag_context(topic_contexts[1])
    """
    from collections import defaultdict

    if not graph_rag_data:
        return {}

    topic_doc_indices = defaultdict(set)  # topic_id -> set of doc indices
    topic_doc_objects = defaultdict(dict)  # topic_id -> {doc_idx: doc_dict}
    topic_doc_similarities = defaultdict(lambda: defaultdict(list))  # topic_id -> {doc_idx: [sim1, sim2, ...]}

    if verbose:
        print("\n📦 Building topic-wide contexts...")

    # Group questions by topic
    topic_questions = defaultdict(list)
    for q in questions:
        topic_id = q.get('topic_id')
        if topic_id:
            topic_questions[topic_id].append(q)

    if verbose:
        print(f"   Found {len(topic_questions)} topics")

    # For each topic, aggregate all retrievals
    for topic_id, topic_q_list in sorted(topic_questions.items()):
        if verbose:
            print(f"   Topic {topic_id}: {len(topic_q_list)} questions")

        # Retrieve docs for each question in this topic
        for q in topic_q_list:
            retrieved = retrieve_with_graph_rag(q, graph_rag_data, verbose=False)

            for doc in retrieved:
                doc_idx = doc.get('_doc_idx')
                if doc_idx is not None:
                    topic_doc_indices[topic_id].add(doc_idx)
                    # Store doc object (overwrites are fine, same doc)
                    topic_doc_objects[topic_id][doc_idx] = doc
                    # Track similarity for this question
                    similarity = doc.get('query_similarity', 0.0)
                    topic_doc_similarities[topic_id][doc_idx].append(similarity)

        n_unique = len(topic_doc_indices[topic_id])
        if verbose:
            print(f"      → Aggregated {n_unique} unique documents")

    # Convert to sorted lists (canonical ordering for cache efficiency)
    topic_contexts = {}
    for topic_id in topic_doc_indices:
        # Get all document indices for this topic (unlimited)
        doc_indices = list(topic_doc_indices[topic_id])

        # Sort by doc index for canonical ordering (cache efficiency)
        sorted_indices = sorted(doc_indices)

        # Build final doc list
        topic_contexts[topic_id] = [
            topic_doc_objects[topic_id][idx] for idx in sorted_indices
        ]

    if verbose:
        total_docs = sum(len(docs) for docs in topic_contexts.values())
        avg_docs = total_docs / len(topic_contexts) if topic_contexts else 0
        print(f"\n   ✅ Built {len(topic_contexts)} topic-wide contexts")
        print(f"   📊 Total documents: {total_docs}")
        print(f"   📊 Avg docs per topic: {avg_docs:.1f}")

    return topic_contexts


def load_preprocessed_contexts(split: str = "dev", data_dir: str = "data/indices") -> Optional[Dict]:
    """
    Load preprocessed topic-wide contexts from disk.

    Args:
        split: Data split name (dev, train, test, sample)
        data_dir: Directory containing preprocessed files

    Returns:
        Dict with 'context_strings' (topic_id -> context string) and 'statistics',
        or None if file doesn't exist
    """
    import pickle
    from pathlib import Path

    preprocessed_path = Path(data_dir) / f"topic_wide_contexts_{split}.pkl"

    if not preprocessed_path.exists():
        return None

    try:
        with open(preprocessed_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n📦 Loaded preprocessed topic-wide contexts from {preprocessed_path}")
        stats = data.get('statistics', {})
        print(f"   Topics: {stats.get('total_topics', 'N/A')}")
        print(f"   Deduplication: {stats.get('dedup_savings_pct', 0):.1f}%")
        print(f"   Cache hit rate: {stats.get('cache_hit_rate_pct', 0):.1f}%")

        return data
    except Exception as e:
        print(f"   ⚠️  Failed to load preprocessed contexts: {e}")
        return None


class PreprocessedContextCache:
    """
    Context cache using preprocessed topic-wide contexts.

    Loads pre-computed topic-wide contexts from disk for maximum efficiency.
    Falls back to static topic context if preprocessed file is not available.

    Usage:
        cache = PreprocessedContextCache(split="dev", docs=docs)
        context = cache.get_context(topic_id)
    """

    def __init__(self, split: str = "dev", docs: List[Dict] = None, data_dir: str = "data/indices"):
        """
        Initialize with preprocessed contexts or fall back to static cache.

        Args:
            split: Data split (dev, train, test, sample)
            docs: Fallback docs data for static cache
            data_dir: Directory containing preprocessed files
        """
        self.split = split
        self.context_cache = {}

        # Try to load preprocessed contexts
        preprocessed = load_preprocessed_contexts(split, data_dir)

        if preprocessed and 'context_strings' in preprocessed:
            self.context_cache = preprocessed['context_strings']
            self.statistics = preprocessed.get('statistics', {})
            self.using_preprocessed = True
            print(f"   ✅ Using preprocessed topic-wide contexts ({len(self.context_cache)} topics)")
        else:
            # Fallback to static topic context
            print(f"   ⚠️  Preprocessed contexts not found, using static topic context")
            self.using_preprocessed = False
            self.statistics = {}

            if docs:
                for entry in docs:
                    tid = entry.get("topic_id")
                    doc_list = entry.get("docs", [])
                    if tid and doc_list:
                        self.context_cache[tid] = "\n\n---\n\n".join([
                            d.get("content") or d.get("snippet") or "" for d in doc_list
                        ])

    def get_context(self, topic_id, question=None) -> str:
        """
        Get context for a topic.

        Args:
            topic_id: Topic ID
            question: Question dict (ignored, context is topic-wide)

        Returns:
            Context string
        """
        return self.context_cache.get(topic_id, "")
