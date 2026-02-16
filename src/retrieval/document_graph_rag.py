"""
Document-Level Hybrid Graph RAG

This module implements a document-to-document HYBRID graph for high-recall retrieval.
The graph uses both semantic (embedding) and lexical (BM25+ with entity boosting)
similarity to create edges, ensuring both conceptual and keyword-based connections.

Key Components:
1. BedrockDocumentEmbedder - Embeds full documents using AWS Bedrock
2. BM25EntitySimilarity - Computes lexical similarity with entity boosting
3. HybridDocumentGraph - Builds hybrid (semantic + lexical) document graph
4. GraphRAGRetriever - Retrieves documents using n-hop graph traversal

Hybrid Formula:
    edge_weight = alpha × semantic_similarity + (1 - alpha) × lexical_similarity
    Default alpha = 0.7 (semantic-dominant)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
import sys
import re

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'configs'))


# Global shared spaCy model (singleton pattern for memory efficiency)
_SHARED_SPACY_MODEL = None

def get_shared_spacy_model():
    """Get or load the shared spaCy model (singleton pattern)."""
    global _SHARED_SPACY_MODEL
    
    if _SHARED_SPACY_MODEL is None:
        try:
            import spacy
            try:
                _SHARED_SPACY_MODEL = spacy.load("en_core_web_sm")
            except OSError:
                print("   Downloading spaCy model en_core_web_sm...")
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                             capture_output=True)
                _SHARED_SPACY_MODEL = spacy.load("en_core_web_sm")
            print(f"   Loaded spaCy for entity extraction (shared instance)")
        except ImportError:
            print("   Warning: spaCy not available, using basic tokenization")
            _SHARED_SPACY_MODEL = None
    
    return _SHARED_SPACY_MODEL


class BM25EntitySimilarity:
    """
    BM25+ with Entity Boosting for lexical similarity computation.
    
    This provides true keyword/lexical matching that complements semantic embeddings.
    Entities (people, places, organizations, events) are boosted to improve
    matching for event reasoning tasks.
    """
    
    def __init__(self, entity_boost: float = 3.0, use_lemmatization: bool = True):
        """
        Initialize BM25+ with entity boosting.
        
        Args:
            entity_boost: Weight multiplier for entities (default 3.0)
            use_lemmatization: Whether to lemmatize tokens (default True)
        """
        self.entity_boost = entity_boost
        self.use_lemmatization = use_lemmatization
        self.bm25 = None
        self.tokenized_docs = None
        # Use shared spaCy model (singleton pattern for memory efficiency)
        self.nlp = get_shared_spacy_model()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with entity boosting.
        
        Args:
            text: Document text
            
        Returns:
            List of tokens with entities repeated for boosting
        """
        if self.nlp is not None:
            doc = self.nlp(text[:100000])  # Limit for spaCy
            
            # Get regular tokens (lemmatized, no stopwords)
            if self.use_lemmatization:
                tokens = [t.lemma_.lower() for t in doc 
                         if not t.is_stop and t.is_alpha and len(t.text) > 1]
            else:
                tokens = [t.text.lower() for t in doc 
                         if not t.is_stop and t.is_alpha and len(t.text) > 1]
            
            # Extract and boost entities
            entities = []
            for ent in doc.ents:
                # Clean entity text
                ent_text = ent.text.lower().strip()
                if len(ent_text) > 1:
                    # Add entity multiple times for boosting
                    entities.extend([ent_text] * int(self.entity_boost))
            
            return tokens + entities
        else:
            # Fallback: basic tokenization
            tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
            # Simple stopword removal
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                        'through', 'during', 'before', 'after', 'above', 'below',
                        'between', 'under', 'again', 'further', 'then', 'once',
                        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                        'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                        'very', 'just', 'also', 'now', 'here', 'there', 'when',
                        'where', 'why', 'how', 'all', 'each', 'every', 'both',
                        'few', 'more', 'most', 'other', 'some', 'such', 'no',
                        'any', 'this', 'that', 'these', 'those', 'it', 'its'}
            return [t for t in tokens if t not in stopwords]
    
    def fit(self, docs: List[str]) -> None:
        """
        Fit BM25+ model on documents.
        
        Args:
            docs: List of document texts
        """
        from rank_bm25 import BM25Plus
        
        print(f"   Tokenizing {len(docs)} documents for BM25+...")
        self.tokenized_docs = [self.tokenize(doc) for doc in docs]
        
        print(f"   Fitting BM25+ model...")
        self.bm25 = BM25Plus(self.tokenized_docs)
    
    def compute_similarity_matrix(self, docs: List[str]) -> np.ndarray:
        """
        Compute pairwise BM25+ similarity matrix.
        
        Args:
            docs: List of document texts
            
        Returns:
            Normalized similarity matrix (n_docs x n_docs)
        """
        if self.bm25 is None:
            self.fit(docs)
        
        n = len(docs)
        sim_matrix = np.zeros((n, n))
        
        print(f"   Computing BM25+ similarity matrix ({n}x{n})...")
        for i, doc_tokens in enumerate(self.tokenized_docs):
            if doc_tokens:  # Skip empty docs
                scores = self.bm25.get_scores(doc_tokens)
                # Normalize to [0, 1]
                max_score = scores.max()
                if max_score > 0:
                    scores = scores / max_score
                sim_matrix[i] = scores
        
        # Make symmetric (average of both directions)
        sim_matrix = (sim_matrix + sim_matrix.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(sim_matrix, 1.0)
        
        return sim_matrix
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25+ scores for a query against all documents.
        
        Args:
            query: Query text
            
        Returns:
            Array of scores for each document
        """
        if self.bm25 is None:
            raise ValueError("BM25+ model not fitted. Call fit() first.")
        
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Normalize
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        
        return scores


class HybridDocumentGraph:
    """
    Hybrid document graph using both semantic and lexical similarity.
    
    Edge weights are computed as:
        hybrid_sim = alpha * semantic_sim + (1 - alpha) * lexical_sim
    
    Default alpha = 0.7 (semantic-dominant, as requested)
    """
    
    def __init__(self,
                 k_neighbors: int = 5,
                 similarity_threshold: float = 0.4,
                 max_hops: int = 2,
                 alpha: float = 0.7,
                 entity_boost: float = 3.0):
        """
        Initialize hybrid document graph.
        
        Args:
            k_neighbors: Number of nearest neighbors per document
            similarity_threshold: Minimum hybrid similarity for edges
            max_hops: Maximum traversal depth
            alpha: Weight for semantic similarity (default 0.7)
            entity_boost: Entity weight multiplier for BM25+ (default 3.0)
        """
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops
        self.alpha = alpha
        self.entity_boost = entity_boost
        
        # Graph data
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_metadata: List[Dict] = []
        self.doc_texts: List[str] = []
        self.adjacency_list: Dict[int, Set[int]] = {}
        
        # Similarity matrices
        self.semantic_similarity: Optional[np.ndarray] = None
        self.lexical_similarity: Optional[np.ndarray] = None
        self.hybrid_similarity: Optional[np.ndarray] = None
        
        # BM25+ model
        self.bm25_model: Optional[BM25EntitySimilarity] = None
    
    def build_graph(self,
                    docs: List[Dict],
                    embedder,
                    precomputed_embeddings: Optional[np.ndarray] = None) -> None:
        """
        Build hybrid document graph.
        
        Args:
            docs: List of document dictionaries
            embedder: Embedder for semantic similarity (if needed)
            precomputed_embeddings: Optional precomputed embeddings
        """
        print(f"\n   Building HYBRID document graph...")
        print(f"   alpha={self.alpha} (semantic), 1-alpha={1-self.alpha:.1f} (lexical)")
        print(f"   k_neighbors={self.k_neighbors}, threshold={self.similarity_threshold}")
        
        # Store metadata and extract texts
        self.doc_metadata = docs
        self.doc_texts = [self._get_doc_text(doc) for doc in docs]
        n_docs = len(docs)
        
        # Step 1: Get semantic embeddings
        if precomputed_embeddings is not None:
            self.doc_embeddings = precomputed_embeddings
            print(f"   Using precomputed embeddings: {self.doc_embeddings.shape}")
        else:
            print(f"   Embedding {n_docs} documents...")
            self.doc_embeddings = embedder.encode(self.doc_texts, input_type='search_document')
        
        # Step 2: Compute semantic similarity
        print(f"   Computing semantic similarity matrix...")
        self.semantic_similarity = cosine_similarity(self.doc_embeddings)
        
        # Step 3: Compute lexical similarity with BM25+
        print(f"   Computing lexical similarity (BM25+ with entity boost={self.entity_boost})...")
        self.bm25_model = BM25EntitySimilarity(entity_boost=self.entity_boost)
        self.lexical_similarity = self.bm25_model.compute_similarity_matrix(self.doc_texts)
        
        # Step 4: Compute hybrid similarity
        print(f"   Computing hybrid similarity (α={self.alpha})...")
        self.hybrid_similarity = (
            self.alpha * self.semantic_similarity + 
            (1 - self.alpha) * self.lexical_similarity
        )
        
        # Step 5: Build adjacency list
        print(f"   Building adjacency list...")
        self.adjacency_list = {i: set() for i in range(n_docs)}
        
        edge_count = 0
        for i in range(n_docs):
            similarities = self.hybrid_similarity[i].copy()
            similarities[i] = -1  # Exclude self
            
            # Get top-k neighbors
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:]
            
            # Add edges above threshold
            for j in top_k_indices:
                if similarities[j] >= self.similarity_threshold:
                    self.adjacency_list[i].add(j)
                    self.adjacency_list[j].add(i)
                    edge_count += 1
        
        # Statistics
        degrees = [len(neighbors) for neighbors in self.adjacency_list.values()]
        print(f"   Graph built: {n_docs} nodes, {edge_count} edges")
        print(f"   Degree stats: min={min(degrees)}, max={max(degrees)}, avg={np.mean(degrees):.1f}")
        
        # Compare semantic vs lexical contribution
        sem_mean = np.mean(self.semantic_similarity[np.triu_indices(n_docs, k=1)])
        lex_mean = np.mean(self.lexical_similarity[np.triu_indices(n_docs, k=1)])
        hyb_mean = np.mean(self.hybrid_similarity[np.triu_indices(n_docs, k=1)])
        print(f"   Avg similarity - Semantic: {sem_mean:.3f}, Lexical: {lex_mean:.3f}, Hybrid: {hyb_mean:.3f}")
    
    def _get_doc_text(self, doc: Dict) -> str:
        """Extract text from document."""
        if 'content' in doc:
            return doc['content']
        elif 'text' in doc:
            return doc['text']
        elif 'docs' in doc:
            return ' '.join([d.get('content', d.get('text', '')) for d in doc['docs']])
        return str(doc)
    
    def traverse(self, entry_indices: List[int], n_hops: Optional[int] = None, 
                 unlimited: bool = True) -> Set[int]:
        """
        Perform BFS traversal from entry points.
        
        Args:
            entry_indices: Starting document indices
            n_hops: Maximum hops (ignored if unlimited=True, default: self.max_hops)
            unlimited: If True (default), traverse entire connected component.
                      If False, limit to n_hops.
        
        Returns:
            Set of all reachable document indices
        """
        if n_hops is None:
            n_hops = self.max_hops
        
        visited = set(entry_indices)
        frontier = set(entry_indices)
        hop_count = 0
        
        while frontier:
            # Stop if we've reached hop limit (only when not unlimited)
            if not unlimited and hop_count >= n_hops:
                break
            
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency_list.get(node, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            
            if not next_frontier:
                break  # No more nodes to explore
            
            frontier = next_frontier
            hop_count += 1
        
        return visited
    
    def get_connected_components(self) -> List[Set[int]]:
        """Find all connected components in the graph."""
        visited = set()
        components = []
        
        for node in range(len(self.doc_metadata)):
            if node not in visited:
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        queue.extend(self.adjacency_list.get(current, set()) - visited)
                
                components.append(component)
        
        return components
    
    def get_hybrid_entry_points(self,
                                 query_embedding: np.ndarray,
                                 query_text: str,
                                 n_semantic: int = 3,
                                 n_lexical: int = 2) -> List[int]:
        """
        Get entry points using both semantic and lexical similarity.

        Ensures exactly n_semantic + n_lexical unique entry points by:
        1. Taking top-n from lexical (BM25)
        2. Taking top-n from semantic (embeddings)
        3. If overlap exists, add next-best from combined ranking to reach target

        Args:
            query_embedding: Query embedding vector
            query_text: Query text for BM25+
            n_semantic: Number of semantic entry points
            n_lexical: Number of lexical entry points

        Returns:
            List of unique entry point indices (exactly n_semantic + n_lexical)
        """
        entry_points, _ = self.get_hybrid_entry_points_detailed(
            query_embedding, query_text, n_semantic, n_lexical
        )
        return entry_points

    def get_hybrid_entry_points_detailed(self,
                                          query_embedding: np.ndarray,
                                          query_text: str,
                                          n_semantic: int = 3,
                                          n_lexical: int = 2):
        """
        Get entry points with detailed type information.

        Returns:
            Tuple of (entry_points, entry_types) where entry_types is a dict:
            {idx: 'lexical' | 'semantic'}

            Note: Gap-fill documents are classified as 'lexical' or 'semantic'
            based on which score is higher for that document.
        """
        target_count = n_semantic + n_lexical

        # Compute all scores
        semantic_scores = cosine_similarity([query_embedding], self.doc_embeddings)[0]

        if self.bm25_model is not None:
            lexical_scores = self.bm25_model.get_scores(query_text)
        else:
            lexical_scores = np.zeros(len(self.doc_embeddings))

        # Rank all documents by each method
        semantic_ranked = np.argsort(semantic_scores)[::-1].tolist()
        lexical_ranked = np.argsort(lexical_scores)[::-1].tolist()

        # Start with top candidates (lexical first, then semantic)
        entry_points = []
        entry_types = {}
        used = set()

        # Add top lexical
        n_lexical_added = 0
        for idx in lexical_ranked:
            if n_lexical_added >= n_lexical:
                break
            if idx not in used:
                entry_points.append(idx)
                entry_types[idx] = 'lexical'
                used.add(idx)
                n_lexical_added += 1

        # Add top semantic (independent count)
        n_semantic_added = 0
        for idx in semantic_ranked:
            if n_semantic_added >= n_semantic:
                break
            if idx not in used:
                entry_points.append(idx)
                entry_types[idx] = 'semantic'
                used.add(idx)
                n_semantic_added += 1

        return entry_points[:target_count], entry_types


class BedrockDocumentEmbedder:
    """
    AWS Bedrock embedder for document-level embeddings.
    Uses Cohere Embed v4 or Amazon Titan for generating document embeddings.
    """

    def __init__(self, model_id: str = 'cohere.embed-v4:0', dimensions: int = 1024):
        """
        Initialize the Bedrock document embedder.

        Args:
            model_id: Bedrock embedding model ID
            dimensions: Output embedding dimensions (default 1024 for Cohere v4)
        """
        from aws_config import get_bedrock_client

        self.bedrock = get_bedrock_client()
        self.model_id = model_id
        self.dimensions = dimensions

        # Determine model type for proper API calls
        self.is_cohere = 'cohere' in model_id.lower()
        self.is_titan = 'titan' in model_id.lower()

        print(f"   Initialized BedrockDocumentEmbedder: {model_id}")

    def encode(self, texts: List[str], input_type: str = 'search_document',
               batch_size: int = 10, verbose: bool = True) -> np.ndarray:
        """
        Encode documents into embeddings.

        Args:
            texts: List of document texts to embed
            input_type: 'search_document' for indexing, 'search_query' for queries
            batch_size: Number of texts to process per API call
            verbose: Print progress

        Returns:
            numpy array of shape (n_texts, dimensions)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        total = len(texts)

        if verbose and total > 1:
            print(f"      Embedding {total} documents in batches of {batch_size}...")

        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch, input_type)
            all_embeddings.append(batch_embeddings)

            if verbose:
                done = min(i + batch_size, total)
                pct = done / total * 100
                print(f"      [{done}/{total}] {pct:.0f}% complete", flush=True)

        return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

    def _encode_batch(self, texts: List[str], input_type: str,
                       max_retries: int = 5, base_delay: float = 2.0) -> np.ndarray:
        """Encode a single batch of texts with retry logic and adaptive batch sizing."""
        import time
        import random

        current_texts = texts
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.is_cohere:
                    body = {
                        "input_type": input_type,
                        "texts": current_texts
                    }
                    if self.dimensions != 1536:
                        body["output_dimension"] = self.dimensions

                elif self.is_titan:
                    body = {
                        "inputText": current_texts[0] if len(current_texts) == 1 else current_texts
                    }
                else:
                    raise ValueError(f"Unsupported embedding model: {self.model_id}")

                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )

                result = json.loads(response['body'].read())

                # Handle different response formats
                if self.is_cohere:
                    if 'embeddings' in result and isinstance(result['embeddings'], dict):
                        embeddings = result['embeddings'].get('float', [])
                    else:
                        embeddings = result.get('embeddings', [])
                elif self.is_titan:
                    embeddings = [result['embedding']]
                else:
                    embeddings = result.get('embeddings', [])

                # If we had to reduce batch size, recursively process remaining texts
                if len(current_texts) < len(texts):
                    remaining = texts[len(current_texts):]
                    remaining_embeddings = self._encode_batch(remaining, input_type, max_retries, base_delay)
                    return np.vstack([np.array(embeddings, dtype=np.float32), remaining_embeddings])

                return np.array(embeddings, dtype=np.float32)

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Reduce batch size on retry: 25 -> 15 -> 10 -> 5 -> 3
                    batch_sizes = [25, 15, 10, 5, 3]
                    new_batch_size = batch_sizes[min(attempt + 1, len(batch_sizes) - 1)]

                    if len(current_texts) > new_batch_size:
                        current_texts = texts[:new_batch_size]
                        print(f"      Retry {attempt + 1}/{max_retries}: reducing batch to {new_batch_size} docs", flush=True)

                    delay = base_delay * (1.5 ** attempt) + random.uniform(0, 1)
                    print(f"      Waiting {delay:.1f}s: {str(e)[:150]}...", flush=True)
                    time.sleep(delay)

        raise last_error


class DocumentSemanticGraph:
    """
    Builds and queries a semantic graph where nodes are documents
    and edges connect semantically similar documents.

    The graph enables n-hop traversal to find all related documents,
    even if they're not directly similar to the query.
    """

    def __init__(self,
                 k_neighbors: int = 5,
                 similarity_threshold: float = 0.4,
                 max_hops: int = 2):
        """
        Initialize the document semantic graph.

        Args:
            k_neighbors: Number of nearest neighbors to connect per document
            similarity_threshold: Minimum cosine similarity for an edge
            max_hops: Maximum graph traversal depth
        """
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops

        # Graph data structures
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_metadata: List[Dict] = []
        self.adjacency_list: Dict[int, Set[int]] = {}
        self.similarity_matrix: Optional[np.ndarray] = None

    def build_graph(self,
                    docs: List[Dict],
                    embedder: BedrockDocumentEmbedder,
                    precomputed_embeddings: Optional[np.ndarray] = None) -> None:
        """
        Build the document semantic graph.

        Args:
            docs: List of document dictionaries with 'content' field
            embedder: Embedder to use if embeddings not precomputed
            precomputed_embeddings: Optional precomputed document embeddings
        """
        print(f"   Building document semantic graph...")
        print(f"   k_neighbors={self.k_neighbors}, threshold={self.similarity_threshold}")

        # Step 1: Get document embeddings
        if precomputed_embeddings is not None:
            self.doc_embeddings = precomputed_embeddings
            print(f"   Using precomputed embeddings: {self.doc_embeddings.shape}")
        else:
            print(f"   Embedding {len(docs)} documents...")
            texts = [self._get_doc_text(doc) for doc in docs]
            self.doc_embeddings = embedder.encode(texts, input_type='search_document')
            print(f"   Generated embeddings: {self.doc_embeddings.shape}")

        # Store document metadata
        self.doc_metadata = docs

        # Step 2: Compute pairwise similarity matrix
        print(f"   Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.doc_embeddings)

        # Step 3: Build adjacency list using k-NN with threshold
        print(f"   Building adjacency list...")
        n_docs = len(docs)
        self.adjacency_list = {i: set() for i in range(n_docs)}

        edge_count = 0
        for i in range(n_docs):
            # Get top-k similar documents (excluding self)
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self

            # Get indices of top-k neighbors
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:]

            # Add edges only if similarity exceeds threshold
            for j in top_k_indices:
                if similarities[j] >= self.similarity_threshold:
                    self.adjacency_list[i].add(j)
                    self.adjacency_list[j].add(i)  # Undirected graph
                    edge_count += 1

        print(f"   Graph built: {n_docs} nodes, {edge_count} edges")

        # Compute graph statistics
        degrees = [len(neighbors) for neighbors in self.adjacency_list.values()]
        print(f"   Degree stats: min={min(degrees)}, max={max(degrees)}, avg={np.mean(degrees):.1f}")

    def _get_doc_text(self, doc: Dict) -> str:
        """Extract text from document for embedding."""
        # Handle different document formats
        if 'content' in doc:
            return doc['content']
        elif 'text' in doc:
            return doc['text']
        elif 'docs' in doc:
            # This is a topic with multiple documents
            return ' '.join([d.get('content', d.get('text', '')) for d in doc['docs']])
        else:
            return str(doc)

    def traverse(self,
                 entry_indices: List[int],
                 n_hops: Optional[int] = None,
                 unlimited: bool = True) -> Set[int]:
        """
        Perform BFS traversal from entry points.

        Args:
            entry_indices: Starting document indices
            n_hops: Maximum hops (ignored if unlimited=True, default: self.max_hops)
            unlimited: If True (default), traverse entire connected component.
                      If False, limit to n_hops.

        Returns:
            Set of all reachable document indices
        """
        if n_hops is None:
            n_hops = self.max_hops

        visited = set(entry_indices)
        frontier = set(entry_indices)
        hop_count = 0

        while frontier:
            # Stop if we've reached hop limit (only when not unlimited)
            if not unlimited and hop_count >= n_hops:
                break

            next_frontier = set()
            for node in frontier:
                neighbors = self.adjacency_list.get(node, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)

            if not next_frontier:
                break  # No more nodes to explore

            frontier = next_frontier
            hop_count += 1

        return visited

    def get_connected_components(self) -> List[Set[int]]:
        """
        Find all connected components in the graph.
        Useful for identifying semantic clusters.

        Returns:
            List of sets, each containing indices of documents in a component
        """
        visited = set()
        components = []

        for node in range(len(self.doc_metadata)):
            if node not in visited:
                # BFS to find all nodes in this component
                component = set()
                queue = deque([node])

                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        queue.extend(self.adjacency_list.get(current, set()) - visited)

                components.append(component)

        return components

    def save(self, path: str) -> None:
        """Save the graph to disk."""
        data = {
            'doc_embeddings': self.doc_embeddings,
            'doc_metadata': self.doc_metadata,
            'adjacency_list': {k: list(v) for k, v in self.adjacency_list.items()},
            'similarity_matrix': self.similarity_matrix,
            'k_neighbors': self.k_neighbors,
            'similarity_threshold': self.similarity_threshold,
            'max_hops': self.max_hops
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"   Graph saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'DocumentSemanticGraph':
        """Load a graph from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        graph = cls(
            k_neighbors=data['k_neighbors'],
            similarity_threshold=data['similarity_threshold'],
            max_hops=data['max_hops']
        )
        graph.doc_embeddings = data['doc_embeddings']
        graph.doc_metadata = data['doc_metadata']
        graph.adjacency_list = {int(k): set(v) for k, v in data['adjacency_list'].items()}
        graph.similarity_matrix = data['similarity_matrix']

        print(f"   Graph loaded from: {path}")
        print(f"   {len(graph.doc_metadata)} documents, {sum(len(v) for v in graph.adjacency_list.values())//2} edges")

        return graph


class GraphRAGRetriever:
    """
    High-recall retriever using document semantic graph.

    Strategy:
    1. Find top-N entry point documents via embedding similarity
    2. Expand to all documents within n-hops of entry points
    3. Optionally filter by cluster size to remove isolated distractors
    """

    def __init__(self,
                 graph: DocumentSemanticGraph,
                 embedder: BedrockDocumentEmbedder,
                 n_entry_points: int = 3,
                 n_hops: int = 2,
                 min_cluster_size: int = 2):
        """
        Initialize the Graph RAG retriever.

        Args:
            graph: Pre-built document semantic graph
            embedder: Embedder for query encoding
            n_entry_points: Number of initial documents to retrieve
            n_hops: Graph traversal depth
            min_cluster_size: Minimum cluster size to keep documents
        """
        self.graph = graph
        self.embedder = embedder
        self.n_entry_points = n_entry_points
        self.n_hops = n_hops
        self.min_cluster_size = min_cluster_size

    def retrieve(self,
                 query: str,
                 topic_id: Optional[int] = None) -> List[Dict]:
        """
        Retrieve documents using graph traversal.

        Args:
            query: The query text (e.g., target event + options)
            topic_id: Optional topic ID to filter by

        Returns:
            List of retrieved document dictionaries
        """
        # Step 1: Encode query
        query_embedding = self.embedder.encode([query], input_type='search_query')[0]

        # Step 2: Find entry points (most similar documents to query)
        if topic_id is not None:
            # Filter to documents from the same topic
            topic_indices = [
                i for i, doc in enumerate(self.graph.doc_metadata)
                if doc.get('topic_id') == topic_id
            ]
            if not topic_indices:
                # Fallback to all documents
                topic_indices = list(range(len(self.graph.doc_metadata)))
        else:
            topic_indices = list(range(len(self.graph.doc_metadata)))

        # Compute similarity to query
        topic_embeddings = self.graph.doc_embeddings[topic_indices]
        similarities = cosine_similarity([query_embedding], topic_embeddings)[0]

        # Get top-N entry points
        top_local_indices = np.argsort(similarities)[-self.n_entry_points:][::-1]
        entry_indices = [topic_indices[i] for i in top_local_indices]

        # Step 3: Expand via graph traversal
        expanded_indices = self.graph.traverse(entry_indices, n_hops=self.n_hops)

        # Step 4: Filter by cluster size (optional distractor removal)
        if self.min_cluster_size > 1:
            # Find connected components among retrieved docs
            subgraph_adj = {
                i: self.graph.adjacency_list[i] & expanded_indices
                for i in expanded_indices
            }

            # Keep only documents in clusters >= min_cluster_size
            filtered_indices = set()
            visited = set()

            for start in expanded_indices:
                if start in visited:
                    continue

                # BFS to find cluster
                cluster = set()
                queue = deque([start])
                while queue:
                    node = queue.popleft()
                    if node not in visited and node in expanded_indices:
                        visited.add(node)
                        cluster.add(node)
                        queue.extend(subgraph_adj.get(node, set()) - visited)

                if len(cluster) >= self.min_cluster_size:
                    filtered_indices.update(cluster)

            expanded_indices = filtered_indices if filtered_indices else expanded_indices

        # Step 5: Return documents sorted by similarity to query
        retrieved_docs = []
        for idx in expanded_indices:
            doc = self.graph.doc_metadata[idx].copy()
            doc['graph_distance'] = self._compute_min_distance(idx, entry_indices)
            doc['query_similarity'] = float(
                cosine_similarity(
                    [query_embedding],
                    [self.graph.doc_embeddings[idx]]
                )[0][0]
            )
            retrieved_docs.append(doc)

        # Sort by query similarity (direct relevance first)
        retrieved_docs.sort(key=lambda x: x['query_similarity'], reverse=True)

        return retrieved_docs

    def _compute_min_distance(self, node: int, entry_points: List[int]) -> int:
        """Compute minimum graph distance from node to any entry point."""
        if node in entry_points:
            return 0

        visited = set(entry_points)
        frontier = set(entry_points)
        distance = 0

        while frontier and distance < self.n_hops + 1:
            distance += 1
            next_frontier = set()
            for current in frontier:
                for neighbor in self.graph.adjacency_list.get(current, set()):
                    if neighbor == node:
                        return distance
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        return distance


class TopicGraphRAGRetriever:
    """
    Specialized retriever for MCQA tasks where documents are organized by topic.

    For each question, it:
    1. Gets all documents for the question's topic
    2. Uses graph traversal within the topic to find related documents
    3. Filters out potential distractors based on connectivity
    """

    def __init__(self,
                 docs: List[Dict],
                 embedder: BedrockDocumentEmbedder,
                 k_neighbors: int = 3,
                 similarity_threshold: float = 0.4,
                 n_hops: int = 2,
                 precomputed_embeddings_path: Optional[str] = None):
        """
        Initialize the topic-aware Graph RAG retriever.

        Args:
            docs: Full docs.json structure (list of topics with docs)
            embedder: Document embedder
            k_neighbors: Neighbors per document in graph
            similarity_threshold: Edge threshold
            n_hops: Traversal depth
            precomputed_embeddings_path: Path to precomputed embeddings
        """
        self.docs = docs
        self.embedder = embedder
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.n_hops = n_hops

        # Build topic ID -> docs mapping
        self.topic_docs: Dict[int, List[Dict]] = {}
        for topic in docs:
            topic_id = topic['topic_id']
            self.topic_docs[topic_id] = topic.get('docs', [])

        # Build per-topic graphs (lazy initialization)
        self.topic_graphs: Dict[int, DocumentSemanticGraph] = {}

        # Load or compute document embeddings
        self.all_doc_embeddings: Dict[int, np.ndarray] = {}
        if precomputed_embeddings_path and Path(precomputed_embeddings_path).exists():
            self._load_embeddings(precomputed_embeddings_path)

        print(f"   TopicGraphRAGRetriever initialized: {len(self.topic_docs)} topics")

    def _load_embeddings(self, path: str) -> None:
        """Load precomputed embeddings."""
        with open(path, 'rb') as f:
            self.all_doc_embeddings = pickle.load(f)
        print(f"   Loaded precomputed embeddings from: {path}")

    def _get_or_build_graph(self, topic_id: int) -> DocumentSemanticGraph:
        """Get or build the graph for a specific topic."""
        if topic_id not in self.topic_graphs:
            docs = self.topic_docs.get(topic_id, [])
            if not docs:
                return None

            graph = DocumentSemanticGraph(
                k_neighbors=self.k_neighbors,
                similarity_threshold=self.similarity_threshold,
                max_hops=self.n_hops
            )

            # Use precomputed embeddings if available
            precomputed = self.all_doc_embeddings.get(topic_id)
            graph.build_graph(docs, self.embedder, precomputed_embeddings=precomputed)

            self.topic_graphs[topic_id] = graph

        return self.topic_graphs[topic_id]

    def retrieve_for_question(self,
                              question: Dict,
                              n_entry_points: int = 2,
                              min_cluster_size: int = 1) -> Tuple[List[Dict], Dict]:
        """
        Retrieve relevant documents for an MCQA question.

        Args:
            question: Question dict with topic_id, target_event, options
            n_entry_points: Number of entry documents
            min_cluster_size: Minimum cluster size (1 = no filtering)

        Returns:
            Tuple of (retrieved_docs, retrieval_metadata)
        """
        topic_id = question['topic_id']

        # Build query from target event + all options
        query_parts = [question['target_event']]
        for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
            if opt in question:
                query_parts.append(question[opt])
        query = ' '.join(query_parts)

        # Get or build topic graph
        graph = self._get_or_build_graph(topic_id)
        if graph is None:
            return [], {'error': f'No documents for topic {topic_id}'}

        # Create retriever for this graph
        retriever = GraphRAGRetriever(
            graph=graph,
            embedder=self.embedder,
            n_entry_points=n_entry_points,
            n_hops=self.n_hops,
            min_cluster_size=min_cluster_size
        )

        # Retrieve documents
        retrieved_docs = retriever.retrieve(query)

        # Compute metadata
        metadata = {
            'topic_id': topic_id,
            'total_topic_docs': len(self.topic_docs.get(topic_id, [])),
            'retrieved_docs': len(retrieved_docs),
            'entry_points': n_entry_points,
            'n_hops': self.n_hops
        }

        return retrieved_docs, metadata


def build_document_embeddings(docs_path: str,
                               output_path: str,
                               model_id: str = 'cohere.embed-v4:0',
                               dimensions: int = 1024,
                               resume: bool = True) -> None:
    """
    Build and save document-level embeddings for all topics.

    Args:
        docs_path: Path to docs.json
        output_path: Path to save embeddings pickle
        model_id: Bedrock embedding model ID
        dimensions: Embedding dimensions
        resume: If True, resume from existing partial embeddings
    """
    print(f"\n{'='*60}")
    print("BUILDING DOCUMENT EMBEDDINGS")
    print(f"{'='*60}")
    print(f"   Input: {docs_path}")
    print(f"   Output: {output_path}")
    print(f"   Model: {model_id}")
    print(f"   Dimensions: {dimensions}")

    # Load docs
    print(f"\n>> Loading documents...", flush=True)
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    total_topics = len(docs)
    total_docs = sum(len(t.get('docs', [])) for t in docs)
    print(f"   Found {total_topics} topics with {total_docs} total documents")

    # Check for existing embeddings to resume
    all_embeddings = {}
    if resume and Path(output_path).exists():
        print(f"\n>> Loading existing embeddings for resume...", flush=True)
        with open(output_path, 'rb') as f:
            all_embeddings = pickle.load(f)
        print(f"   Found {len(all_embeddings)} existing topic embeddings")

    # Initialize embedder
    print(f"\n>> Initializing embedder...", flush=True)
    embedder = BedrockDocumentEmbedder(model_id=model_id, dimensions=dimensions)

    # Count already processed docs
    docs_processed = sum(
        len([t for t in docs if t['topic_id'] == tid][0].get('docs', []))
        for tid in all_embeddings.keys()
        if any(t['topic_id'] == tid for t in docs)
    )

    print(f"\n>> Embedding documents by topic...", flush=True)
    print(f"{'='*60}")

    skipped = 0
    for i, topic in enumerate(docs):
        topic_id = topic['topic_id']
        topic_docs = topic.get('docs', [])

        if not topic_docs:
            continue

        # Skip already processed topics
        if topic_id in all_embeddings:
            skipped += 1
            print(f"\n[Topic {i+1}/{total_topics}] ID={topic_id} - SKIPPED (already embedded)", flush=True)
            continue

        print(f"\n[Topic {i+1}/{total_topics}] ID={topic_id}")
        print(f"   Name: {topic['topic'][:60]}...")
        print(f"   Documents: {len(topic_docs)}", flush=True)

        # Get document texts
        texts = []
        for doc in topic_docs:
            text = doc.get('content', doc.get('text', ''))
            # Prepend title if available
            if 'title' in doc:
                text = f"{doc['title']}\n\n{text}"
            texts.append(text)

        # Embed with verbose progress
        embeddings = embedder.encode(texts, input_type='search_document', verbose=True)
        all_embeddings[topic_id] = embeddings

        docs_processed += len(topic_docs)
        print(f"   Shape: {embeddings.shape}")
        print(f"   Total progress: {docs_processed}/{total_docs} docs ({docs_processed/total_docs*100:.1f}%)", flush=True)

        # Save incrementally after each topic (for crash recovery)
        with open(output_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        print(f"   Saved checkpoint", flush=True)

    # Final save
    print(f"\n{'='*60}")
    print("SAVING EMBEDDINGS")
    print(f"{'='*60}")
    with open(output_path, 'wb') as f:
        pickle.dump(all_embeddings, f)

    print(f"   Saved to: {output_path}")
    print(f"   Topics: {len(all_embeddings)}")
    print(f"   Skipped: {skipped}")
    print(f"   Documents: {docs_processed}")
    print(f"\n>> DONE!", flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build document embeddings for Graph RAG')
    parser.add_argument('--docs', type=str, required=True, help='Path to docs.json')
    parser.add_argument('--output', type=str, required=True, help='Output path for embeddings')
    parser.add_argument('--model', type=str, default='cohere.embed-v4:0', help='Embedding model')
    parser.add_argument('--dimensions', type=int, default=1024, help='Embedding dimensions')

    args = parser.parse_args()

    build_document_embeddings(
        docs_path=args.docs,
        output_path=args.output,
        model_id=args.model,
        dimensions=args.dimensions
    )
