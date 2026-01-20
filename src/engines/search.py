"""
DEPRECATED: This module is no longer used in the current architecture.
The system now uses:
- Neo4j Fulltext Index for fuzzy entity search (src/core/retrieval.py)
- Qdrant for hybrid vector search (src/engines/qdrant.py)

This file is kept for reference only. Consider removing in future cleanup.
"""
import warnings
warnings.warn(
    "src.engines.search is deprecated and not used. "
    "Use src.core.retrieval.GraphRetrieval or src.engines.qdrant instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import json
import bm25s
import faiss
import numpy as np
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

class BaseRetrievalEngine(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index a list of documents.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Search the indexed documents.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            
        Returns:
            List of document dictionaries with relevance scores.
        """
        pass
    
    @abstractmethod
    def save(self, directory: str) -> bool:
        """Save the index to disk.
        
        Args:
            directory: Directory path to save the index.
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> 'BaseRetrievalEngine':
        """Load an index from disk.
        
        Args:
            directory: Directory path containing the saved index.
            
        Returns:
            BaseRetrievalEngine: Loaded retrieval engine instance.
        """
        pass


class BM25RetrievalEngine(BaseRetrievalEngine):
    """BM25 implementation of the retrieval engine using bm25s library.
    
    This implementation provides document indexing and search functionality using
    the BM25 ranking algorithm. It supports multiple languages and includes features
    like document metadata management and index persistence.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize the BM25 retrieval engine.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.document_store = []
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index a list of documents.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful.
        """
        try:          
            self.document_store = documents

            corpus_texts = [doc.get('content', '') for doc in documents]
            corpus_tokens = bm25s.tokenize(corpus_texts)
            
            self.retriever.index(corpus_tokens)
            
            return True
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return False

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> tuple:
        """Search the indexed documents.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            
        Returns:
            tuple: (doc_ids, scores) as numpy arrays after filtering by min_score.
        """
        if not query.strip():
            return [], []
        
        try:                
            query_tokens = bm25s.tokenize(query)
            
            # Retrieve results
            doc_ids, scores = self.retriever.retrieve(query_tokens, k=top_k)
            
            # Filter by min_score using boolean indexing
            mask = scores >= min_score
            return doc_ids[mask], scores[mask]
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return [], []
    
    def save(self, directory: str) -> bool:
        """Save the index to disk.
        
        Args:
            directory: Directory path to save the index.
            
        Returns:
            bool: True if save was successful.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save BM25 index
            self.retriever.save(directory)
            
            # Save additional engine state
            state = {
                'k1': self.k1,
                'b': self.b,
                'corpus' : self.document_store
            }
            
            with open(os.path.join(directory, 'engine_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'BM25RetrievalEngine':
        """Load an index from disk.
        
        Args:
            directory: Directory path containing the saved index.
            
        Returns:
            BM25RetrievalEngine: Loaded retrieval engine instance.
        """
        try:
            # Load engine state
            with open(os.path.join(directory, 'engine_state.json'), 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Create new engine instance with saved parameters
            engine = cls(
                k1=state['k1'],
                b=state['b'],
            )
            
            # Load BM25 index
            engine.retriever = bm25s.BM25.load(directory, load_corpus = True)
            
            # Restore document store and count
            engine.document_store = state['corpus']
            

            
            return engine
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None


class DenseRetrievalEngine(BaseRetrievalEngine):
    """Dense retrieval engine implementation using sentence transformers and FAISS.
    
    This implementation provides document indexing and search functionality using
    dense embeddings generated by sentence transformers and FAISS for efficient
    similarity search.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dimension: int = 384,
                 metric: str = "cosine", 
                 batch_size: int = 32):
        """Initialize the dense retrieval engine.
        
        Args:
            model_name: Name of the sentence transformer model (default: all-MiniLM-L6-v2).
            dimension: Dimension of embeddings (default: 384).
            metric: Distance metric for FAISS index (default: cosine).
        """
        self.model_name = model_name
        self.dimension = dimension
        self.metric = metric
        
        self.model = SentenceTransformer(model_name, token=False)
        if self.model_name == "vinai/phobert-base-v2":
            self.model.max_seq_length = 256
        
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
            
        self.document_store = []
        self.doc_count = 0
        self.batch_size = batch_size
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index a list of documents.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful.
        """
        try:
            self.document_store = documents
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            corpus = documents
            
            embeddings = self.model.encode(corpus, convert_to_numpy=True, batch_size = self.batch_size)
            
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings)
            
            self.doc_count = len(self.document_store)
            
            return True
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return False
    
    def search(self, query: str, 
               top_k: int = 5, 
               min_score: float = 0.0) -> tuple:
        """Search the indexed documents.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score threshold.
            
        Returns:
            tuple: (doc_ids, scores) as numpy arrays after filtering by min_score.
        """
        if not query.strip():
            return [], []
        
        try:
            adjusted_top_k = min(top_k, self.doc_count)
            if adjusted_top_k == 0:
                return [], []
            
            query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            
            if self.metric == "cosine":
                faiss.normalize_L2(query_embedding)
            
            scores, doc_ids = self.index.search(query_embedding, adjusted_top_k)
            
            # Convert to 1D arrays
            scores = scores[0]
            doc_ids = doc_ids[0]
            
            # Filter by min_score
            if self.metric == "cosine":
                mask = scores >= min_score
            else:
                # Convert L2 distance to similarity score (1 / (1 + distance))
                scores = 1 / (1 + scores)
                mask = scores >= min_score
            
            return doc_ids[mask], scores[mask]
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return [], []
    
    def save(self, directory: str) -> bool:
        """Save the index to disk.
        
        Args:
            directory: Directory path to save the index.
            
        Returns:
            bool: True if save was successful.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(directory, 'faiss.index'))
            
            # Save additional engine state
            state = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'metric': self.metric,
                'doc_count': self.doc_count,
                'document_store': self.document_store
            }
            
            with open(os.path.join(directory, 'engine_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'DenseRetrievalEngine':
        """Load an index from disk.
        
        Args:
            directory: Directory path containing the saved index.
            
        Returns:
            DenseRetrievalEngine: Loaded retrieval engine instance.
        """
        try:
            with open(os.path.join(directory, 'engine_state.json'), 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Create new engine instance with saved parameters
            engine = cls(
                model_name=state['model_name'],
                dimension=state['dimension'],
                metric=state['metric']
            )
            
            # Restore document store and count
            engine.document_store = state['document_store']
            engine.doc_count = state['doc_count']
            
            engine.index = faiss.read_index(os.path.join(directory, 'faiss.index'))
            
            return engine
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None


class HybridRetrievalEngine(BaseRetrievalEngine):
    """Hybrid retrieval engine combining BM25 and dense retrieval using RRF (Reciprocal Rank Fusion).
    
    This implementation combines results from BM25 and dense retrievers using
    reciprocal rank fusion (RRF) to leverage both sparse and dense representations.
    """
    
    def __init__(
        self, 
        bm25_params: Dict[str, Any] = None,
        dense_params: Dict[str, Any] = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: float = 60.0
    ):
        """Initialize the hybrid retrieval engine.
        
        Args:
            bm25_params: Parameters for BM25 engine initialization (default: None).
            dense_params: Parameters for dense engine initialization (default: None).
            bm25_weight: Weight for BM25 scores in fusion (default: 0.5).
            dense_weight: Weight for dense scores in fusion (default: 0.5).
            rrf_k: Constant for RRF score computation (default: 60.0).
        """
        self.bm25_engine = BM25RetrievalEngine(**(bm25_params or {}))
        self.dense_engine = DenseRetrievalEngine(**(dense_params or {}))
        
        # Set fusion parameters
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        
        self.document_store = []
        self.doc_count = 0
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents using both BM25 and dense engines.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful.
        """
        try:
            # Reset document store
            self.document_store = documents
            
            # Store documents
            self.doc_count = len(self.document_store)
            
            # Index documents sequentially
            bm25_success = self.bm25_engine.index_documents(documents)
            dense_success = self.dense_engine.index_documents(documents)
            
            return bm25_success and dense_success
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return False
    
    def _normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize scores to [0,1] range using different methods.
        
        Args:
            scores: Raw scores array.
            method: Normalization method ('minmax', 'softmax', or 'zscore').
            
        Returns:
            numpy.ndarray: Normalized scores.
        """
        if len(scores) == 0:
            return scores
            
        if method == 'softmax':
            # Softmax normalization
            exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
            return exp_scores / np.sum(exp_scores)
        elif method == 'zscore':
            # Z-score normalization
            std = np.std(scores)
            if std == 0:
                return np.ones_like(scores)
            mean = np.mean(scores)
            return (scores - mean) / std
        else:  # minmax
            # Min-max normalization
            score_min = np.min(scores)
            score_max = np.max(scores)
            if score_max == score_min:
                return np.ones_like(scores)
            return (scores - score_min) / (score_max - score_min)
    
    def _apply_bm25_strategy(self, scores: np.ndarray, strategy: str = 'bm25') -> np.ndarray:
        """Apply different scoring strategies to BM25 scores.
        
        Args:
            scores: Raw BM25 scores.
            strategy: Scoring strategy ('bm25', 'log', or 'sqrt').
            
        Returns:
            numpy.ndarray: Transformed scores.
        """
        if strategy == 'log':
            return np.log1p(scores)
        elif strategy == 'sqrt':
            return np.sqrt(scores)
        return scores  # raw bm25
    
    def _compute_rrf_scores(self, doc_ids_scores: List[tuple], bm25_strategy: str = 'bm25') -> tuple:
        """Compute RRF scores for a list of document IDs and scores.
        
        Args:
            doc_ids_scores: List of (doc_ids, scores) tuples from retrievers.
            bm25_strategy: Strategy for BM25 score transformation.
            
        Returns:
            tuple: (unique_doc_ids, final_scores) as numpy arrays.
        """
        if not doc_ids_scores:
            return np.array([]), np.array([])
            
        # Process BM25 and dense scores separately
        bm25_doc_ids, bm25_scores = doc_ids_scores[0][1]
        dense_doc_ids, dense_scores = doc_ids_scores[1][1]
        
        # Handle empty results
        if len(bm25_scores) == 0 and len(dense_scores) == 0:
            return np.array([]), np.array([])
        elif len(bm25_scores) == 0:
            return np.array(dense_doc_ids), self._normalize_scores(dense_scores, 'softmax')
        elif len(dense_scores) == 0:
            return np.array(bm25_doc_ids), self._normalize_scores(self._apply_bm25_strategy(bm25_scores, bm25_strategy), 'softmax')
        
        # Apply BM25 strategy and normalize scores using softmax for better score distribution
        bm25_scores = self._normalize_scores(self._apply_bm25_strategy(bm25_scores, bm25_strategy), 'softmax')
        dense_scores = self._normalize_scores(dense_scores, 'softmax')
        
        # Get ranks (1-based)
        bm25_ranks = np.argsort(-bm25_scores).argsort() + 1
        dense_ranks = np.argsort(-dense_scores).argsort() + 1
        
        # Compute RRF scores with adaptive k
        rrf_scores = {}
        adaptive_k = self.rrf_k * np.sqrt(max(len(bm25_ranks), len(dense_ranks)))  # Scale k with result size
        
        for i, doc_id in enumerate(bm25_doc_ids):
            rrf_scores[doc_id] = self.bm25_weight / (adaptive_k + bm25_ranks[i])
            
        for i, doc_id in enumerate(dense_doc_ids):
            dense_score = self.dense_weight / (adaptive_k + dense_ranks[i])
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += dense_score
            else:
                rrf_scores[doc_id] = dense_score
        
        # Convert to arrays
        unique_doc_ids = np.array(list(rrf_scores.keys()))
        final_scores = np.array(list(rrf_scores.values()))
        
        return unique_doc_ids, final_scores
    
    def search(self, query: str, 
               top_k: int = 5, 
               min_score: float = 0.0,
               bm25_strategy: str = 'bm25') -> tuple:
        """Search using both engines and combine results using RRF.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            bm25_strategy: Strategy for BM25 score transformation ('bm25', 'log', or 'sqrt').
            
        Returns:
            tuple: (doc_ids, scores) as numpy arrays.
        """
        if not query.strip():
            return [], []
        
        try:
            # Adjust top_k for individual retrievers
            adjusted_top_k = min(top_k * 2, self.doc_count)  # Get more results for better fusion
            if adjusted_top_k == 0:
                return [], []
            
            # Search with both engines sequentially
            bm25_results = self.bm25_engine.search(query, adjusted_top_k)
            dense_results = self.dense_engine.search(query, adjusted_top_k)
            
            # Combine results using RRF
            doc_ids_scores = [
                (self.bm25_weight, bm25_results),
                (self.dense_weight, dense_results)
            ]
            
            unique_doc_ids, final_scores = self._compute_rrf_scores(doc_ids_scores, bm25_strategy)
            
            # Sort by score
            rank_order = np.argsort(-final_scores)  # Descending order
            
            # Apply top_k and min_score filters
            mask = final_scores[rank_order] >= min_score
            filtered_indices = rank_order[mask][:top_k]
            
            result_doc_ids = unique_doc_ids[filtered_indices]
            result_scores = final_scores[filtered_indices]
            
            return result_doc_ids, result_scores
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return [], []
    
    def save(self, directory: str) -> bool:
        """Save both engines and hybrid state to disk.
        
        Args:
            directory: Directory path to save the engines.
            
        Returns:
            bool: True if save was successful.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save individual engines
            bm25_dir = os.path.join(directory, 'bm25')
            dense_dir = os.path.join(directory, 'dense')
            os.makedirs(bm25_dir, exist_ok=True)
            os.makedirs(dense_dir, exist_ok=True)
            
            bm25_success = self.bm25_engine.save(bm25_dir)
            dense_success = self.dense_engine.save(dense_dir)
            
            # Save hybrid engine state
            state = {
                'bm25_weight': self.bm25_weight,
                'dense_weight': self.dense_weight,
                'rrf_k': self.rrf_k,
                'doc_count': self.doc_count,
                'document_store': self.document_store
            }
            
            with open(os.path.join(directory, 'engine_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return bm25_success and dense_success
        except Exception as e:
            print(f"Error saving engines: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'HybridRetrievalEngine':
        """Load both engines and hybrid state from disk.
        
        Args:
            directory: Directory path containing the saved engines.
            
        Returns:
            HybridRetrievalEngine: Loaded hybrid engine instance.
        """
        try:
            with open(os.path.join(directory, 'engine_state.json'), 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Create new engine instance
            engine = cls(
                bm25_weight=state['bm25_weight'],
                dense_weight=state['dense_weight'],
                rrf_k=state['rrf_k']
            )
            
            # Restore document store and count
            engine.document_store = state['document_store']
            engine.doc_count = state['doc_count']
            
            # Load individual engines
            bm25_dir = os.path.join(directory, 'bm25')
            dense_dir = os.path.join(directory, 'dense')
            
            engine.bm25_engine = BM25RetrievalEngine.load(bm25_dir)
            engine.dense_engine = DenseRetrievalEngine.load(dense_dir)
            
            if not engine.bm25_engine or not engine.dense_engine:
                return None
            
            return engine
        except Exception as e:
            print(f"Error loading engines: {str(e)}")
            return None