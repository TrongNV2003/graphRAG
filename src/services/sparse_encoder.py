"""Sparse Vector Encoder using FastEmbed SPLADE model"""

import logging
from typing import List, Optional
from src.models import SparseVector

logger = logging.getLogger(__name__)


class SparseEncoder:
    """
    Sparse vector encoder using FastEmbed SPLADE.
    
    SPLADE (SParse Lexical AnD Expansion) is a neural model that learns to expand 
    queries and documents with related terms, producing high-quality sparse vectors.
    
    Features:
    - Neural-based token weighting (better than TF-IDF/BM25)
    - Query/document expansion for better recall
    - Learned importance weights
    """
    
    # SPLADE models
    MODELS = {
        "splade-multilingual": "Qdrant/bm42-all-minilm-l6-v2-attentions",  # Multilingual BM42
        "splade-pp": "prithivida/Splade_PP_en_v1",  # SPLADE++ for English
    }
    
    def __init__(
        self, 
        model_name: str = "splade-multilingual",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize SPLADE encoder.
        
        Args:
            model_name: Model to use ("splade-pp" or "splade-multilingual")
            cache_dir: Directory to cache model files
        """
        self._model = None
        self._model_name = self.MODELS.get(model_name, model_name)
        self._cache_dir = cache_dir
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of the SPLADE model"""
        if self._initialized:
            return
        
        try:
            from fastembed import SparseTextEmbedding
            
            logger.info(f"Loading SPLADE model: {self._model_name}")
            self._model = SparseTextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir
            )
            self._initialized = True
            logger.info("SPLADE model loaded successfully")
            
        except ImportError:
            raise ImportError("fastembed required for SPLADE. Install: pip install fastembed")
        except Exception as e:
            raise Exception(f"Failed to load SPLADE model: {e}")
    
    def encode(self, text: str) -> SparseVector:
        """
        Encode text into a sparse vector using SPLADE.
        
        Args:
            text: Input text to encode
            
        Returns:
            SparseVector with indices and values
        """
        if not text or not text.strip():
            return SparseVector(indices=[], values=[])
        
        self._ensure_initialized()
        
        try:
            # FastEmbed returns a generator, convert to list
            embeddings = list(self._model.embed([text]))
            
            if not embeddings:
                return SparseVector(indices=[], values=[])
            
            sparse_emb = embeddings[0]
            
            # Convert numpy arrays to Python lists
            return SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist()
            )
            
        except Exception as e:
            logger.error(f"SPLADE encoding failed: {e}")
            return SparseVector(indices=[], values=[])
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[SparseVector]:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            List of SparseVector objects
        """
        if not texts:
            return []
        
        self._ensure_initialized()
        
        try:
            # FastEmbed handles batching internally
            embeddings = list(self._model.embed(texts, batch_size=batch_size))
            
            results = []
            for sparse_emb in embeddings:
                results.append(SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"SPLADE batch encoding failed: {e}")
            return [SparseVector(indices=[], values=[]) for _ in texts]
    
    @staticmethod
    def list_available_models() -> List[dict]:
        """List all available sparse embedding models"""
        try:
            from fastembed import SparseTextEmbedding
            return SparseTextEmbedding.list_supported_models()
        except ImportError:
            return []


# Singleton instance
_encoder: Optional[SparseEncoder] = None

def get_sparse_encoder(model_name: str = "splade-multilingual") -> SparseEncoder:
    """
    Get singleton sparse encoder instance.
    
    Args:
        model_name: "splade-pp" (English) or "splade-multilingual" 
    """
    global _encoder
    if _encoder is None:
        _encoder = SparseEncoder(model_name=model_name)
    return _encoder
