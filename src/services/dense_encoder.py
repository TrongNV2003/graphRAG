"""Dense Vector Encoder - SentenceTransformer and OpenAI support"""

import logging
from typing import List, Optional

from src.config.setting import embed_config, api_config

logger = logging.getLogger(__name__)


class DenseEncoder:
    """
    Dense vector encoder with singleton pattern.
    
    Supports:
    - Local models via SentenceTransformer (default)
    - OpenAI embedding models (text-embedding-3-small/large)
    """
    
    OPENAI_MODELS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        provider: str = "local"
    ):
        """
        Initialize dense encoder.
        
        Args:
            model_name: Model name (e.g., "contextboxai/halong_embedding")
            provider: "local" (SentenceTransformer) or "openai"
        """
        self._model_name = model_name or embed_config.embedder_model
        self._provider = provider.lower()
        self._model = None
        self._client = None
        self._initialized = False
        self._dimension = None
    
    def _ensure_initialized(self):
        """Lazy initialization of the embedding model"""
        if self._initialized:
            return
        
        if self._provider == "openai":
            self._init_openai()
        else:
            self._init_local()
        
        self._initialized = True
    
    def _init_local(self):
        """Initialize local SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading local model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")
            
        except ImportError:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
        except Exception as e:
            raise Exception(f"Failed to load model '{self._model_name}': {e}")
    
    def _init_openai(self):
        """Initialize OpenAI embedding client"""
        try:
            from openai import OpenAI
            
            if self._model_name not in self.OPENAI_MODELS:
                raise ValueError(f"Unknown OpenAI model: {self._model_name}. "
                               f"Supported: {list(self.OPENAI_MODELS.keys())}")
            
            api_key = api_config.openai_api_key or api_config.api_key
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            
            logger.info(f"Using OpenAI embedding model: {self._model_name}")
            self._client = OpenAI(api_key=api_key)
            self._dimension = self.OPENAI_MODELS[self._model_name]
            
        except ImportError:
            raise ImportError("openai required. Install: pip install openai")
    
    def encode(self, text: str) -> List[float]:
        """
        Encode single text into dense vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return []
        
        self._ensure_initialized()
        
        if self._provider == "openai":
            return self._encode_openai(text)
        else:
            return self._encode_local(text)
    
    def _encode_local(self, text: str) -> List[float]:
        """Encode using local SentenceTransformer"""
        try:
            embedding = self._model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local encoding failed: {e}")
            return []
    
    def _encode_openai(self, text: str) -> List[float]:
        """Encode using OpenAI API"""
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI encoding failed: {e}")
            return []
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        self._ensure_initialized()
        
        if self._provider == "openai":
            return self._encode_batch_openai(texts, batch_size)
        else:
            return self._encode_batch_local(texts, batch_size)
    
    def _encode_batch_local(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Batch encode using local model"""
        try:
            embeddings = self._model.encode(texts, batch_size=batch_size)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Local batch encoding failed: {e}")
            return [[] for _ in texts]
    
    def _encode_batch_openai(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Batch encode using OpenAI API"""
        embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self._client.embeddings.create(
                    model=self._model_name,
                    input=batch
                )
                # Sort by index to maintain order
                batch_embeddings = sorted(response.data, key=lambda x: x.index)
                embeddings.extend([e.embedding for e in batch_embeddings])
            
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI batch encoding failed: {e}")
            return [[] for _ in texts]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        self._ensure_initialized()
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self._model_name
    
    @property
    def provider(self) -> str:
        """Get provider name"""
        return self._provider


# Singleton instance
_encoder: Optional[DenseEncoder] = None


def get_dense_encoder(
    model_name: Optional[str] = None,
    provider: str = "local"
) -> DenseEncoder:
    """
    Get singleton dense encoder instance.
    
    Args:
        model_name: Model name (default from config)
        provider: "local" or "openai"
        
    Returns:
        DenseEncoder instance
    """
    global _encoder
    if _encoder is None:
        _encoder = DenseEncoder(model_name=model_name, provider=provider)
    return _encoder
