"""Qdrant Vector Store - Hybrid Search with Dense + Sparse vectors"""

import logging
from typing import List, Dict, Any, Optional

from src.models import VectorPoint, SearchResult
from src.config.setting import qdrant_config

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector store implementation.
    Supports both cloud and local Qdrant instances with hybrid search.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333
    ):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant Cloud URL
            api_key: Qdrant API key
            host: Local Qdrant host
            port: Local Qdrant port
        """
        self._url = url or qdrant_config.url
        self._api_key = api_key or qdrant_config.api_key
        self._host = host or qdrant_config.host
        self._port = port or qdrant_config.port
        self._client = None
    
    def _get_client(self):
        """Get or create sync Qdrant client"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                
                if self._url:
                    self._client = QdrantClient(
                        url=self._url,
                        api_key=self._api_key
                    )
                else:
                    self._client = QdrantClient(
                        host=self._host,
                        port=self._port
                    )
                
                logger.info(f"Connected to Qdrant at {self._url or f'{self._host}:{self._port}'}")
            except ImportError:
                raise ImportError("qdrant-client required. Install with: pip install qdrant-client")
        
        return self._client
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        enable_sparse: bool = True
    ) -> bool:
        """
        Create a new collection in Qdrant with named vectors.
        
        Args:
            name: Collection name
            dimension: Dense vector dimension
            distance: Distance metric (cosine, euclidean, dot)
            enable_sparse: Enable sparse vectors for hybrid search
        """
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams
        
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        client = self._get_client()
        
        try:
            # Named vectors config for hybrid search
            vectors_config = {
                "dense": VectorParams(
                    size=dimension,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            }
            
            # Sparse vectors config for keyword search
            sparse_vectors_config = None
            if enable_sparse:
                sparse_vectors_config = {
                    "sparse": SparseVectorParams()
                }
            
            client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            logger.info(f"Created collection '{name}' with dimension {dimension}, sparse={enable_sparse}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{name}' already exists")
                return True
            logger.error(f"Failed to create collection '{name}': {e}")
            raise
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        client = self._get_client()
        
        try:
            client.delete_collection(collection_name=name)
            logger.info(f"Deleted collection '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """Check if collection exists"""
        client = self._get_client()
        
        try:
            collections = client.get_collections()
            return any(c.name == name for c in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection '{name}': {e}")
            return False
    
    def upsert(
        self,
        collection: str,
        points: List[VectorPoint]
    ) -> bool:
        """
        Insert or update vectors in Qdrant.
        Supports both dense-only and hybrid (dense + sparse) vectors.
        """
        from qdrant_client.models import PointStruct, SparseVector
        
        client = self._get_client()
        
        qdrant_points = []
        for p in points:
            # Build named vectors
            vectors = {
                "dense": p.vector
            }
            
            # Add sparse vector if provided
            if p.sparse_indices and p.sparse_values:
                vectors["sparse"] = SparseVector(
                    indices=p.sparse_indices,
                    values=p.sparse_values
                )
            
            qdrant_points.append(PointStruct(
                id=p.id,
                vector=vectors,
                payload=p.payload
            ))
        
        try:
            client.upsert(
                collection_name=collection,
                points=qdrant_points
            )
            logger.debug(f"Upserted {len(points)} points to '{collection}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert to '{collection}': {e}")
            raise
    
    def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors (dense only)"""
        client = self._get_client()
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        try:
            results = client.query_points(
                collection_name=collection,
                query=vector,
                using="dense",
                limit=top_k,
                query_filter=query_filter,
                score_threshold=threshold
            )
        except Exception as e:
            if "Not existing vector name" in str(e) or "dense" in str(e):
                # Fallback to unnamed vector (old collections)
                logger.debug(f"Collection '{collection}' uses unnamed vectors, falling back")
                results = client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=threshold
                )
            else:
                logger.error(f"Search failed in '{collection}': {e}")
                raise
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {}
            )
            for r in results.points
        ]
    
    def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Qdrant hybrid search using prefetch + RRF fusion.
        Combines dense (semantic) and sparse (BM25-style keyword) search.
        
        Args:
            query_vector: Dense embedding vector
            sparse_indices: Sparse vector indices (from sparse encoder)
            sparse_values: Sparse vector values (from sparse encoder)
        """
        from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
        
        client = self._get_client()
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        try:
            prefetch_limit = top_k * 2
            
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=prefetch_limit
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=prefetch_limit
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=threshold
            )
            
            search_results = [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {}
                )
                for r in results.points
            ]
            
            logger.debug(f"Hybrid search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to dense-only: {e}")
            return self.search(
                collection=collection,
                vector=query_vector,
                top_k=top_k,
                threshold=threshold,
                filter_conditions=filter_conditions
            )
    
    def delete(
        self,
        collection: str,
        ids: List[str]
    ) -> bool:
        """Delete vectors by ID"""
        from qdrant_client.models import PointIdsList
        
        client = self._get_client()
        
        try:
            client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids)
            )
            logger.debug(f"Deleted {len(ids)} points from '{collection}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from '{collection}': {e}")
            return False
    
    def delete_by_filter(
        self,
        collection: str,
        filter_conditions: Dict[str, Any]
    ) -> int:
        """Delete vectors matching filter conditions"""
        from qdrant_client.models import FilterSelector
        
        client = self._get_client()
        query_filter = self._build_filter(filter_conditions)
        
        try:
            client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=query_filter)
            )
            logger.debug(f"Deleted points from '{collection}' with filter")
            return 1
        except Exception as e:
            logger.error(f"Failed to delete by filter from '{collection}': {e}")
            return 0
    
    def _build_filter(self, conditions: Dict[str, Any]):
        """Build Qdrant filter from conditions dict"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        must_conditions = []
        
        for field, value in conditions.items():
            must_conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        return Filter(must=must_conditions)
    
    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        client = self._get_client()
        
        try:
            info = client.get_collection(collection_name=name)
            return {
                "name": name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{name}': {e}")
            return None


def create_qdrant_store() -> QdrantVectorStore:
    """Factory function to create QdrantVectorStore instance"""
    return QdrantVectorStore(
        url=qdrant_config.url,
        api_key=qdrant_config.api_key,
        host=qdrant_config.host,
        port=qdrant_config.port
    )
