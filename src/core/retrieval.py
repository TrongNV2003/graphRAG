from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod
from langchain_neo4j import Neo4jGraph
from typing import List, Dict, Optional, Any

from src.engines.qdrant import QdrantVectorStore, create_qdrant_store
from src.services.sparse_encoder import get_sparse_encoder
from src.services.dense_encoder import get_dense_encoder
from src.config.setting import qdrant_config


class BaseRetrieval(ABC):
    def __init__(self, graph_db: Neo4jGraph):
        self.graph_db = graph_db
        
    @abstractmethod
    def retrieve(self, **kwargs: Any) -> List[Dict]:
        pass


class GraphRetrieval(BaseRetrieval):
    """
    Graph retrieval with nodes + relationships.
    Args:
        graph_db: Neo4jGraph instance
        graph_limit: Maximum number of graph to retrieve
    """
    def __init__(self, graph_db: Neo4jGraph, graph_limit: int = 10):
        super().__init__(graph_db)
        self.graph_limit = graph_limit

    def retrieve(self, target_entities: List[str], excluded_entities: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve relevant graph based on the entities list, excluding specified entities.
        Args:
            target_entities: List of entity names to search in graph DB
            excluded_entities: List of entity names to exclude from results
        Returns:
            List of graph triples as dicts
        """
        if not target_entities:
            return []
        
        excluded_entities = excluded_entities or []
        
        # Exclusion condition
        exclusion_clause = ""
        if excluded_entities:
            exclusion_clause = """
            AND NOT (toLower(n.id) IN $excluded OR toLower(n.entity_role) IN $excluded)
            AND NOT (toLower(m.id) IN $excluded OR toLower(m.entity_role) IN $excluded)
            """
        
        # Exact match on id/entity_role for any entity in the list
        cypher_exact = f"""
        UNWIND $entities as entity_name
        MATCH (n)
        WHERE (toLower(n.id) = toLower(entity_name) OR toLower(n.entity_role) = toLower(entity_name))
        {exclusion_clause}
        MATCH (n)-[r]-(m)
        WHERE 1=1 {exclusion_clause.replace('AND', '') if excluded_entities else ''}
        WITH DISTINCT n, r, m
        RETURN {{ id: n.id, entity_role: n.entity_role, type: labels(n)[0] }} AS source,
               type(r) AS relationship,
               {{ id: m.id, entity_role: m.entity_role, type: labels(m)[0] }} AS target
        LIMIT $limit
        """

        excluded_lower = [e.lower() for e in excluded_entities] if excluded_entities else []
        params = {"entities": target_entities, "excluded": excluded_lower, "limit": self.graph_limit}
        results = self._query_graph(cypher_exact, params)

        if results:
            return results

        # Fallback: substring match on id/entity_role for any entity in the list
        cypher_contains = f"""
        UNWIND $entities as entity_name
        MATCH (n)
        WHERE (toLower(n.id) CONTAINS toLower(entity_name) OR toLower(n.entity_role) CONTAINS toLower(entity_name))
        {exclusion_clause}
        MATCH (n)-[r]-(m)
        WHERE 1=1 {exclusion_clause.replace('AND', '') if excluded_entities else ''}
        WITH DISTINCT n, r, m
        RETURN {{ id: n.id, entity_role: n.entity_role, type: labels(n)[0] }} AS source,
               type(r) AS relationship,
               {{ id: m.id, entity_role: m.entity_role, type: labels(m)[0] }} AS target
        LIMIT $limit
        """

        return self._query_graph(cypher_contains, params)

    def _query_graph(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.graph_db.query(cypher, params=params or {})


class QdrantChunkRetrieval(BaseRetrieval):
    """Chunk retrieval using Qdrant Hybrid Search (Dense + Sparse).
    
    Args:
        graph_db: Neo4jGraph instance (used for loading chunks if needed)
        vector_store: QdrantVectorStore instance
        collection_name: Qdrant collection name
        top_k: Maximum number of results to return
        threshold: Minimum similarity score threshold
        auto_build: Whether to auto-create collection on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        vector_store: Optional[QdrantVectorStore] = None,
        collection_name: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        auto_build: bool = True,
    ):
        super().__init__(graph_db)
        self.vector_store = vector_store or create_qdrant_store()
        self.collection_name = collection_name or qdrant_config.collection_name
        self.top_k = top_k
        self.threshold = threshold
        
        # Dense encoder (singleton)
        self.dense_encoder = get_dense_encoder()
        
        # Sparse encoder (lazy loaded)
        self._sparse_encoder = None
        
        # Ensure collection exists
        if auto_build:
            self._ensure_collection()
    
    @property
    def sparse_encoder(self):
        if self._sparse_encoder is None:
            self._sparse_encoder = get_sparse_encoder()
        return self._sparse_encoder
    
    def _ensure_collection(self):
        """Ensure collection exists, create if not"""
        if not self.vector_store.collection_exists(self.collection_name):
            dimension = self.dense_encoder.get_dimension()
            self.vector_store.create_collection(
                name=self.collection_name,
                dimension=dimension,
                enable_sparse=True
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query text
            
        Returns:
            List of chunk dicts with scores
        """
        # Generate dense embedding
        query_vector = self.dense_encoder.encode(query)
        
        # Generate sparse embedding
        sparse_result = self.sparse_encoder.encode(query)
        
        # Hybrid search
        results = self.vector_store.hybrid_search(
            collection=self.collection_name,
            query_vector=query_vector,
            sparse_indices=sparse_result.indices,
            sparse_values=sparse_result.values,
            top_k=self.top_k,
            threshold=self.threshold if self.threshold > 0 else None
        )
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk_id": r.payload.get("chunk_id", r.id),
                "chunk_text": r.payload.get("content", ""),
                "score": r.score,
                "metadata": r.payload
            })
        
        return formatted_results

    def semantic_search(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks using dense vector search only (no sparse/keyword).
        
        Args:
            query: Search query text
            
        Returns:
            List of chunk dicts with scores
        """
        # Generate dense embedding only
        query_vector = self.dense_encoder.encode(query)
        
        # Dense-only search
        results = self.vector_store.search(
            collection=self.collection_name,
            vector=query_vector,
            top_k=self.top_k,
            threshold=self.threshold if self.threshold > 0 else None
        )
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk_id": r.payload.get("chunk_id", r.id),
                "chunk_text": r.payload.get("content", ""),
                "score": r.score,
                "metadata": r.payload
            })
        
        return formatted_results


class HybridRetrieval:
    """Hybrid RAG: combine graph triples (Neo4j) and chunk embeddings (Qdrant).
    
    Args:
        graph_db: Neo4jGraph instance
        vector_store: QdrantVectorStore instance (optional)
        graph_limit: Maximum number of graph results
        chunk_top_k: Maximum number of chunk results
        chunk_threshold: Minimum similarity score threshold for chunks
        auto_build: Whether to auto-create Qdrant collection on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        vector_store: Optional[QdrantVectorStore] = None,
        graph_limit: int = 10,
        chunk_top_k: int = 10,
        chunk_threshold: float = 0.0,
        auto_build: bool = True,
    ):
        self.graph_retrieval = GraphRetrieval(graph_db=graph_db, graph_limit=graph_limit)
        self.chunk_retrieval = QdrantChunkRetrieval(
            graph_db=graph_db,
            vector_store=vector_store,
            top_k=chunk_top_k,
            threshold=chunk_threshold,
            auto_build=auto_build
        )

    def retrieve(self, query: str, target_entities: List[str], excluded_entities: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Hybrid retrieval combining graph triples and semantic chunks.
        
        Args:
            query: Search query text for semantic chunk retrieval
            target_entities: List of entity names for graph retrieval
            excluded_entities: List of entity names to exclude from graph results
            
        Returns:
            Dict with 'graph' and 'chunk' keys
        """
        graph_results = self.graph_retrieval.retrieve(target_entities, excluded_entities)
        chunk_results = self.chunk_retrieval.retrieve(query)
        
        return {
            "graph": graph_results,
            "chunk": chunk_results,
        }


if __name__ == "__main__":
    from src.config.setting import neo4j_config

    graph_db = Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password,
    )
    
    retrieval = HybridRetrieval(
        graph_db,
        graph_limit=10,
        chunk_top_k=5,
        chunk_threshold=0.0,
        auto_build=True
    )
    
    target_entities = ["Elizabeth"]
    query = "Tell me about Elizabeth and her relationships."
    results = retrieval.retrieve(query=query, target_entities=target_entities)
    
    logger.info(f"Graph results: {len(results['graph'])}")
    logger.info(f"Chunk results: {len(results['chunk'])}")
    print("Graph:", results["graph"][:3])
    print("\nChunks:")
    for i, chunk in enumerate(results["chunk"][:3], 1):
        print(f"  {i}. Score: {chunk['score']:.3f} | {chunk['chunk_text'][:100]}...")
