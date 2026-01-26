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
    """
    def __init__(self, graph_db: Neo4jGraph):
        super().__init__(graph_db)

    def retrieve(
        self,
        target_entities: List[str],
        excluded_entities: Optional[List[str]] = None,
        graph_limit: int = 10
    ) -> List[Dict]:
        """Retrieve relevant graph based on the entities list, excluding specified entities.
        
        Strategy:
        - Exact Match (high precision)
        - Fulltext Fuzzy Search (handles typos)
        - Substring Match (fallback)
        
        Args:
            target_entities: List of entity names to search in graph DB
            excluded_entities: List of entity names to exclude from results
            graph_limit: Maximum number of graph to retrieve
        Returns:
            List of graph triples as dicts
        """
        if not target_entities:
            return []
        
        excluded_entities = excluded_entities or []
        excluded_lower = [e.lower() for e in excluded_entities] if excluded_entities else []
        
        # Exact Match
        results = self._query_exact_match(target_entities, excluded_lower, graph_limit)
        if results:
            logger.debug(f"Exact match found {len(results)} results")
            return results
        
        # Fulltext Fuzzy Search
        results = self._query_fulltext(target_entities, excluded_lower, graph_limit)
        if results:
            logger.info(f"Fulltext fuzzy search found {len(results)} results")
            return results
        
        # Substring (CONTAINS) Fallback
        results = self._query_contains(target_entities, excluded_lower, graph_limit)
        if results:
            logger.info(f"Substring match found {len(results)} results")
        else:
            logger.warning(f"No graph results found for entities: {target_entities}")
        
        return results

    def _build_lucene_query(self, query: str) -> str:
        """Build Lucene query with escaping and fuzzy matching.
        
        Returns: query with prefix (*) and fuzzy (~1) matching.
        """
        import re
        # Escape Lucene special characters
        escaped = re.sub(r'([+\-&|!(){}[\]^"~*?:\\/])', r'\\\1', query.strip())
        # Combine prefix match and fuzzy match (edit distance 1)
        return f"{escaped}* OR {escaped}~1"

    def _query_fulltext(self, target_entities: List[str], excluded_lower: List[str], graph_limit: int) -> List[Dict]:
        """Query using Neo4j fulltext index for fuzzy matching."""
        from src.config.setting import retrieval_config
        
        try:
            all_results = []
            for entity_name in target_entities:
                lucene_query = self._build_lucene_query(entity_name)
                
                cypher = """
                CALL db.index.fulltext.queryNodes("entity_fulltext_index", $lucene_query)
                YIELD node, score
                WHERE score >= $min_score
                  AND NOT (toLower(node.id) IN $excluded OR toLower(node.entity_role) IN $excluded)
                WITH node, score
                MATCH (node)-[r]-(m)
                WHERE NOT (toLower(m.id) IN $excluded OR toLower(m.entity_role) IN $excluded)
                WITH DISTINCT node, r, m, score
                ORDER BY score DESC
                RETURN { id: node.id, entity_role: node.entity_role, type: labels(node)[0] } AS source,
                       type(r) AS relationship,
                       { id: m.id, entity_role: m.entity_role, type: labels(m)[0] } AS target
                LIMIT $limit
                """
                
                params = {
                    "lucene_query": lucene_query,
                    "min_score": retrieval_config.fuzzy_min_score,
                    "excluded": excluded_lower,
                    "limit": graph_limit
                }
                
                results = self._query_graph(cypher, params)
                all_results.extend(results)
            
            # Deduplicate
            seen = set()
            unique_results = []
            for r in all_results:
                key = (r.get("source", {}).get("id"), r.get("relationship"), r.get("target", {}).get("id"))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            return unique_results[:graph_limit]
            
        except Exception as e:
            logger.warning(f"Fulltext search failed: {e}. Falling back to substring match.")
            return []

    def _query_exact_match(self, target_entities: List[str], excluded_lower: List[str], graph_limit: int) -> List[Dict]:
        """Query with exact match on id/entity_role."""
        exclusion_clause = ""
        if excluded_lower:
            exclusion_clause = """
            AND NOT (toLower(n.id) IN $excluded OR toLower(n.entity_role) IN $excluded)
            AND NOT (toLower(m.id) IN $excluded OR toLower(m.entity_role) IN $excluded)
            """
        
        cypher = f"""
        UNWIND $entities as entity_name
        MATCH (n)
        WHERE (toLower(n.id) = toLower(entity_name) OR toLower(n.entity_role) = toLower(entity_name))
        {exclusion_clause}
        MATCH (n)-[r]-(m)
        WHERE 1=1 {exclusion_clause.replace('AND', '') if excluded_lower else ''}
        WITH DISTINCT n, r, m
        RETURN {{ id: n.id, entity_role: n.entity_role, type: labels(n)[0] }} AS source,
               type(r) AS relationship,
               {{ id: m.id, entity_role: m.entity_role, type: labels(m)[0] }} AS target
        LIMIT $limit
        """
        
        params = {
            "entities": target_entities,
            "excluded": excluded_lower,
            "limit": graph_limit
        }

        return self._query_graph(cypher, params)

    def _query_contains(self, target_entities: List[str], excluded_lower: List[str], graph_limit: int) -> List[Dict]:
        """Query with substring (CONTAINS) matching as final fallback."""
        exclusion_clause = ""
        if excluded_lower:
            exclusion_clause = """
            AND NOT (toLower(n.id) IN $excluded OR toLower(n.entity_role) IN $excluded)
            AND NOT (toLower(m.id) IN $excluded OR toLower(m.entity_role) IN $excluded)
            """
        
        cypher = f"""
        UNWIND $entities as entity_name
        MATCH (n)
        WHERE (toLower(n.id) CONTAINS toLower(entity_name) OR toLower(n.entity_role) CONTAINS toLower(entity_name))
        {exclusion_clause}
        MATCH (n)-[r]-(m)
        WHERE 1=1 {exclusion_clause.replace('AND', '') if excluded_lower else ''}
        WITH DISTINCT n, r, m
        RETURN {{ id: n.id, entity_role: n.entity_role, type: labels(n)[0] }} AS source,
               type(r) AS relationship,
               {{ id: m.id, entity_role: m.entity_role, type: labels(m)[0] }} AS target
        LIMIT $limit
        """
        
        params = {
            "entities": target_entities,
            "excluded": excluded_lower,
            "limit": graph_limit
        }

        return self._query_graph(cypher, params)

    def _query_graph(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.graph_db.query(cypher, params=params or {})


class QdrantChunkRetrieval(BaseRetrieval):
    """Chunk retrieval using Qdrant Hybrid Search (Dense + Sparse).
    
    Args:
        graph_db: Neo4jGraph instance (used for loading chunks if needed)
        vector_store: QdrantVectorStore instance
        collection_name: Qdrant collection name
        auto_build: Whether to auto-create collection on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        vector_store: Optional[QdrantVectorStore] = None,
        collection_name: Optional[str] = None,
        auto_build: bool = True,
    ):
        super().__init__(graph_db)
        self.vector_store = vector_store or create_qdrant_store()
        self.collection_name = collection_name or qdrant_config.collection_name
        
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
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold
            
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
            top_k=top_k,
            threshold=threshold if threshold > 0 else None
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

    def semantic_search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """Retrieve relevant chunks using dense vector search only (no sparse/keyword).
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of chunk dicts with scores
        """
        # Generate dense embedding only
        query_vector = self.dense_encoder.encode(query)
        
        # Dense-only search
        results = self.vector_store.search(
            collection=self.collection_name,
            vector=query_vector,
            top_k=top_k,
            threshold=threshold if threshold > 0 else None
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
        auto_build: Whether to auto-create Qdrant collection on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        vector_store: Optional[QdrantVectorStore] = None,
        auto_build: bool = True,
    ):
        self.graph_retrieval = GraphRetrieval(graph_db=graph_db)
        self.chunk_retrieval = QdrantChunkRetrieval(
            graph_db=graph_db,
            vector_store=vector_store,
            auto_build=auto_build
        )

    def retrieve(
        self, 
        query: str, 
        target_entities: List[str], 
        excluded_entities: Optional[List[str]] = None,
        graph_limit: int = 10,
        chunk_top_k: int = 5,
        chunk_threshold: float = 0.0
    ) -> Dict[str, List[Dict]]:
        """Hybrid retrieval combining graph triples and semantic chunks.
        
        Args:
            query: Search query text for semantic chunk retrieval
            target_entities: List of entity names for graph retrieval
            excluded_entities: List of entity names to exclude from graph results
            graph_limit: Optional override for graph limit
            chunk_top_k: Optional override for chunk top_k
            chunk_threshold: Optional override for chunk threshold
            
        Returns:
            Dict with 'graph' and 'chunk' keys
        """
        graph_results = self.graph_retrieval.retrieve(target_entities, excluded_entities, graph_limit=graph_limit)
        chunk_results = self.chunk_retrieval.retrieve(query, top_k=chunk_top_k, threshold=chunk_threshold)
        
        return {
            "graph": graph_results,
            "chunk": chunk_results,
        }
