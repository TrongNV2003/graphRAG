from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod
from langchain_neo4j import Neo4jGraph
from typing import List, Dict, Optional, Any

from src.engines.search import HybridRetrievalEngine


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


class EmbeddingRetrieval(BaseRetrieval):
    """Chunk retrieval with Hybrid Search Engine (BM25 + dense).
    Args:
        graph_db: Neo4jGraph instance
        search_engine: Hybrid search engine
        top_k: Maximum number of results to return
        threshold: Minimum similarity score threshold
        auto_build: Whether to auto-load or build index on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        search_engine: HybridRetrievalEngine,
        top_k: int = 5,
        threshold: float = 0,
        auto_build: bool = True,
    ):
        super().__init__(graph_db)
        self.search_engine = search_engine
        self._index_built = False
        self.top_k = top_k
        self.threshold = threshold
        self.index_path = "data/hybrid_index"
        
        # Auto-load or build index
        if auto_build:
            if Path(self.index_path).exists():
                self._load_index()
            else:
                self._build_index()
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks with similarity threshold.
        
        Args:
            query: Search query text
            
        Returns:
            List of chunk dicts with scores >= threshold
        """
        doc_ids, scores = self.search_engine.search(query, top_k=self.top_k, min_score=self.threshold)
        results: List[Dict] = []

        for doc_id, score in zip(doc_ids, scores):
            try:
                doc = self.search_engine.document_store[int(doc_id)]
            except (IndexError, ValueError, TypeError):
                continue
            if not isinstance(doc, dict):
                continue
            meta = doc.get("metadata", {}) or {}
            results.append(
                {
                    "chunk_id": meta.get("id"),
                    "chunk_text": doc.get("content", ""),
                    "score": float(score),
                    "metadata": meta,
                }
            )

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:self.top_k]

    def _build_index(self, save: bool = True, force: bool = False) -> None:
        """Load chunks from Neo4j and build search index.
        
        Args:
            save: Whether to save the index to disk after building
            force: Force rebuild even if index was already built
        """
        if self._index_built and not force:
            logger.info("Index already built, skipping. Use force=True to rebuild.")
            return
            
        try:
            logger.info("Building index from Neo4j chunks...")
            
            cypher = """
            MATCH (c:Chunk)
            RETURN c.id AS id, c.text AS content, c.doc_id AS doc_id, c.chunk_type AS chunk_type
            """
            results = self.graph_db.query(cypher)

            # Deduplicate by chunk ID (keep first occurrence)
            seen_ids = set()
            documents: List[Dict] = []
            for record in results:
                chunk_id = record.get("id")
                content = record.get("content") or ""
                
                if not content or chunk_id in seen_ids:
                    continue
                
                seen_ids.add(chunk_id)
                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "id": chunk_id,
                            "doc_id": record.get("doc_id"),
                            "chunk_type": record.get("chunk_type"),
                        },
                    }
                )

            logger.info(f"Loaded {len(documents)} unique chunks from Neo4j")
            
            if documents:
                self.search_engine.index_documents(documents)
                logger.info(f"Indexed {len(documents)} documents into search engine")
                self._index_built = True
                
                if save:
                    self._save_index()
            else:
                logger.warning("No chunks found in Neo4j to index")
                
        except Exception as e:
            logger.error(f"Error building index: {e}")
            self._index_built = False
    
    def _save_index(self) -> None:
        """Save the search engine index to disk."""
        success = self.search_engine.save(self.index_path)
        if success:
            logger.info(f"Saved search index to {self.index_path}")
        else:
            logger.error(f"Failed to save search index to {self.index_path}")
    
    def _load_index(self) -> None:
        """Load a previously saved search engine index from disk."""
        try:
            logger.info(f"Loading saved index from {self.index_path}")
            loaded_engine = HybridRetrievalEngine.load(self.index_path)
            if loaded_engine:
                self.search_engine = loaded_engine
                self._index_built = True
                logger.info(f"Loaded index with {self.search_engine.doc_count} documents")
            else:
                logger.error(f"Failed to load index from {self.index_path}, will build new index")
                self._build_index()
        except Exception as e:
            logger.error(f"Error loading index: {e}, will build new index")
            self._build_index()


class HybridRetrieval:
    """Hybrid RAG: combine graph triples and chunk embeddings.
    Args:
        graph_db: Neo4jGraph instance
        search_engine: Hybrid search engine
        graph_limit: Maximum number of graph to retrieve
        chunk_top_k: Maximum number of chunk results to return
        chunk_threshold: Minimum similarity score threshold for chunks
        auto_build: Whether to auto-load or build index on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        search_engine: HybridRetrievalEngine,
        graph_limit: int = 10,
        chunk_top_k: int = 10,
        chunk_threshold: float = 0.5,
        auto_build: bool = True,
    ):
        self.graph_retrieval = GraphRetrieval(graph_db = graph_db, graph_limit=graph_limit)
        self.embedding_retrieval = EmbeddingRetrieval(
            graph_db, 
            search_engine=search_engine,
            top_k=chunk_top_k,
            threshold=chunk_threshold,
            auto_build=auto_build
        )

    def retrieve(self, query: str, target_entities: List[str], excluded_entities: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Hybrid retrieval combining graph triples and semantic chunks.
        
        Args:
            target_entities: List of entity names for graph retrieval
            query: Search query text for semantic chunk retrieval
            excluded_entities: List of entity names to exclude from graph results
            
        Returns:
            Dict with 'graph' and 'chunks' keys
        """
        graph_results = self.graph_retrieval.retrieve(target_entities, excluded_entities)
        chunk_results = self.embedding_retrieval.retrieve(query)
        
        return {
            "graph": graph_results,
            "chunk": chunk_results,
        }

if __name__ == "__main__":
    from src.config.setting import neo4j_config, embed_config

    graph_db = Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password,
    )

    search_engine = HybridRetrievalEngine(dense_params={
        "model_name": embed_config.embedder_model,
    })
    
    retrieval = HybridRetrieval(
        graph_db,
        search_engine=search_engine,
        graph_limit=10,
        chunk_top_k=10,
        chunk_threshold=0,
        auto_build=True
    )
    target_entities = ["Elizabeth"]
    query = "Tell me about Elizabeth and her relationships."
    results = retrieval.retrieve(query=query, target_entities=target_entities)
    
    logger.info(f"Graph results: {len(results['graph'])}")
    logger.info(f"Chunk results: {len(results['chunks'])}")
    print("Graph:", results["graph"][:3])
    print("\nChunks:")
    for i, chunk in enumerate(results["chunks"][:3], 1):
        print(f"  {i}. Score: {chunk['score']:.3f} | {chunk['chunk_text'][:100]}...")
