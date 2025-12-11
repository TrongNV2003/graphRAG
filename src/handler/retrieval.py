import os
from pathlib import Path
from loguru import logger
from typing import List, Dict, Optional
from langchain_neo4j import Neo4jGraph

from src.engines.search import HybridRetrievalEngine


class GraphRetrieval:
    """
    Graph retrieval with nodes + relationships.
    """

    def __init__(self, graph_db: Neo4jGraph):
        self.graph_db = graph_db

    def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        # Exact match on id/role
        cypher_exact = """
        MATCH (n)
        WHERE toLower(n.id) = toLower($q) OR toLower(n.role) = toLower($q)
        MATCH (n)-[r]-(m)
        WITH DISTINCT n, r, m
        RETURN { id: n.id, role: n.role, type: labels(n)[0] } AS source,
               type(r) AS relationship,
               { id: m.id, role: m.role, type: labels(m)[0] } AS target
        LIMIT $limit
        """

        params = {"q": query, "limit": limit}
        results = self._query_graph(cypher_exact, params)

        if results:
            return results

        # Fallback: substring match on id/role
        cypher_contains = """
        MATCH (n)
        WHERE toLower(n.id) CONTAINS toLower($q) OR toLower(n.role) CONTAINS toLower($q)
        MATCH (n)-[r]-(m)
        WITH DISTINCT n, r, m
        RETURN { id: n.id, role: n.role, type: labels(n)[0] } AS source,
               type(r) AS relationship,
               { id: m.id, role: m.role, type: labels(m)[0] } AS target
        LIMIT $limit
        """

        return self._query_graph(cypher_contains, params)

    def _query_graph(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.graph_db.query(cypher, params=params or {})


class EmbeddingRetrieval:
    """Chunk retrieval with Hybrid Search Engine (BM25 + dense).
    Args:
        graph_db: Neo4jGraph instance
        search_engine: Hybrid search engine
        auto_build: Whether to auto-load or build index on init
    """
    def __init__(
        self,
        graph_db: Neo4jGraph,
        search_engine: HybridRetrievalEngine,
        auto_build: bool = True,
    ):
        self.graph_db = graph_db
        self.search_engine = search_engine
        self._index_built = False
        self.index_path = "data/hybrid_index"
        
        # Auto-load or build index
        if auto_build:
            if Path(self.index_path).exists():
                self._load_index()
            else:
                self._build_index()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0,
    ) -> List[Dict]:
        """Retrieve relevant chunks with similarity threshold.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold (default: 0.5)
            
        Returns:
            List of chunk dicts with scores >= threshold
        """
        doc_ids, scores = self.search_engine.search(query, top_k=top_k, min_score=threshold)
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
        return results[:top_k]

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
                
                # Skip duplicates and empty content
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
        auto_build: Whether to auto-load or build index on init
    """

    def __init__(
        self,
        graph_db: Neo4jGraph,
        search_engine: HybridRetrievalEngine,
        auto_build: bool = True,
    ):
        self.graph_retrieval = GraphRetrieval(graph_db)
        self.embedding_retrieval = EmbeddingRetrieval(
            graph_db, 
            search_engine=search_engine,
            auto_build=auto_build
        )

    def retrieve(
        self,
        query: str,
        graph_limit: int = 10,
        chunk_top_k: int = 10,
        chunk_threshold: float = 0.5,
    ) -> Dict[str, List[Dict]]:
        """Hybrid retrieval combining graph triples and semantic chunks.
        
        Args:
            query: Search query text
            graph_limit: Max number of graph triples to return
            chunk_top_k: Max number of chunks to return
            chunk_threshold: Minimum similarity score for chunks (default: 0.5)
            
        Returns:
            Dict with 'graph' and 'chunks' keys
        """
        graph_results = self.graph_retrieval.retrieve(query, limit=graph_limit)
        chunk_results = self.embedding_retrieval.retrieve(
            query, 
            top_k=chunk_top_k, 
            threshold=chunk_threshold
        )
        return {
            "graph": graph_results,
            "chunks": chunk_results,
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
    
    retrieval = HybridRetrieval(graph_db, search_engine=search_engine, auto_build=True)

    query = "Elizabeth"
    results = retrieval.retrieve(
        query, 
        graph_limit=10, 
        chunk_top_k=10,
        chunk_threshold=0
    )

    logger.info(f"Graph results: {len(results['graph'])}")
    logger.info(f"Chunk results: {len(results['chunks'])}")
    print("Graph triples:", results["graph"])
    print("\nChunks:")
    for i, chunk in enumerate(results["chunks"], 1):
        print(f"  {i}. Score: {chunk['score']:.3f} | {chunk['chunk_text'][:100]}...")