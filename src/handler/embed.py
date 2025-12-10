import uuid
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer


class EmbedStorage:
    def __init__(self, model_name: str, graph_db: Neo4jGraph, label: str = "Chunk"):
        self.graph_db = graph_db
        self.embedder = SentenceTransformer(model_name)
        self.label = label # Label node cho các chunk embeddings

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def _format_chunk(self, chunk: Union[Dict[str, Any], Any], doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
        content = getattr(chunk, "content", None) or chunk.get("content", "")
        if not content or not str(content).strip():
            return None

        metadata = getattr(chunk, "metadata", None) or chunk.get("metadata", {}) or {}
        chunk_type = getattr(chunk, "chunk_type", None) or chunk.get("chunk_type", "")
        chunk_type_val = chunk_type.value if hasattr(chunk_type, "value") else str(chunk_type)

        chunk_id = metadata.get("chunk_id") or metadata.get("id") or str(uuid.uuid4())

        try:
            embedding = self._get_embedding(content)
        except Exception as e:
            logger.error(f"Embedding failed for chunk_id={chunk_id}: {e}")
            return None

        return {
            "id": chunk_id,
            "text": content,
            "doc_id": doc_id or metadata.get("doc_id"),
            "chunk_type": chunk_type_val,
            "embedding": embedding,
        }

    def store_embeddings(self, chunks: List[Union[Dict[str, Any], Any]], doc_id: Optional[str] = None) -> int:
        """
        Embed and store chunks into the graph database.
        Args:
            chunks (List[Union[Dict[str, Any], Any]]): List of chunks to embed and store.
            doc_id (Optional[str]): Optional document ID to associate with each chunk.
        Returns:
            int: Number of chunks successfully stored.
        """
        
        rows: List[Dict[str, Any]] = []
        for chunk in chunks:
            formatted = self._format_chunk(chunk, doc_id)
            if formatted:
                rows.append(formatted)

        if not rows:
            logger.info("No valid chunks to embed/store.")
            return 0

        query = f"""
        UNWIND $rows as row
        MERGE (c:{self.label} {{id: row.id}})
        ON CREATE SET c.text = row.text,
                      c.embedding = row.embedding,
                      c.doc_id = row.doc_id,
                      c.chunk_type = row.chunk_type
        ON MATCH SET c.text = row.text,
                     c.embedding = row.embedding,
                     c.doc_id = row.doc_id,
                     c.chunk_type = row.chunk_type
        RETURN count(c) as count
        """

        try:
            result = self.graph_db.query(query, params={"rows": rows})
            stored = result[0]["count"] if result else 0
            logger.info(f"Stored/updated {stored} chunk embeddings")
            return stored
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return 0

    def ensure_vector_index(self, dimension: int, similarity: str = "cosine") -> None:
        index_query = f"""
        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
        FOR (c:{self.label}) ON (c.embedding)
        OPTIONS {{dimension: $dim, similarityFunction: $sim}}
        """
        try:
            self.graph_db.query(index_query, params={"dim": dimension, "sim": similarity})
            logger.info("Vector index ensured for chunk embeddings")
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")