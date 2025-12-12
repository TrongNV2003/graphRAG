import uuid
from loguru import logger
from collections import defaultdict
from langchain_neo4j import Neo4jGraph
from typing import Any, Dict, List, Optional, Union
from sentence_transformers import SentenceTransformer

from src.config.setting import embed_config

class GraphStorage:
    """
    GraphStorage handles storing nodes and relationships into a Neo4j graph database.
    It ensures necessary constraints are set up and provides methods to store graph data.
    """
    def __init__(self, graph_db: Neo4jGraph):
        self.graph_db = graph_db
        self.__setup_schema()

    def __setup_schema(self):
        """Đảm bảo các constraints và indexes cần thiết đã tồn tại trong database."""
        logger.info("Setting up database constraints")
        
        constraint_query = """
        CREATE CONSTRAINT entity_id_uniqueness IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """
        
        try:
            self.graph_db.query(constraint_query)
        except Exception as e:
            logger.error(f"Failed to create constraint: {e}")

    def store_graph(self, graph_data: dict) -> None:
        """
        Store nodes and relationships into the graph database.
        Args:
            graph_data (dict): A dictionary containing 'nodes' and 'relationships'.
        """
        
        # Add nodes
        nodes_to_store = []
        for node in graph_data.get("nodes", []):
            id = node.get("id")
            entity_type = node.get("entity_type", "Unknown")
            entity_role = node.get("entity_role", "")
            
            if not id or id.strip() == "":
                logger.warning(f"Skipping node with missing 'id': {node}")
                continue
            
            normalized_type = self._normalize_label(entity_type)

            nodes_to_store.append({
                "label": normalized_type,
                "properties": {
                    "id": id,
                    "entity_type": entity_type,
                    "entity_role": entity_role,
                }
            })

        node_query = """
        UNWIND $nodes as node
        MERGE (e:Entity {id: node.properties.id})
        ON CREATE SET e = node.properties
        ON MATCH SET e += node.properties
        WITH e, node
        CALL apoc.create.addLabels(e, [node.label]) YIELD node as ignored
        RETURN count(e) as node_count
        """

        try:
            if nodes_to_store:
                result = self.graph_db.query(node_query, params={"nodes": nodes_to_store})
                if result:
                    logger.info(f"Stored/updated {len(nodes_to_store)} nodes.")
        except Exception as e:
            logger.error(f"Error storing nodes: {str(e)}")


        # Add relationships
        rels_to_store = []
        for rel in graph_data.get("relationships", []):
            if not all(key in rel for key in ["source", "target", "relationship_type"]):
                logger.warning(f"Skipping invalid relationship with missing keys: {rel}")
                continue
            if rel["source"] == rel["target"]:
                logger.warning(f"Skipping self-referential relationship: {rel}")
                continue
            
            source = rel["source"].strip()
            target = rel["target"].strip()
            relationship_type = rel["relationship_type"]
            
            if source == "" or target == "":
                logger.warning(f"Skipping relationship with empty source/target: {rel}")
                continue
            
            if not relationship_type or relationship_type.strip() == "" or relationship_type.strip() == "-":
                logger.warning(f"Skipping relationship with empty/invalid category: {rel}")
                continue
            
            normalized_rel_type = self._normalize_label(relationship_type)
            
            if not normalized_rel_type or normalized_rel_type == "_":
                logger.warning(f"Skipping relationship with invalid normalized type: '{normalized_rel_type}'")
                continue
            
            rels_to_store.append({
                "source": source,
                "target": target,
                "type": normalized_rel_type
            })
            
        rels_by_type = defaultdict(list)
        for rel in rels_to_store:
            rels_by_type[rel['type']].append(rel)

        for rel_type, rels_data in rels_by_type.items():
            rel_query = f"""
            UNWIND $rels_data as rel
            MATCH (source:Entity {{id: rel.source}})
            MATCH (target:Entity {{id: rel.target}})
            MERGE (source)-[r:`{rel_type}`]->(target)
            RETURN count(r) as rel_count
            """
            try:
                result = self.graph_db.query(rel_query, params={"rels_data": rels_data})
                if result:
                    if result[0]['rel_count'] < len(rels_data):
                        logger.warning(f"Only {result[0]['rel_count']}/{len(rels_data)} relationships created - some nodes may be missing")
                    logger.info(f"Stored/updated {len(rels_data)} relationships of type '{rel_type}.")
            except Exception as e:
                logger.error(f"Failed to store relationships of type '{rel_type}': {str(e)}")

    def _normalize_label(self, label: str) -> str:
        """Normalize label for Neo4j"""
        return label.replace(" ", "_").replace("-", "_").upper()

    def clear_all(self):
        """Clear all graph data."""
        try:
            self.graph_db.query("MATCH (n) DETACH DELETE n")
        except Exception as e:
            logger.error(f"Error clearing graph data: {e}")


class EmbedStorage:
    """
    EmbedStorage handles embedding and storing chunks into a Neo4j graph database.
    """
    def __init__(self, graph_db: Neo4jGraph, label: str = "Chunk"):
        self.graph_db = graph_db
        self.embedder = SentenceTransformer(embed_config.embedder_model)
        self.label = label # Label node cho các chunk embeddings
        self.__setup_schema()
    
    def __setup_schema(self):
        """Đảm bảo constraint uniqueness cho Chunk.id."""
        logger.info(f"Setting up database constraints for {self.label}")
        
        constraint_query = f"""
        CREATE CONSTRAINT chunk_id_uniqueness IF NOT EXISTS
        FOR (c:{self.label}) REQUIRE c.id IS UNIQUE
        """
        
        try:
            self.graph_db.query(constraint_query)
            logger.info(f"Constraint chunk_id_uniqueness ensured for {self.label}")
        except Exception as e:
            logger.error(f"Failed to create {self.label} constraint: {e}")
        self.__setup_schema()

    def __setup_schema(self):
        """Ensure necessary constraints exist for Chunk nodes."""
        logger.info(f"Setting up database constraints for {self.label}")
        
        constraint_query = f"""
        CREATE CONSTRAINT chunk_id_uniqueness IF NOT EXISTS
        FOR (c:{self.label}) REQUIRE c.id IS UNIQUE
        """
        
        try:
            self.graph_db.query(constraint_query)
        except Exception as e:
            logger.error(f"Failed to create {self.label} constraint: {e}")

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