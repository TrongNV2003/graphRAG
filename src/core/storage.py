import uuid
from loguru import logger
from collections import defaultdict
from langchain_neo4j import Neo4jGraph
from typing import Any, Dict, List, Optional, Union

from src.services.dense_encoder import get_dense_encoder


class GraphStorage:
    """
    GraphStorage handles storing nodes and relationships into a Neo4j graph database.
    It ensures necessary constraints are set up and provides methods to store graph data.
    """
    def __init__(self, graph_db: Neo4jGraph):
        self.graph_db = graph_db
        self.__setup_schema()

    def __setup_schema(self):
        """Make sure necessary constraints and indexes exist in the database."""
        logger.info("Setting up database constraints")
        
        constraint_query = """
        CREATE CONSTRAINT entity_id_uniqueness IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """
        
        try:
            self.graph_db.query(constraint_query)
        except Exception as e:
            logger.error(f"Failed to create constraint: {e}")

        # Create fulltext index for fuzzy search on Entity names
        index_query = """
        CREATE FULLTEXT INDEX entity_fulltext_index IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.id]
        """
        try:
            self.graph_db.query(index_query)
            logger.info("Fulltext index 'entity_fulltext_index' ensured")
        except Exception as e:
            logger.error(f"Failed to create fulltext index: {e}")

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
                node_result = self.graph_db.query(node_query, params={"nodes": nodes_to_store})
                logger.info(f"Upserted {node_result[0]['node_count']} nodes.")
        except Exception as e:
            logger.error(f"Error storing nodes: {str(e)}")


        # Add relationships
        rels_to_store = []
        for rel in graph_data.get("relationships", []):
            # Validate relationship structure
            if not all(key in rel for key in ["source", "target", "relationship_type"]):
                continue

            # Skip self-referential relationships
            if rel["source"] == rel["target"]:
                continue
            
            source = rel["source"].strip()
            target = rel["target"].strip()
            relationship_type = rel["relationship_type"]
            
            # Validate source, target, and relationship_type
            if source == "" or target == "":
                continue
            
            # Skip invalid relationship types
            if not relationship_type or relationship_type.strip() == "" or relationship_type.strip() == "-":
                continue
            
            normalized_rel_type = self._normalize_label(relationship_type)
            
            # Skip invalid normalized relationship types
            if not normalized_rel_type or normalized_rel_type == "_":
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
                relationship_result = self.graph_db.query(rel_query, params={"rels_data": rels_data})
                logger.info(f"Upserted {relationship_result[0]['rel_count']} relationships.")
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


class QdrantEmbedStorage:
    """
    QdrantEmbedStorage handles embedding and storing chunks into Qdrant vector database.
    Supports both dense and sparse vectors for hybrid search.
    """
    def __init__(
        self,
        collection_name: Optional[str] = None,
        auto_create: bool = True
    ):
        from src.engines.qdrant import create_qdrant_store
        from src.services.sparse_encoder import get_sparse_encoder
        from src.config.setting import qdrant_config
        
        self.vector_store = create_qdrant_store()
        self.collection_name = collection_name or qdrant_config.collection_name
        self.embedder = get_dense_encoder()
        self._sparse_encoder = None
        
        if auto_create:
            self._ensure_collection()
    
    @property
    def sparse_encoder(self):
        if self._sparse_encoder is None:
            from src.services.sparse_encoder import get_sparse_encoder
            self._sparse_encoder = get_sparse_encoder()
        return self._sparse_encoder
    
    def _ensure_collection(self):
        """Ensure collection exists, create if not"""
        if not self.vector_store.collection_exists(self.collection_name):
            dimension = self.embedder.get_dimension()
            self.vector_store.create_collection(
                name=self.collection_name,
                dimension=dimension,
                enable_sparse=True
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def store_embeddings(self, chunks: List[Union[Dict[str, Any], Any]], doc_id: Optional[str] = None) -> int:
        """
        Embed and store chunks into Qdrant vector database.
        
        Args:
            chunks: List of chunks to embed and store.
            doc_id: Optional document ID to associate with each chunk.
            
        Returns:
            int: Number of chunks successfully stored.
        """
        from src.models import VectorPoint
        
        points: List[VectorPoint] = []
        
        for chunk in chunks:
            formatted = self._format_chunk(chunk, doc_id)
            if formatted:
                points.append(formatted)
        
        if not points:
            logger.info("No valid chunks to embed/store in Qdrant.")
            return 0
        
        try:
            self.vector_store.upsert(self.collection_name, points)
            logger.info(f"Stored {len(points)} chunk embeddings to Qdrant")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to store embeddings to Qdrant: {e}")
            return 0
    
    def _format_chunk(self, chunk: Union[Dict[str, Any], Any], doc_id: Optional[str]) -> Optional['VectorPoint']:
        """Format chunk into VectorPoint for Qdrant storage"""
        from src.models import VectorPoint
        
        content = getattr(chunk, "content", None) or chunk.get("content", "")
        if not content or not str(content).strip():
            return None
        
        metadata = getattr(chunk, "metadata", None) or chunk.get("metadata", {}) or {}
        chunk_type = getattr(chunk, "chunk_type", None) or chunk.get("chunk_type", "")
        chunk_type_val = chunk_type.value if hasattr(chunk_type, "value") else str(chunk_type)
        
        chunk_id = metadata.get("chunk_id") or metadata.get("id") or str(uuid.uuid4())
        
        try:
            # Generate dense embedding
            dense_embedding = self.embedder.encode(content)
            
            # Generate sparse embedding
            sparse_result = self.sparse_encoder.encode(content)
            
            return VectorPoint(
                id=chunk_id,
                vector=dense_embedding,
                payload={
                    "chunk_id": chunk_id,
                    "content": content,
                    "doc_id": doc_id or metadata.get("doc_id"),
                    "chunk_type": chunk_type_val,
                },
                sparse_indices=sparse_result.indices,
                sparse_values=sparse_result.values
            )
        except Exception as e:
            logger.error(f"Embedding failed for chunk_id={chunk_id}: {e}")
            return None
    
    def clear_collection(self) -> bool:
        """Clear all data from collection"""
        return self.vector_store.delete_collection(self.collection_name)