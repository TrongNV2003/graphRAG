import uuid
from loguru import logger
from collections import defaultdict
from langchain_neo4j import Neo4jGraph
from typing import Any, Dict, List, Optional, Union

from src.config.dataclass import VectorPoint
from src.engines.qdrant import QdrantVectorStore
from src.services.dense_encoder import get_dense_encoder

class GraphStorage:
    """
    GraphStorage handles storing nodes and relationships into a Neo4j graph database.
    It ensures necessary constraints are set up and provides methods to store graph data.
    """
    def __init__(self, graph_db: Neo4jGraph):
        self.graph_db = graph_db

    def setup_schema(self):
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
            
            # Skip nodes with empty label
            if not normalized_type or normalized_type.strip("_") == "":
                continue

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

        total_upserted = 0
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
                total_upserted += relationship_result[0]['rel_count']
            except Exception as e:
                logger.error(f"Failed to store relationships of type '{rel_type}': {str(e)}")
        
        if total_upserted > 0:
            logger.info(f"Upserted {total_upserted} relationships.")

    def _normalize_label(self, label: str) -> str:
        """Normalize label for Neo4j"""
        return label.replace(" ", "_").replace("-", "_").upper()

    def clear_all(self):
        """Clear all graph data."""
        try:
            self.graph_db.query("MATCH (n) DETACH DELETE n")
        except Exception as e:
            logger.error(f"Error clearing graph data: {e}")


    def backup_graph(self, output_path: str, batch_size: int = 5000) -> dict:
        """
        Backup the entire graph to a zip file containing CSVs.
        
        Args:
            output_path: Path for the output zip file.
            batch_size: Number of records to fetch per batch.
            
        Returns:
            dict: Metadata about the backup (counts, timestamp).
        """
        import csv
        import json
        import zipfile
        from datetime import datetime
        from io import StringIO
        
        node_count = 0
        rel_count = 0
        timestamp = datetime.now().isoformat()
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Export Nodes
            nodes_buffer = StringIO()
            writer = csv.writer(nodes_buffer)
            writer.writerow(['id', 'labels', 'entity_type', 'entity_role'])
            
            offset = 0
            while True:
                nodes = self._get_nodes_batch(offset, batch_size)
                if not nodes:
                    break
                for node in nodes:
                    writer.writerow([
                        node.get('id', ''),
                        node.get('labels', ''),
                        node.get('entity_type', ''),
                        node.get('entity_role', ''),
                    ])
                    node_count += 1
                offset += batch_size
            
            zf.writestr('nodes.csv', nodes_buffer.getvalue())
            nodes_buffer.close()
            
            # Export Relationships
            rels_buffer = StringIO()
            writer = csv.writer(rels_buffer)
            writer.writerow(['source', 'target', 'type'])
            
            offset = 0
            while True:
                rels = self._get_rels_batch(offset, batch_size)
                if not rels:
                    break
                for rel in rels:
                    writer.writerow([
                        rel.get('source', ''),
                        rel.get('target', ''),
                        rel.get('type', ''),
                    ])
                    rel_count += 1
                offset += batch_size
            
            zf.writestr('relationships.csv', rels_buffer.getvalue())
            rels_buffer.close()
            
            # Write Metadata
            metadata = {
                "backup_timestamp": timestamp,
                "node_count": node_count,
                "relationship_count": rel_count,
                "schema_version": "1.0"
            }
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        logger.info(f"Backup completed: {node_count} nodes, {rel_count} relationships -> {output_path}")
        return metadata

    def _get_nodes_batch(self, skip: int, limit: int) -> List[Dict]:
        """Get a batch of nodes for backup."""
        query = """
        MATCH (n:Entity)
        RETURN n.id AS id, 
               reduce(s = '', l IN labels(n) | s + CASE WHEN s = '' THEN l ELSE ';' + l END) AS labels,
               n.entity_type AS entity_type,
               n.entity_role AS entity_role
        ORDER BY n.id
        SKIP $skip LIMIT $limit
        """
        try:
            return self.graph_db.query(query, params={"skip": skip, "limit": limit})
        except Exception as e:
            logger.error(f"Error fetching nodes batch: {e}")
            return []

    def _get_rels_batch(self, skip: int, limit: int) -> List[Dict]:
        """Get a batch of relationships for backup."""
        query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        RETURN s.id AS source, t.id AS target, type(r) AS type
        ORDER BY s.id, t.id
        SKIP $skip LIMIT $limit
        """
        try:
            return self.graph_db.query(query, params={"skip": skip, "limit": limit})
        except Exception as e:
            logger.error(f"Error fetching relationships batch: {e}")
            return []

    def restore_graph(self, zip_path: str, clear_existing: bool = True, batch_size: int = 5000) -> dict:
        """
        Restore graph from a backup zip file.
        
        Args:
            zip_path: Path to the backup zip file.
            clear_existing: Whether to clear existing data before restore.
            batch_size: Number of records to upsert per batch.
            
        Returns:
            dict: Restore statistics.
        """
        import csv
        import json
        import zipfile
        from io import TextIOWrapper
        
        if clear_existing:
            logger.info("Clearing existing graph data before restore...")
            self.clear_all()
        
        nodes_restored = 0
        rels_restored = 0
        errors = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Restore Nodes
            with zf.open('nodes.csv') as f:
                reader = csv.DictReader(TextIOWrapper(f, 'utf-8'))
                batch = []
                for row in reader:
                    batch.append(row)
                    if len(batch) >= batch_size:
                        count, err = self._upsert_nodes_batch(batch)
                        nodes_restored += count
                        if err:
                            errors.append(err)
                        batch = []
                if batch:
                    count, err = self._upsert_nodes_batch(batch)
                    nodes_restored += count
                    if err:
                        errors.append(err)
            
            # Restore Relationships
            with zf.open('relationships.csv') as f:
                reader = csv.DictReader(TextIOWrapper(f, 'utf-8'))
                batch = []
                for row in reader:
                    batch.append(row)
                    if len(batch) >= batch_size:
                        count, err = self._upsert_rels_batch(batch)
                        rels_restored += count
                        if err:
                            errors.append(err)
                        batch = []
                if batch:
                    count, err = self._upsert_rels_batch(batch)
                    rels_restored += count
                    if err:
                        errors.append(err)
            
            # Read Metadata
            try:
                with zf.open('metadata.json') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        
        result = {
            "nodes_restored": nodes_restored,
            "relationships_restored": rels_restored,
            "errors": errors,
            "original_metadata": metadata
        }
        
        logger.info(f"Restore completed: {nodes_restored} nodes, {rels_restored} relationships")
        return result

    def _upsert_nodes_batch(self, rows: List[Dict]) -> tuple:
        """Upsert a batch of nodes from CSV rows."""
        nodes_to_store = []
        for row in rows:
            id_val = row.get('id', '').strip()
            if not id_val:
                continue
            
            labels_str = row.get('labels', 'Entity')
            labels_list = [l.strip() for l in labels_str.split(';') if l.strip()]
            # Use second label as the type label (first is always 'Entity')
            type_label = labels_list[1] if len(labels_list) > 1 else 'UNKNOWN'
            
            nodes_to_store.append({
                "label": type_label,
                "properties": {
                    "id": id_val,
                    "entity_type": row.get('entity_type', ''),
                    "entity_role": row.get('entity_role', ''),
                }
            })
        
        if not nodes_to_store:
            return 0, None
        
        query = """
        UNWIND $nodes as node
        MERGE (e:Entity {id: node.properties.id})
        ON CREATE SET e = node.properties
        ON MATCH SET e += node.properties
        WITH e, node
        CALL apoc.create.addLabels(e, [node.label]) YIELD node as ignored
        RETURN count(e) as node_count
        """
        try:
            result = self.graph_db.query(query, params={"nodes": nodes_to_store})
            return result[0]['node_count'], None
        except Exception as e:
            logger.error(f"Error restoring nodes batch: {e}")
            return 0, str(e)

    def _upsert_rels_batch(self, rows: List[Dict]) -> tuple:
        """Upsert a batch of relationships from CSV rows."""
        rels_by_type = defaultdict(list)
        for row in rows:
            source = row.get('source', '').strip()
            target = row.get('target', '').strip()
            rel_type = row.get('type', '').strip()
            if source and target and rel_type:
                rels_by_type[rel_type].append({"source": source, "target": target})
        
        total = 0
        for rel_type, rels_data in rels_by_type.items():
            query = f"""
            UNWIND $rels_data as rel
            MATCH (source:Entity {{id: rel.source}})
            MATCH (target:Entity {{id: rel.target}})
            MERGE (source)-[r:`{rel_type}`]->(target)
            RETURN count(r) as rel_count
            """
            try:
                result = self.graph_db.query(query, params={"rels_data": rels_data})
                total += result[0]['rel_count']
            except Exception as e:
                logger.error(f"Error restoring relationships of type '{rel_type}': {e}")
                return total, str(e)
        
        return total, None


class QdrantEmbedStorage:
    """
    QdrantEmbedStorage handles embedding and storing chunks into Qdrant vector database.
    Supports both dense and sparse vectors for hybrid search.
    """
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        collection_name: Optional[str] = None,
        auto_create: bool = True
    ):
        from src.engines.qdrant import create_qdrant_store
        from src.config.setting import qdrant_config
        
        self.vector_store = vector_store or create_qdrant_store()
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