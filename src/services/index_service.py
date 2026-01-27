from openai import OpenAI
from loguru import logger
from typing import List, Optional, Dict
from langchain_neo4j import Neo4jGraph

from src.config.dataclass import StructuralChunk
from src.engines.llm import EntityExtractionLLM
from src.processing.chunking import TwoPhaseDocumentChunker
from src.core.storage import GraphStorage, QdrantEmbedStorage
from src.processing.postprocessing import EntityPostprocessor


class GraphIndexing:
    def __init__(
        self,
        client: OpenAI,
        graph_db: Neo4jGraph,
        chunker: TwoPhaseDocumentChunker,
        extractor: EntityExtractionLLM,
        postprocessor: EntityPostprocessor,
        storage: GraphStorage,
        qdrant_storage: QdrantEmbedStorage,
        clear_old_graph: bool = False
    ):
        self.client = client
        self.graph_db = graph_db
        self.chunker = chunker
        self.storage = storage
        self.extractor = extractor
        self.postprocessor = postprocessor
        self.qdrant_storage = qdrant_storage
        self.clear_old_graph = clear_old_graph

    def chunking(self, document: dict, max_new_chunk_size: Optional[int] = None) -> List['StructuralChunk']:
        chunks = self.chunker.chunk_document(document, max_new_chunk_size=max_new_chunk_size)
        return chunks

    def indexing(self, chunks: List['StructuralChunk']) -> None:
        """Index pre-chunked data into the graph database."""
        batch_size = 5
        all_nodes: List[Dict] = []
        all_relationships: List[Dict] = []
        batch_nodes: List[Dict] = []
        batch_relationships: List[Dict] = []
        batch_chunks: List['StructuralChunk'] = []
        total_chunks = len(chunks)
        
        if self.clear_old_graph:
            logger.info("Clearing existing graph data")
            self.storage.clear_all()

        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "content", None) or chunk.get("content", "")
            if not text:
                continue

            logger.info(f"Processing chunk {i + 1}/{total_chunks}")
            
            try:
                extracted_data = self.extractor.call(text=text)
            except Exception:
                logger.warning(f"API call failed for chunk {i + 1}")
                continue
            
            if not extracted_data or "nodes" not in extracted_data or "relationships" not in extracted_data:
                continue
            
            cleaned_nodes: List[Dict] = []
            for node in extracted_data.get("nodes", []):
                cleaned_nodes.append({
                    "id": self.postprocessor(node.get("id", "")),
                    "entity_type": self.postprocessor(node.get("entity_type", "")),
                    "entity_role": self.postprocessor(node.get("entity_role", "")),
                })
            
            cleaned_relationships: List[Dict] = []
            for rel in extracted_data.get("relationships", []):
                cleaned_relationships.append({
                    "source": self.postprocessor(rel.get("source", "")),
                    "target": self.postprocessor(rel.get("target", "")),
                    "relationship_type": self.postprocessor(rel.get("relationship_type", "")),
                })

            batch_nodes.extend(cleaned_nodes)
            batch_relationships.extend(cleaned_relationships)
            
            # Prepare chunks for embedding storage
            batch_chunks.append(chunk)
            
            logger.info(f"Extracted chunk {i + 1}: {len(cleaned_nodes)} entities, {len(cleaned_relationships)} relationships")

            if (i + 1) % batch_size == 0 or (i + 1) == total_chunks:
                dedup_nodes = self._deduplicate_entities(batch_nodes)
                dedup_relationships = self._deduplicate_relationships(batch_relationships)
                
                batch_graph_data = {
                    "nodes": dedup_nodes,
                    "relationships": dedup_relationships
                }
                self.storage.store_graph(batch_graph_data)
                
                all_nodes.extend(dedup_nodes)
                all_relationships.extend(dedup_relationships)

                # Store embeddings to Qdrant for hybrid search
                if batch_chunks:
                    self.qdrant_storage.store_embeddings(batch_chunks)
                
                # Reset batch
                batch_nodes = []
                batch_relationships = []
                batch_chunks = []
                

        entity_count = self.graph_db.query('MATCH (e:Entity) RETURN count(e) as count')[0]['count']
        rel_count = self.graph_db.query('MATCH ()-[r]->() RETURN count(r) as count')[0]['count']
        
        try:
            collection_info = self.qdrant_storage.vector_store.get_collection_info(
                self.qdrant_storage.collection_name
            )
            chunk_count = collection_info.get("points_count", 0) if collection_info else 0
        except Exception as e:
            logger.warning(f"Failed to get chunk count from Qdrant: {e}")
            chunk_count = 0
        
        logger.info(f"Database Statistics - Total Entities: {entity_count}, Total Relationships: {rel_count}, Total Chunks: {chunk_count}")
        
        return all_nodes, all_relationships
            
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on (id, entity_type, entity_role)."""
        seen = set()
        deduplicated: List[Dict] = []
        
        for entity in entities:
            entity_id = entity.get('id', '').strip()
            entity_type = entity.get('entity_type', '').strip()
            entity_role = entity.get('entity_role', '').strip()
            
            if not entity_id:
                continue
            
            key = (entity_id, entity_type, entity_role)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        Remove duplicate relationships based on source, target, and relationship_type.
        If duplicates exist, keep the first occurrence.
        """
        seen = set()
        deduplicated = []
        
        for relationship in relationships:
            source = relationship.get('source', '')
            target = relationship.get('target', '')
            rel_type = relationship.get('relationship_type', '')
            
            key = (source, target, rel_type)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(relationship)
            else:
                logger.debug(f"Duplicate relationship found and removed: source='{source}', target='{target}', type='{rel_type}'")
        
        return deduplicated
