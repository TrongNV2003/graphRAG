from loguru import logger
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer

from src.config.setting import neo4j_config, embed_config

class GraphStorage:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )
        self.embedder = SentenceTransformer(embed_config.embedder_model)

    def _standard_label(self, label: str) -> str:
        return label.replace(" ", "_").replace("-", "_").upper()
    
    def _standard_property(self, value: str) -> str:
        """Chuẩn hóa dấu nháy đơn"""
        return value.replace("'", "\\'")

    def _get_embedding(self, node: dict) -> list:
        text = f"{node['id']} {node.get('role', '')}".strip()
        return self.embedder.encode(text).tolist()

    def store(self, graph_data, clear_old_graph=False):
        if clear_old_graph:
            self.graph.query("MATCH (n) DETACH DELETE n")

        # Add nodes
        for node in graph_data["nodes"]:
            sanitized_type = self._standard_label(node["type"])
            sanitized_id = self._standard_property(node["id"])
            sanitized_role = self._standard_property(node.get("role", ""))
            embedding = self._get_embedding(node)
            
            query = f"""
            MERGE (n:{sanitized_type} {{id: $id}})
            SET n.name = $id, n.role = $role, n.embedding = $embedding
            """

            try:
                self.graph.query(query, params={
                    "id": sanitized_id,
                    "role": sanitized_role,
                    "embedding": embedding
                })
            except Exception as e:
                logger.error(f"Error storing node {sanitized_id}: {str(e)}")

        # Add relationships
        valid_relationships = 0
        for rel in graph_data["relationships"]:
            if not all(key in rel for key in ["source", "target", "type"]):
                logger.warning(f"Skipping invalid relationship with missing keys: {rel}")
                continue
            
            sanitized_rel_type = self._standard_label(rel["type"])
            sanitized_source = self._standard_property(rel["source"])
            sanitized_target = self._standard_property(rel["target"])
            
            query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:`{sanitized_rel_type}`]->(target)
            """
            
            try:
                self.graph.query(query, params={
                    "source_id": sanitized_source,
                    "target_id": sanitized_target
                })
                valid_relationships += 1
            except Exception as e:
                logger.error(f"Failed to store relationship {sanitized_source} -> {sanitized_target}: {str(e)}")

        logger.info(f"Stored {len(graph_data['nodes'])} nodes and {valid_relationships} relationships")