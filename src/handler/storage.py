import re
from loguru import logger
from collections import defaultdict
from langchain_neo4j import Neo4jGraph


class GraphStorage:
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

    def store(self, graph_data: dict) -> None:
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

    # def _get_embedding(self, node: dict) -> list:
    #     text = f"{node['id']} {node.get('role', '')}".strip()
    #     return self.embedder.encode(text).tolist()
