from langchain_neo4j import Neo4jGraph

class GraphStorage:
    def __init__(self, url: str, username: str, password: str):
        self.graph = Neo4jGraph(url=url, username=username, password=password)

    def _standard_label(self, label: str) -> str:
        return label.replace(" ", "_").replace("-", "_").upper()
    
    def _stardard_property(self, value: str) -> str:
        """Chuẩn hóa dấu nháy đơn"""
        return value.replace("'", "\\'")

    def store(self, graph_data, clear_old_graph=False):
        # Xóa đồ thị cũ
        if clear_old_graph:
            self.graph.query("MATCH (n) DETACH DELETE n")

        # Add nodes
        for node in graph_data["nodes"]:
            sanitized_type = self._standard_label(node["type"])
            sanitized_id = self._stardard_property(node["id"])
            
            query = f"""
            MERGE (n:{sanitized_type} {{id: $id}})
            SET n.name = $id
            """

            self.graph.query(query, params={"id": sanitized_id})

        # Add relationships
        for rel in graph_data["relationships"]:
            sanitized_rel_type = self._standard_label(rel["type"])
            sanitized_source = self._stardard_property(rel["source"])
            sanitized_target = self._stardard_property(rel["target"])
            
            query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:`{sanitized_rel_type}`]->(target)
            """
            
            self.graph.query(query, params={
                "source_id": sanitized_source,
                "target_id": sanitized_target
            })