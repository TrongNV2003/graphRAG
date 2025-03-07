from loguru import logger
from langchain_neo4j import Neo4jGraph
from typing import List, Dict

from graphRAG.config.setting import neo4j_config

class GraphRetriever:
    def __init__(self, url: str, username: str, password: str):
        self.graph = Neo4jGraph(url=url, username=username, password=password)

    def _query_graph(self, cypher: str, params: Dict = None, limit: int = 25) -> List[Dict]:
        """Truy vấn Neo4j và trả về danh sách kết quả."""
        full_cypher = f"{cypher} LIMIT {limit}"
        results = self.graph.query(full_cypher, params=params or {})
        return results

    def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        """
            Lấy các cặp node từ node source có node id là query.
               (node n)     relationship     (node m)
            (node source) ---------------> (node target)
        """
        cypher_query = """
            MATCH (n {id: $node_id})-[r]->(m)
            RETURN n.id AS source, type(r) AS relationship, m.id AS target
            UNION
            MATCH (n)-[r]->(m {id: $node_id})
            RETURN n.id AS source, type(r) AS relationship, m.id AS target
        """

        params = {"node_id": query}
        results = self._query_graph(cypher_query, params, limit=limit)

        return results

    
# if __name__ == "__main__":
#     visualizer = GraphRetriever(
#         url=neo4j_config.url,
#         username=neo4j_config.username,
#         password=neo4j_config.password
#     )
#     query = "Elizabeth I"
#     results = visualizer.retrieve(query, limit=20)

#     logger.info(f"Retrieved {len(results)} relationships for '{query}':")
#     for result in results:
#         print(f"{result['source']} -[{result['relationship']}]-> {result['target']}")