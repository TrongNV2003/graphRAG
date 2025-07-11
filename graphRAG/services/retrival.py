from loguru import logger
from typing import List, Dict
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer

from graphRAG.config.setting import neo4j_config, model_config

class GraphRetriever:
    def __init__(self, url: str, username: str, password: str):
        self.graph = Neo4jGraph(url=url, username=username, password=password)
        self.embedder = SentenceTransformer(model_config.embedder_model)

    def _get_embedding(self, text: str) -> list:
        return self.embedder.encode(text).tolist()

    def _query_graph(self, cypher: str, params: Dict = None) -> List[Dict]:
        return self.graph.query(cypher, params=params or {})

    def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        cypher_query_keyword = """
            MATCH (n)
            WHERE n.id = $node_id OR n.role = $node_id
            MATCH (n)-[r]-(other)
            RETURN DISTINCT
                CASE WHEN startNode(r) = n THEN n.id ELSE other.id END AS source,
                type(r) AS relationship,
                CASE WHEN endNode(r) = n THEN n.id ELSE other.id END AS target
            LIMIT $limit
        """
       
        """
        # lấy các query có node_id hoặc role trùng với query
        MATCH (n)
            WHERE n.id = $node_id OR n.role = $node_id
            WITH n
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n.id AS source, type(r) AS relationship, m.id AS target
            UNION
            MATCH (n)
            WHERE n.id = $node_id OR n.role = $node_id
            WITH n
            OPTIONAL MATCH (s)-[r]->(n)
            RETURN s.id AS source, type(r) AS relationship, n.id AS target
            LIMIT $limit
        """
        
        params = {"node_id": query, "limit": limit}
        results = self._query_graph(cypher_query_keyword, params)

        if not results:
            logger.info(f"No results found for keyword '{query}', trying vector search...")
            query_embedding = self._get_embedding(query)
            
            cypher_query_vector = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT 1
                
                CALL (n) {
                    MATCH (n)-[r]->(m)
                    RETURN n.id AS source, type(r) AS relationship, m.id AS target
                    
                    UNION
                    
                    MATCH (s)-[r]->(n)
                    RETURN s.id AS source, type(r) AS relationship, n.id AS target
                }
                
                RETURN source, relationship, target
                LIMIT $limit
            """
            params = {"query_embedding": query_embedding, "limit": limit}
            results = self._query_graph(cypher_query_vector, params)

        return results
    
if __name__ == "__main__":
    retrieval = GraphRetriever(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password
    )
    query = "Who was ruled England?"
    results = retrieval.retrieve(query, limit=20)

    if results:
        logger.info(f"Retrieved {len(results)} relationships for '{query}':")
        for result in results:
            print(f"{result['source']} -[{result['relationship']}]-> {result['target']}")
    else:
        logger.warning(f"No results found for '{query}'.")