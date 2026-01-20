from typing import List
from openai import OpenAI
from langchain_neo4j import Neo4jGraph

from src.config.dataclass import StructuralChunk
from src.processing.dataloaders import DataLoader
from src.services.indexing import GraphIndexing
from src.services.querying import GraphQuerying
from src.config.setting import neo4j_config, api_config


class Pipeline:
    def __init__(self):
        self.dataloader = DataLoader()

        self.client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)
    
        self.graph_db = Neo4jGraph(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )
        
    def pipeline_indexing(self, query_keyword: str, load_max_docs: int = 10) -> None:
        """Index documents into the graph database.
        
        Args:
            query_keyword: The keyword to search query in Wikipedia.
            load_max_docs: Maximum number of documents to load.
        """
        self.graph_indexing = GraphIndexing(
            self.client,
            graph_db=self.graph_db,
            chunk_size=2048,
            clear_old_graph=True
        )
        
        raw_docs = self.dataloader.load(query_keyword, load_max_docs=load_max_docs)
        
        chunks: List[StructuralChunk] = []
        for doc in raw_docs:
            chunks.extend(self.graph_indexing.chunking(doc["content"]))
            
        self.graph_indexing.indexing(chunks=chunks)
    
    def pipeline_querying(self, query: str) -> str:
        """Query the graph database and generate a response.
        
        Args:
            query: The input query string.
        
        Returns:
            The response generated from the graph database.
        """
        self.graph_querying = GraphQuerying(
            client=self.client,
            graph_db=self.graph_db
        )
        
        answer = self.graph_querying.response(query=query)
        return answer

    def visualize_knowledge_graph(self, limit: int = 100):
        """Visualize the knowledge graph."""
        from src.services.visualization import visualize_knowledge_graph
        visualize_knowledge_graph(self.graph_db, limit=limit)