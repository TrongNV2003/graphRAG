from functools import lru_cache
from openai import OpenAI
from langchain_neo4j import Neo4jGraph

from src.config.setting import api_config, neo4j_config, llm_config
from src.processing.dataloaders import DataLoader
from src.engines.qdrant import create_qdrant_store, QdrantVectorStore

@lru_cache()
def get_openai_client() -> OpenAI:
    # Use OpenAI API for GPT models, custom API for others
    is_openai_model = llm_config.llm_model.lower().startswith(("gpt-", "o1-", "openai/"))
    
    if is_openai_model and api_config.openai_api_key:
        return OpenAI(api_key=api_config.openai_api_key)
    else:
        return OpenAI(api_key=api_config.api_key or "EMPTY", base_url=api_config.base_url)

@lru_cache()
def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password,
    )

@lru_cache()
def get_data_loader() -> DataLoader:
    return DataLoader()

@lru_cache()
def get_qdrant_store() -> QdrantVectorStore:
    return create_qdrant_store()
