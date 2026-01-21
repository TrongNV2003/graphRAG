import os
from typing import List, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    app_name: str = "HybridRAG"
    app_description: str = "API for HybridRAG system combining Knowledge Graph and Vector Database"
    app_version: str = "1.0.0"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"
    
    log_level: str = "INFO"


class APIConfig(BaseSettings):
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for OpenAI API",
        alias="API_URL",
    )
    api_key: Optional[str] = Field(
        description="API key for OpenAI",
        alias="API_KEY",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI API",
        alias="OPENAI_API_KEY",
    )


class LLMTaskParams(BaseModel):
    max_tokens: int = Field(default=8192, description="Maximum number of tokens for API responses")
    temperature: float = Field(default=0.0, description="Sampling temperature; higher values make output more random")
    top_p: float = Field(default=0.95, description="Nucleus sampling parameter; higher values increase randomness")
    presence_penalty: float = Field(default=0.5, description="Penalty for new tokens based on existing ones; higher values discourage repetition, range [-2.0, 2.0] but typically suggest to [-1.0, 1.0]")
    frequency_penalty: float = Field(default=0.5, description="Frequency penalty for new tokens based on their frequency; higher values discourage frequent tokens, range [-2.0, 2.0] but typically suggest to [-1.0, 1.0]")


class LLMConfig(BaseSettings):
    llm_model: str = Field(
        default="Qwen/Qwen3-4B",
        description="Large Language model name to be used (e.g., GPT-4)",
        alias="LLM_MODEL",
    )
    stop_tokens: List[str] = Field(
        default=["</s>", "EOS", "<|im_end|>"],
        alias="STOP_TOKENS",
        description="Tokens that indicate the end of a sequence",
    )
    seed: int = Field(
        default=42,
        alias="SEED",
        description="Random seed for sampling"
    )
    
    @property
    def generation(self) -> LLMTaskParams:
        return LLMTaskParams(
            max_tokens=int(os.getenv("GENERATION_MAX_TOKENS", 16384)),
            temperature=float(os.getenv("GENERATION_TEMPERATURE", 0.5)),
            top_p=float(os.getenv("GENERATION_TOP_P", 0.95)),
            presence_penalty=float(os.getenv("GENERATION_PRESENCE_PENALTY", 0.5)),
            frequency_penalty=float(os.getenv("GENERATION_FREQUENCY_PENALTY", 0.5)),
        )
    
    @property
    def extraction(self) -> LLMTaskParams:
        return LLMTaskParams(
            max_tokens=int(os.getenv("EXTRACTION_MAX_TOKENS", 4096)),
            temperature=float(os.getenv("EXTRACTION_TEMPERATURE", 0.3)),
            top_p=float(os.getenv("EXTRACTION_TOP_P", 0.95)),
            presence_penalty=float(os.getenv("EXTRACTION_PRESENCE_PENALTY", 0.5)),
            frequency_penalty=float(os.getenv("EXTRACTION_FREQUENCY_PENALTY", 0.5)),
        )
    
    class Config:
        env_nested_delimiter = '_'


class Neo4jConfig(BaseSettings):
    username: str = Field(..., alias="NEO4J_USERNAME")
    password: str = Field(..., alias="NEO4J_PASSWORD")
    url: str = Field(..., alias="NEO4J_URI")

    def get_driver(self):
        driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        driver.verify_connectivity()
        return driver


class EmbeddingModelConfig(BaseSettings):
    embedder_model: str = Field(
        default="contextboxai/halong_embedding",
        description="Model name for sentence embedding",
        alias="EMBEDDER_MODEL"
    )


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration"""
    url: Optional[str] = Field(
        default=None,
        description="Qdrant Cloud URL (optional)",
        alias="QDRANT_URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (optional)",
        alias="QDRANT_API_KEY"
    )
    host: str = Field(
        default="localhost",
        description="Local Qdrant host",
        alias="QDRANT_HOST"
    )
    port: int = Field(
        default=6333,
        description="Local Qdrant port",
        alias="QDRANT_PORT"
    )
    collection_name: str = Field(
        default="hybridrag_chunks",
        description="Collection name for chunk vectors",
        alias="QDRANT_COLLECTION"
    )


class RetrievalConfig(BaseSettings):
    """Configuration Fuzzy retrieval"""
    fuzzy_min_score: float = Field(
        default=0.5,
        description="Minimum score threshold for fuzzy fulltext search (0.0-1.0)",
        alias="FUZZY_MIN_SCORE"
    )

settings = Settings()
api_config = APIConfig()
llm_config = LLMConfig()
neo4j_config = Neo4jConfig()
embed_config = EmbeddingModelConfig()
qdrant_config = QdrantConfig()
retrieval_config = RetrievalConfig()
