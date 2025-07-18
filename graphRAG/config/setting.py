from pydantic import Field
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic_settings import BaseSettings

load_dotenv()

class LLMConfig(BaseSettings):
    base_url: str = Field(
        description="Base URL for OpenAI API",
        alias="LLM_URL",
    )
    api_key: str = Field(
        description="API key for OpenAI",
        alias="LLM_KEY",
    )
    model: str = Field(
        description="Model name to be used (e.g., GPT-4)",
        alias="LLM_MODEL",
    )
    max_tokens: int = Field(
        default=1024,
        alias="MAX_TOKENS",
        description="Maximum number of tokens for API responses",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature; higher values make output more random",
        alias="TEMPERATURE",
    )
    top_p: float = Field(
        default=0.95,
        alias="TOP_P",
        description="Nucleus sampling parameter; higher values increase randomness",
    )
    seed: int = Field(default=42, alias="SEED", description="Random seed for sampling")

class Neo4jConfig(BaseSettings):
    username: str = Field(..., alias="NEO4J_USERNAME")
    password: str = Field(..., alias="NEO4J_PASSWORD")
    url: str = Field(..., alias="NEO4J_URI")

    def get_driver(self):
        driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        driver.verify_connectivity()
        return driver

class ModelConfig(BaseSettings):
    embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for sentence embedding",
        alias="EMBEDDER_MODEL"
    )

llm_config = LLMConfig()
neo4j_config = Neo4jConfig()
model_config = ModelConfig()
