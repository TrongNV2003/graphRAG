from typing import List, Dict, Any
from pydantic import BaseModel, Field


# ===== Indexing Schemas =====

class WikipediaIndexRequest(BaseModel):
    query_keyword: str = Field(..., description="The keyword to search on Wikipedia")
    max_docs: int = Field(10, ge=1, le=50, description="Maximum number of documents to load")
    clear_old: bool = Field(False, description="Whether to clear the existing knowledge graph before indexing")


class IndexingResponse(BaseModel):
    status: str
    message: str
    entities_count: int
    relationships_count: int
    chunks_count: int


# ===== Querying Schemas =====

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural language query")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of chunk results to return")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Similarity threshold for filtering chunk results")
    graph_limit: int = Field(10, ge=1, le=50, description="Maximum number of graph results to return")


class QueryResponse(BaseModel):
    answer: str
    graph_context: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_context: List[Dict[str, Any]] = Field(default_factory=list)


# ===== Search Schemas =====

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of results to return")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Similarity threshold for filtering results")


class SearchResponse(BaseModel):
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    total: int
    search_type: str
