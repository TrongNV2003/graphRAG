from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IndexingResponse(BaseModel):
    status: str
    message: str
    entities_count: int
    relationships_count: int
    chunks_count: int

class QueryResponse(BaseModel):
    answer: str
    graph_context: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_context: List[Dict[str, Any]] = Field(default_factory=list)

class SearchResponse(BaseModel):
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    total: int
    search_type: str
