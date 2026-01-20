from pydantic import BaseModel, Field

class WikipediaIndexRequest(BaseModel):
    query_keyword: str = Field(..., description="The keyword to search on Wikipedia")
    max_docs: int = Field(10, ge=1, le=50, description="Maximum number of documents to load")
    clear_old: bool = Field(False, description="Whether to clear the existing knowledge graph before indexing")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural language query")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of results to return")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Similarity threshold for filtering results")
