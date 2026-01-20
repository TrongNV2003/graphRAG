from fastapi import APIRouter, Depends, HTTPException
from langchain_neo4j import Neo4jGraph
from openai import OpenAI

from src.api.dependencies import get_openai_client, get_neo4j_graph
from src.api.schemas.requests import QueryRequest, SearchRequest
from src.api.schemas.responses import QueryResponse, SearchResponse
from src.services.querying import GraphQuerying

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_knowledge_graph(
    request: QueryRequest,
    client: OpenAI = Depends(get_openai_client),
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
):
    """Full Hybrid Retrieval: Graph RAG + Qdrant Hybrid Search with LLM generation."""
    try:
        querying_service = GraphQuerying(client=client, graph_db=graph_db)
        
        # Use the new detailed response method
        result = querying_service.response_detailed(query=request.query)
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic", response_model=QueryResponse)
async def semantic_search(
    request: SearchRequest,
    client: OpenAI = Depends(get_openai_client),
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
):
    """Semantic Search: Dense vector search + LLM generation."""
    try:
        querying_service = GraphQuerying(client=client, graph_db=graph_db)
        result = querying_service.semantic_response(query=request.query, top_k=request.top_k, threshold=request.threshold)
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid", response_model=QueryResponse)
async def hybrid_search(
    request: SearchRequest,
    client: OpenAI = Depends(get_openai_client),
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
):
    """Hybrid Search: Qdrant hybrid (Dense + Sparse) + LLM generation."""
    try:
        querying_service = GraphQuerying(client=client, graph_db=graph_db)
        result = querying_service.hybrid_response(query=request.query, top_k=request.top_k, threshold=request.threshold)
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
