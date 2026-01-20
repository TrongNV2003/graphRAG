from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from langchain_neo4j import Neo4jGraph
from openai import OpenAI
from typing import List

from src.api.dependencies import get_openai_client, get_neo4j_graph, get_data_loader
from src.api.schemas.requests import WikipediaIndexRequest
from src.api.schemas.responses import IndexingResponse
from src.services.indexing import GraphIndexing
from src.config.schemas import StructuralChunk

router = APIRouter()

@router.post("/wikipedia", response_model=IndexingResponse)
async def index_wikipedia(
    request: WikipediaIndexRequest,
    client: OpenAI = Depends(get_openai_client),
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
    dataloader = Depends(get_data_loader)
):
    try:
        # Load documents
        raw_docs = dataloader.load(request.query_keyword, load_max_docs=request.max_docs)
        if not raw_docs:
            raise HTTPException(status_code=404, detail="No documents found for the given keyword")

        # Initialize indexing service
        indexing_service = GraphIndexing(
            client=client,
            graph_db=graph_db,
            chunk_size=2048,
            clear_old_graph=request.clear_old
        )

        # Chunk documents
        chunks: List[StructuralChunk] = []
        for doc in raw_docs:
            chunks.extend(indexing_service.chunking(doc["content"]))

        # Perform indexing
        indexing_service.indexing(chunks=chunks)

        # Get stats
        entity_count = graph_db.query("MATCH (e:Entity) RETURN count(e) as count")[0]["count"]
        rel_count = graph_db.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        chunk_count = graph_db.query("MATCH (c:Chunk) RETURN count(c) as count")[0]["count"]

        return IndexingResponse(
            status="success",
            message=f"Successfully indexed {len(raw_docs)} documents",
            entities_count=entity_count,
            relationships_count=rel_count,
            chunks_count=chunk_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
