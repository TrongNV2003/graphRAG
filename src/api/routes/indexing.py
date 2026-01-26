from typing import List
from fastapi import APIRouter, Depends, HTTPException

from src.services.index_service import GraphIndexing
from src.config.dataclass import StructuralChunk
from src.config.schemas import WikipediaIndexRequest, IndexingResponse
from src.api.dependencies import get_data_loader, get_indexing_service

router = APIRouter()


@router.post("/wikipedia", response_model=IndexingResponse)
async def index_wikipedia(
    request: WikipediaIndexRequest,
    dataloader = Depends(get_data_loader),
    indexing_service: GraphIndexing = Depends(get_indexing_service)
):
    try:
        raw_docs = dataloader.load(request.query_keyword, load_max_docs=request.max_docs)
        if not raw_docs:
            raise HTTPException(status_code=404, detail="No documents found for the given keyword")

        # Config override
        indexing_service.clear_old_graph = request.clear_old

        # Chunk documents
        chunks: List[StructuralChunk] = []
        for doc in raw_docs:
            chunks.extend(indexing_service.chunking(doc["content"]))

        indexing_service.indexing(chunks=chunks)

        entity_count = indexing_service.graph_db.query("MATCH (e:Entity) RETURN count(e) as count")[0]["count"]
        rel_count = indexing_service.graph_db.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        chunk_count = indexing_service.graph_db.query("MATCH (c:Chunk) RETURN count(c) as count")[0]["count"]

        return IndexingResponse(
            status="success",
            message=f"Successfully indexed {len(raw_docs)} documents",
            entities_count=entity_count,
            relationships_count=rel_count,
            chunks_count=chunk_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
