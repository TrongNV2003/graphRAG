import os
from langchain_neo4j import Neo4jGraph
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query

from src.engines.qdrant import QdrantVectorStore
from src.api.dependencies import get_neo4j_graph, get_qdrant_store
from src.services.visualize_service import visualize_knowledge_graph
from src.config.setting import qdrant_config

router = APIRouter()

@router.get("/stats")
async def get_graph_stats(
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
    qdrant_store: QdrantVectorStore = Depends(get_qdrant_store)
):
    try:
        entity_count = graph_db.query("MATCH (e:Entity) RETURN count(e) as count")[0]["count"]
        rel_count = graph_db.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        
        try:
            collection_info = qdrant_store.get_collection_info(qdrant_config.collection_name)
            chunk_count = collection_info.get("points_count", 0) if collection_info else 0
        except Exception:
            chunk_count = 0
        
        return {
            "entities_count": entity_count,
            "relationships_count": rel_count,
            "chunks_count": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def remove_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass

@router.get("/visualize", response_class=HTMLResponse)
async def get_graph_visualization(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, ge=1, le=1000),
    graph_db: Neo4jGraph = Depends(get_neo4j_graph)
):
    try:
        html_path = visualize_knowledge_graph(graph_db=graph_db, limit=limit)
        
        if not html_path or not os.path.exists(html_path):
            raise HTTPException(status_code=404, detail="Could not generate graph visualization")
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Schedule file removal
        background_tasks.add_task(remove_temp_file, html_path)
        
        return html_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
