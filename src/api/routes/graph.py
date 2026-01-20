import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
from langchain_neo4j import Neo4jGraph

from src.api.dependencies import get_neo4j_graph
from src.services.visualization import visualize_knowledge_graph

router = APIRouter()

@router.get("/stats")
async def get_graph_stats(graph_db: Neo4jGraph = Depends(get_neo4j_graph)):
    try:
        entity_count = graph_db.query("MATCH (e:Entity) RETURN count(e) as count")[0]["count"]
        rel_count = graph_db.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        chunk_count = graph_db.query("MATCH (c:Chunk) RETURN count(c) as count")[0]["count"]
        
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
