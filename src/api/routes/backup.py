from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from langchain_neo4j import Neo4jGraph
import tempfile
import os

from src.api.dependencies import get_neo4j_graph
from src.core.storage import GraphStorage

router = APIRouter()


@router.get("/backup")
async def backup_graph(
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
):
    """
    Backup the entire graph to a ZIP file containing CSVs.
    
    Returns a downloadable ZIP file with:
    - nodes.csv
    - relationships.csv
    - metadata.json
    """
    try:
        storage = GraphStorage(graph_db)
        
        temp_dir = tempfile.mkdtemp()
        backup_path = os.path.join(temp_dir, "graph_backup.zip")
        
        metadata = storage.backup_graph(backup_path)
        
        return FileResponse(
            path=backup_path,
            media_type="application/zip",
            filename="graph_backup.zip",
            headers={
                "X-Node-Count": str(metadata.get("node_count", 0)),
                "X-Relationship-Count": str(metadata.get("relationship_count", 0)),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/restore")
async def restore_graph(
    file: UploadFile = File(...),
    clear_existing: bool = True,
    graph_db: Neo4jGraph = Depends(get_neo4j_graph),
):
    """
    Restore graph from a backup ZIP file.
    
    Args:
        file: The backup ZIP file to restore from.
        clear_existing: Whether to clear existing data before restore (default: True).
        
    Returns:
        Restore statistics including counts and any errors.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    try:
        storage = GraphStorage(graph_db)
        
        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "uploaded_backup.zip")
        
        with open(temp_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)
        
        result = storage.restore_graph(temp_path, clear_existing=clear_existing)
        
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return {
            "status": "success",
            "nodes_restored": result.get("nodes_restored", 0),
            "relationships_restored": result.get("relationships_restored", 0),
            "errors": result.get("errors", []),
            "original_metadata": result.get("original_metadata", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")
