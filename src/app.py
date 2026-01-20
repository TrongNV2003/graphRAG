from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import indexing, querying, graph

def create_app() -> FastAPI:
    app = FastAPI(
        title="HybridRAG API",
        description="API for HybridRAG system combining Knowledge Graph and Vector Database",
        version="1.0.0",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routes
    app.include_router(indexing.router, prefix="/api/v1/indexing", tags=["Indexing"])
    app.include_router(querying.router, prefix="/api/v1/query", tags=["Querying"])
    app.include_router(graph.router, prefix="/api/v1/graph", tags=["Graph"])

    @app.get("/api/v1/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy"}

    logger.info("HybridRAG API initialized")
    return app

app = create_app()
