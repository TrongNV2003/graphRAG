import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.config.setting import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    
    # Startup
    logger.info("Starting HybridRAG")
    
    try:
        from src.api.dependencies import get_neo4j_graph
        from src.core.storage import GraphStorage
        
        graph_db = get_neo4j_graph()
        storage = GraphStorage(graph_db)
        storage.setup_schema()
        logger.info("Graph schema initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize graph schema on startup: {e}")
        
    yield
    
    # Shutdown
    logger.info("Shutting down HybridRAG")


def create_app() -> FastAPI:
    """Factory function to create FastAPI application"""
    
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        }
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with structured response"""
        errors = exc.errors()
        logger.warning(f"Validation error on {request.url.path}: {errors}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Validation Error",
                "detail": errors,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions - prevent stacktrace leak"""
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for load balancers"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }


    # Include Routes
    from src.api.routes import indexing, querying, graph, backup
    
    app.include_router(indexing.router, prefix=f"{settings.api_v1_prefix}/indexing", tags=["Indexing"])
    app.include_router(querying.router, prefix=f"{settings.api_v1_prefix}/query", tags=["Querying"])
    app.include_router(graph.router, prefix=f"{settings.api_v1_prefix}/graph", tags=["Graph"])
    app.include_router(backup.router, prefix=f"{settings.api_v1_prefix}/backup", tags=["Backup"])
    
    return app


app = create_app()
