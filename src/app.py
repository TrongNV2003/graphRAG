from loguru import logger
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.api.routes import indexing, querying, graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    
    # Startup
    logger.info("Starting HybridRAG API")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HybridRAG API")


def create_app() -> FastAPI:
    """Factory function to create FastAPI application"""
    
    app = FastAPI(
        title="HybridRAG API",
        description="API for HybridRAG system combining Knowledge Graph and Vector Database",
        version="1.0.0",
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
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests with timing"""
        start_time = datetime.now()
        
        logger.debug(f"Request: {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with structured response"""
        errors = exc.errors()
        logger.warning(f"Validation error on {request.url.path}: {errors}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            "name": "HybridRAG API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

    @app.get("/api/v1/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for load balancers"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}


    # Include Routes
    app.include_router(indexing.router, prefix="/api/v1/indexing", tags=["Indexing"])
    app.include_router(querying.router, prefix="/api/v1/query", tags=["Querying"])
    app.include_router(graph.router, prefix="/api/v1/graph", tags=["Graph"])
    
    return app


app = create_app()
