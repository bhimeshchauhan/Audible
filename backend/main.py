"""
FastAPI backend for PDF character relationship extraction.

Run with:
    uvicorn backend.main:app --reload
Or:
    python -m backend.main
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .config import config

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="PDF Character Graph API",
    description="Extract character relationships from PDF books and return interactive graph HTML",
    version="1.0.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
)

# CORS middleware - configured via environment variables
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.origins,
    allow_credentials=config.cors.allow_credentials,
    allow_methods=config.cors.allow_methods,
    allow_headers=config.cors.allow_headers,
)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


# Root health check (convenience)
@app.get("/health")
async def root_health():
    """Root health check endpoint."""
    return {"status": "healthy", "environment": config.environment}


# =============================================================================
# Startup/Shutdown Events
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Log startup configuration."""
    print("\n" + "=" * 60)
    print("PDF Character Graph API Starting")
    print("=" * 60)
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"CORS Origins: {config.cors.origins}")
    print(f"Max Upload Size: {config.upload.max_size_bytes / (1024*1024):.0f}MB")
    print(f"LLM Mode: {'Local (Ollama)' if config.llm.use_local else 'Cloud (OpenAI)'}")
    print("=" * 60 + "\n")


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=config.server.workers if not config.server.reload else 1,
    )
