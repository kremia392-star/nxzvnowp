"""
BDH Interpretability Suite - FastAPI Backend

Provides REST API for:
- Model inference with activation extraction
- Sparsity analysis
- Monosemanticity probing
- Graph topology queries
- Real-time Hebbian tracking

The frontend can work in two modes:
1. Live mode: Calls these APIs for real-time analysis
2. Playback mode: Uses pre-computed JSON (no backend needed)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

# Import routes
from backend.routes import inference, analysis, models, visualization


# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the project root directory (parent of backend/)
_PROJECT_ROOT = Path(__file__).parent.parent

class Settings:
    """Application settings."""
    PROJECT_NAME: str = "BDH Interpretability Suite"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Model paths - resolve relative to project root
    CHECKPOINT_DIR: Path = Path(os.getenv("CHECKPOINT_DIR", str(_PROJECT_ROOT / "checkpoints")))
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "french")
    
    # Device
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]


settings = Settings()


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    print("=" * 60)
    print(f"[BDH] {settings.PROJECT_NAME} v{settings.VERSION}")
    print("=" * 60)
    print(f"Device: {settings.DEVICE}")
    print(f"Checkpoint dir: {settings.CHECKPOINT_DIR}")
    
    # Initialize model registry
    from backend.services.model_service import ModelService
    app.state.model_service = ModelService(
        checkpoint_dir=settings.CHECKPOINT_DIR,
        device=settings.DEVICE
    )
    
    # Try to load default model
    try:
        app.state.model_service.load_model(settings.DEFAULT_MODEL)
        print(f"Loaded default model: {settings.DEFAULT_MODEL}")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
    
    print("=" * 60)
    print("[READY] Server ready!")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n[SHUTDOWN] Shutting down...")
    app.state.model_service.unload_all()


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    Backend API for the BDH Interpretability Suite.
    
    ## Features
    
    - **Inference**: Run text through BDH model with activation extraction
    - **Analysis**: Sparsity measurement, monosemanticity probing
    - **Visualization**: Graph topology, attention patterns
    - **Models**: Load/unload model checkpoints
    
    ## Modes
    
    The frontend can work in two modes:
    1. **Live mode**: Calls these APIs for real-time analysis
    2. **Playback mode**: Uses pre-computed JSON (backend optional)
    """,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTES
# =============================================================================

app.include_router(
    inference.router,
    prefix=f"{settings.API_PREFIX}/inference",
    tags=["inference"]
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_PREFIX}/analysis",
    tags=["analysis"]
)

app.include_router(
    models.router,
    prefix=f"{settings.API_PREFIX}/models",
    tags=["models"]
)

app.include_router(
    visualization.router,
    prefix=f"{settings.API_PREFIX}/visualization",
    tags=["visualization"]
)


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": settings.DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get(f"{settings.API_PREFIX}/status")
async def api_status():
    """API status with loaded models."""
    model_service = app.state.model_service
    
    return {
        "status": "running",
        "loaded_models": model_service.list_loaded_models(),
        "available_models": model_service.list_available_models(),
        "device": settings.DEVICE,
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    import traceback
    tb = traceback.format_exc()
    print(f"[ERROR] {type(exc).__name__}: {exc}\n{tb}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
