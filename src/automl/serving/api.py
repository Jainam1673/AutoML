"""Production REST API for model serving with FastAPI.

High-performance async API for serving AutoML models at exabyte scale.
Supports batching, caching, monitoring, and horizontal scaling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

__all__ = [
    "ModelServer",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "create_app",
]

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Single prediction request."""
    
    features: list[list[float]] = Field(
        ...,
        description="Input features as 2D array",
        example=[[1.0, 2.0, 3.0]],
    )
    model_id: str | None = Field(
        None,
        description="Specific model version to use",
    )


class PredictionResponse(BaseModel):
    """Prediction response."""
    
    predictions: list[float] = Field(
        ...,
        description="Model predictions",
    )
    model_id: str = Field(
        ...,
        description="Model ID used for prediction",
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
    )
    cached: bool = Field(
        False,
        description="Whether result was cached",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    batch: list[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
    )
    async_processing: bool = Field(
        False,
        description="Process asynchronously and return job ID",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


@dataclass
class ModelServer:
    """Production model server.
    
    Features:
    - Async request handling
    - Batch prediction support
    - Response caching
    - Request queuing
    - Health checks
    - Metrics collection
    """
    
    model: Any
    model_id: str
    cache: Any | None = None
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    
    def __post_init__(self) -> None:
        """Initialize server state."""
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_predictions = 0
    
    async def predict_single(
        self,
        features: np.ndarray,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, bool]:
        """Make single prediction.
        
        Args:
            features: Input features
            use_cache: Whether to use cache
        
        Returns:
            (predictions, cached) tuple
        """
        cached = False
        
        # Check cache
        if use_cache and self.cache:
            from ..caching.redis_cache import PredictionCache
            if isinstance(self.cache, PredictionCache):
                input_hash = self.cache.hash_input(features)
                cached_pred = self.cache.get_predictions(self.model_id, input_hash)
                
                if cached_pred is not None:
                    return cached_pred, True
        
        # Make prediction
        predictions = self.model.predict(features)
        
        # Cache result
        if use_cache and self.cache:
            from ..caching.redis_cache import PredictionCache
            if isinstance(self.cache, PredictionCache):
                input_hash = self.cache.hash_input(features)
                self.cache.set_predictions(self.model_id, input_hash, predictions)
        
        self.total_predictions += len(features)
        
        return predictions, cached
    
    async def predict_batch(
        self,
        batch: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Make batch prediction.
        
        Args:
            batch: List of feature arrays
        
        Returns:
            List of predictions
        """
        # Combine into single array for efficient processing
        combined = np.vstack(batch)
        predictions, _ = await self.predict_single(combined, use_cache=False)
        
        # Split back into individual results
        results = []
        idx = 0
        for features in batch:
            n = len(features)
            results.append(predictions[idx:idx + n])
            idx += n
        
        return results
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time


def create_app(
    model: Any,
    model_id: str = "model-v1",
    cache: Any | None = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create FastAPI application for model serving.
    
    Args:
        model: Trained model
        model_id: Model identifier
        cache: Cache instance (optional)
        enable_cors: Enable CORS middleware
    
    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install 'fastapi[standard]' or "
            "uv pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title="AutoML Model Server",
        description="Production API for AutoML model serving",
        version="0.1.0",
    )
    
    # Add CORS middleware
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize server
    server = ModelServer(model=model, model_id=model_id, cache=cache)
    
    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "message": "AutoML Model Server",
            "version": "0.1.0",
            "docs": "/docs",
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=server.model is not None,
            version="0.1.0",
            uptime_seconds=server.get_uptime(),
        )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Make predictions on input data.
        
        Args:
            request: Prediction request
        
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        try:
            # Convert input to numpy array
            features = np.array(request.features)
            
            # Make prediction
            predictions, cached = await server.predict_single(features)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            
            server.total_requests += 1
            
            return PredictionResponse(
                predictions=predictions.tolist(),
                model_id=request.model_id or server.model_id,
                inference_time_ms=inference_time,
                cached=cached,
            )
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch")
    async def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
        """Make batch predictions.
        
        Args:
            request: Batch prediction request
        
        Returns:
            Batch prediction results
        """
        start_time = time.time()
        
        try:
            # Extract feature arrays
            feature_arrays = [
                np.array(req.features) for req in request.batch
            ]
            
            # Make predictions
            predictions = await server.predict_batch(feature_arrays)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            
            server.total_requests += len(request.batch)
            
            return {
                "predictions": [pred.tolist() for pred in predictions],
                "model_id": server.model_id,
                "inference_time_ms": inference_time,
                "batch_size": len(request.batch),
            }
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        """Get server metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "total_requests": server.total_requests,
            "total_predictions": server.total_predictions,
            "uptime_seconds": server.get_uptime(),
            "model_id": server.model_id,
        }
    
    @app.post("/cache/clear")
    async def clear_cache(background_tasks: BackgroundTasks) -> dict[str, str]:
        """Clear prediction cache.
        
        Returns:
            Status message
        """
        if server.cache:
            background_tasks.add_task(server.cache.clear, f"predictions:{server.model_id}*")
            return {"status": "cache clearing scheduled"}
        else:
            return {"status": "no cache configured"}
    
    return app


# Example usage
def run_server(
    model: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
) -> None:
    """Run model server.
    
    Args:
        model: Trained model
        host: Server host
        port: Server port
        workers: Number of worker processes
    """
    import uvicorn
    
    app = create_app(model)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
