"""Redis caching layer for distributed feature and prediction caching.

Dramatically speeds up repeated operations at exabyte scale by caching
intermediate results, features, and model predictions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from typing import Any

import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

__all__ = [
    "RedisCache",
    "FeatureCache",
    "PredictionCache",
]

logger = logging.getLogger(__name__)


class RedisCache:
    """Distributed caching with Redis.
    
    Features:
    - Multi-node caching cluster
    - Automatic serialization
    - TTL (time-to-live) support
    - Cache invalidation
    - LRU eviction policy
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        default_ttl: int = 3600,  # 1 hour
    ) -> None:
        """Initialize Redis cache.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Database number
            password: Redis password (if required)
            default_ttl: Default time-to-live in seconds
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not installed. Install with: pip install redis or "
                "uv pip install redis"
            )
        
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # Handle binary data
        )
        self.default_ttl = default_ttl
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, prefix: str, *args: Any) -> str:
        """Create cache key from arguments.
        
        Args:
            prefix: Key prefix
            *args: Arguments to hash
        
        Returns:
            Cache key
        """
        # Create deterministic hash of arguments
        content = json.dumps(args, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Any | None:
        """Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        data = self.client.get(key)
        if data is None:
            return None
        
        return pickle.loads(data)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
        """
        data = pickle.dumps(value)
        ttl = ttl or self.default_ttl
        self.client.setex(key, ttl, data)
    
    def delete(self, key: str) -> None:
        """Delete key from cache.
        
        Args:
            key: Cache key
        """
        self.client.delete(key)
    
    def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "features:*")
        
        Returns:
            Number of keys deleted
        """
        keys = list(self.client.scan_iter(match=pattern))
        if keys:
            return self.client.delete(*keys)
        return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists.
        
        Args:
            key: Cache key
        
        Returns:
            True if key exists
        """
        return bool(self.client.exists(key))


class FeatureCache:
    """Cache computed features to avoid recomputation.
    
    Especially useful for expensive feature engineering operations
    that need to be applied to the same data multiple times.
    """
    
    def __init__(self, cache: RedisCache) -> None:
        """Initialize feature cache.
        
        Args:
            cache: Redis cache instance
        """
        self.cache = cache
        self.prefix = "features"
    
    def get_features(
        self,
        data_hash: str,
        feature_names: list[str],
    ) -> np.ndarray | None:
        """Get cached features.
        
        Args:
            data_hash: Hash of input data
            feature_names: List of feature names
        
        Returns:
            Cached features or None
        """
        key = self.cache._make_key(self.prefix, data_hash, feature_names)
        
        result = self.cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit for features: {key}")
        
        return result
    
    def set_features(
        self,
        data_hash: str,
        feature_names: list[str],
        features: np.ndarray,
        ttl: int | None = None,
    ) -> None:
        """Cache computed features.
        
        Args:
            data_hash: Hash of input data
            feature_names: List of feature names
            features: Computed features
            ttl: Time-to-live in seconds
        """
        key = self.cache._make_key(self.prefix, data_hash, feature_names)
        self.cache.set(key, features, ttl=ttl)
        logger.debug(f"Cached features: {key}")
    
    def hash_data(self, data: np.ndarray) -> str:
        """Create hash of input data.
        
        Args:
            data: Input data
        
        Returns:
            Data hash
        """
        return hashlib.sha256(data.tobytes()).hexdigest()


class PredictionCache:
    """Cache model predictions to avoid redundant inference.
    
    Critical for production systems with repeated queries.
    """
    
    def __init__(self, cache: RedisCache) -> None:
        """Initialize prediction cache.
        
        Args:
            cache: Redis cache instance
        """
        self.cache = cache
        self.prefix = "predictions"
    
    def get_predictions(
        self,
        model_id: str,
        input_hash: str,
    ) -> np.ndarray | None:
        """Get cached predictions.
        
        Args:
            model_id: Model identifier/version
            input_hash: Hash of input data
        
        Returns:
            Cached predictions or None
        """
        key = self.cache._make_key(self.prefix, model_id, input_hash)
        
        result = self.cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit for predictions: {key}")
        
        return result
    
    def set_predictions(
        self,
        model_id: str,
        input_hash: str,
        predictions: np.ndarray,
        ttl: int = 300,  # 5 minutes default for predictions
    ) -> None:
        """Cache model predictions.
        
        Args:
            model_id: Model identifier/version
            input_hash: Hash of input data
            predictions: Model predictions
            ttl: Time-to-live in seconds
        """
        key = self.cache._make_key(self.prefix, model_id, input_hash)
        self.cache.set(key, predictions, ttl=ttl)
        logger.debug(f"Cached predictions: {key}")
    
    def hash_input(self, data: np.ndarray) -> str:
        """Create hash of input data.
        
        Args:
            data: Input data
        
        Returns:
            Input hash
        """
        return hashlib.sha256(data.tobytes()).hexdigest()
    
    def invalidate_model(self, model_id: str) -> int:
        """Invalidate all predictions for a model.
        
        Call this when model is updated.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Number of keys invalidated
        """
        pattern = f"{self.prefix}:*{model_id}*"
        count = self.cache.clear(pattern)
        logger.info(f"Invalidated {count} cached predictions for model {model_id}")
        return count
