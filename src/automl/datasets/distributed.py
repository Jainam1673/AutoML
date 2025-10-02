"""Distributed data processing with Dask for exabyte-scale datasets.

Handles datasets larger than memory using Dask DataFrames and distributed arrays.
Supports streaming from cloud storage (S3, GCS, Azure Blob).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

try:
    import dask
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

__all__ = [
    "DaskDataLoader",
    "StreamingDataset",
    "ChunkedDataProcessor",
]

logger = logging.getLogger(__name__)


class DaskDataLoader:
    """Load and process exabyte-scale datasets with Dask.
    
    Features:
    - Out-of-core computation (datasets larger than RAM)
    - Distributed processing across cluster
    - Lazy evaluation for efficiency
    - Cloud storage integration (S3, GCS, Azure)
    - Automatic partitioning and parallelization
    """
    
    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "4GB",
        scheduler_address: str | None = None,
    ) -> None:
        """Initialize Dask data loader.
        
        Args:
            n_workers: Number of worker processes
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            scheduler_address: External scheduler address (None for local)
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not installed. Install with: pip install 'dask[complete]' or "
                "uv pip install 'dask[complete]'"
            )
        
        self.client: Client | None = None
        self._setup_cluster(n_workers, threads_per_worker, memory_limit, scheduler_address)
    
    def _setup_cluster(
        self,
        n_workers: int,
        threads_per_worker: int,
        memory_limit: str,
        scheduler_address: str | None,
    ) -> None:
        """Setup Dask cluster."""
        if scheduler_address:
            # Connect to external cluster
            self.client = Client(scheduler_address)
            logger.info(f"Connected to Dask cluster at {scheduler_address}")
        else:
            # Create local cluster
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
            )
            self.client = Client(cluster)
            logger.info(
                f"Started local Dask cluster with {n_workers} workers, "
                f"{threads_per_worker} threads each, {memory_limit} memory limit"
            )
        
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")
    
    def load_csv(
        self,
        path: str | Path,
        chunksize: str = "100MB",
        **kwargs: Any,
    ) -> dd.DataFrame:
        """Load CSV file(s) as Dask DataFrame.
        
        Supports:
        - Local files: "/path/to/data.csv"
        - S3: "s3://bucket/data/*.csv"
        - GCS: "gcs://bucket/data/*.csv"
        - Azure: "abfs://container/data/*.csv"
        
        Args:
            path: File path or glob pattern
            chunksize: Size of each partition
            **kwargs: Additional arguments for dd.read_csv
        
        Returns:
            Dask DataFrame (lazy evaluation)
        """
        logger.info(f"Loading CSV from {path} with chunksize {chunksize}")
        
        ddf = dd.read_csv(
            path,
            blocksize=chunksize,
            **kwargs,
        )
        
        logger.info(
            f"Loaded DataFrame with {ddf.npartitions} partitions, "
            f"shape {ddf.shape[0].compute()} x {len(ddf.columns)}"
        )
        
        return ddf
    
    def load_parquet(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> dd.DataFrame:
        """Load Parquet file(s) as Dask DataFrame.
        
        Parquet is recommended for large datasets due to:
        - Columnar format (efficient reading)
        - Built-in compression
        - Schema preservation
        - Fast filtering
        
        Args:
            path: File path or directory
            **kwargs: Additional arguments for dd.read_parquet
        
        Returns:
            Dask DataFrame
        """
        logger.info(f"Loading Parquet from {path}")
        
        ddf = dd.read_parquet(path, **kwargs)
        
        logger.info(
            f"Loaded Parquet with {ddf.npartitions} partitions, "
            f"{len(ddf.columns)} columns"
        )
        
        return ddf
    
    def to_sklearn_arrays(
        self,
        ddf: dd.DataFrame,
        target_col: str,
        feature_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Dask DataFrame to numpy arrays for sklearn.
        
        Warning: This loads data into memory. Use only for small datasets
        or after significant filtering/aggregation.
        
        Args:
            ddf: Dask DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (None = all except target)
        
        Returns:
            (X, y) numpy arrays
        """
        if feature_cols is None:
            feature_cols = [col for col in ddf.columns if col != target_col]
        
        logger.info("Converting Dask DataFrame to numpy arrays (loading into memory)")
        
        X = ddf[feature_cols].compute().values
        y = ddf[target_col].compute().values
        
        logger.info(f"Converted to arrays: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def sample(
        self,
        ddf: dd.DataFrame,
        frac: float = 0.01,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Sample from large dataset for prototyping.
        
        Args:
            ddf: Dask DataFrame
            frac: Fraction to sample (0.01 = 1%)
            random_state: Random seed
        
        Returns:
            Pandas DataFrame (in memory)
        """
        logger.info(f"Sampling {frac*100:.2f}% of dataset")
        
        sample_df = ddf.sample(frac=frac, random_state=random_state).compute()
        
        logger.info(f"Sampled {len(sample_df)} rows")
        
        return sample_df
    
    def close(self) -> None:
        """Close Dask client."""
        if self.client:
            self.client.close()
            logger.info("Dask client closed")


class StreamingDataset:
    """Streaming dataset for incremental learning on infinite data.
    
    Useful for:
    - Real-time data streams
    - Datasets too large for disk
    - Online learning scenarios
    - Continuous model updates
    """
    
    def __init__(
        self,
        source: str,
        batch_size: int = 1000,
        buffer_size: int = 10000,
    ) -> None:
        """Initialize streaming dataset.
        
        Args:
            source: Data source (URL, file, database, etc.)
            batch_size: Rows per batch
            buffer_size: Internal buffer size
        """
        self.source = source
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._buffer: list[Any] = []
    
    def __iter__(self):
        """Iterate over batches."""
        # Implementation depends on data source
        # This is a template
        raise NotImplementedError("Subclass must implement __iter__")
    
    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Get next batch of data.
        
        Returns:
            (X, y) for next batch
        """
        raise NotImplementedError("Subclass must implement next_batch")


class ChunkedDataProcessor:
    """Process data in chunks to handle datasets larger than memory.
    
    Applies transformations chunk-by-chunk without loading entire dataset.
    """
    
    def __init__(self, chunk_size: int = 100000) -> None:
        """Initialize chunked processor.
        
        Args:
            chunk_size: Rows per chunk
        """
        self.chunk_size = chunk_size
    
    def process_csv_in_chunks(
        self,
        input_path: str | Path,
        output_path: str | Path,
        transform_fn: Any,
    ) -> None:
        """Process CSV file in chunks.
        
        Args:
            input_path: Input CSV path
            output_path: Output CSV path
            transform_fn: Function to apply to each chunk
        """
        logger.info(
            f"Processing {input_path} in chunks of {self.chunk_size} rows"
        )
        
        first_chunk = True
        chunks_processed = 0
        
        for chunk in pd.read_csv(input_path, chunksize=self.chunk_size):
            # Apply transformation
            transformed_chunk = transform_fn(chunk)
            
            # Write to output
            mode = "w" if first_chunk else "a"
            header = first_chunk
            
            transformed_chunk.to_csv(
                output_path,
                mode=mode,
                header=header,
                index=False,
            )
            
            first_chunk = False
            chunks_processed += 1
            
            if chunks_processed % 10 == 0:
                logger.info(f"Processed {chunks_processed} chunks")
        
        logger.info(
            f"Completed processing {chunks_processed} chunks, "
            f"output saved to {output_path}"
        )
    
    def fit_transform_incremental(
        self,
        path: str | Path,
        transformer: Any,
        target_col: str | None = None,
    ) -> None:
        """Fit transformer incrementally on large dataset.
        
        Useful for StandardScaler, MinMaxScaler, etc. that support
        partial_fit() method.
        
        Args:
            path: Path to CSV file
            transformer: sklearn transformer with partial_fit()
            target_col: Target column to exclude
        """
        logger.info(f"Fitting transformer incrementally on {path}")
        
        for chunk in pd.read_csv(path, chunksize=self.chunk_size):
            if target_col:
                X = chunk.drop(columns=[target_col])
            else:
                X = chunk
            
            # Incremental fit
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X)
            else:
                raise ValueError(
                    f"Transformer {type(transformer).__name__} does not support partial_fit"
                )
        
        logger.info("Incremental fitting completed")
