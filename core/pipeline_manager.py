"""
Pipeline lifecycle management with type safety

This module provides the PipelineManager class for handling pipeline loading,
caching, and cleanup with full type safety and memory management.
"""

from typing import Dict, Optional, Callable, TypeVar, Generic, List, cast
import weakref
import gc
import torch
from contextlib import contextmanager

from ..types import PipelineProtocol, DeviceType, PipelineLoadError

P = TypeVar('P', bound=PipelineProtocol)

class PipelineManager(Generic[P]):
    """
    Type-safe pipeline lifecycle management with LRU caching
    
    Features:
    - Lazy loading with configurable cache size
    - Automatic memory cleanup and garbage collection
    - LRU eviction when cache is full
    - Type-safe pipeline handling
    - Device placement management
    """
    
    def __init__(self, cache_size: int = 2) -> None:
        """
        Initialize pipeline manager
        
        Args:
            cache_size: Maximum number of pipelines to keep in memory
        """
        self._pipelines: Dict[str, P] = {}
        self._cache_size = max(1, cache_size)  # Ensure at least 1
        self._access_order: List[str] = []
        
    def get_or_load(
        self, 
        pipeline_id: str, 
        loader_func: Callable[[], P],
        force_reload: bool = False
    ) -> P:
        """
        Get cached pipeline or load new one with type safety
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            loader_func: Function that creates and returns the pipeline
            force_reload: Whether to force reloading even if cached
            
        Returns:
            The loaded pipeline
            
        Raises:
            PipelineLoadError: If pipeline loading fails
        """
        if pipeline_id in self._pipelines and not force_reload:
            self._update_access_order(pipeline_id)
            return self._pipelines[pipeline_id]
            
        # Load new pipeline
        try:
            pipeline = loader_func()
            self._add_pipeline(pipeline_id, pipeline)
            return pipeline
        except Exception as e:
            raise PipelineLoadError(f"Failed to load pipeline '{pipeline_id}': {e}") from e
    
    def unload(self, pipeline_id: str) -> bool:
        """
        Safely unload specific pipeline
        
        Args:
            pipeline_id: ID of pipeline to unload
            
        Returns:
            True if successfully unloaded, False if pipeline not found
        """
        if pipeline_id not in self._pipelines:
            return False
            
        try:
            pipeline = self._pipelines.pop(pipeline_id)
            del pipeline
            self._cleanup_memory()
            self._access_order.remove(pipeline_id)
            return True
        except Exception as e:
            print(f"Warning: Error unloading pipeline {pipeline_id}: {e}")
            return False
    
    def unload_all(self) -> None:
        """Unload all cached pipelines"""
        pipeline_ids = list(self._pipelines.keys())
        for pipeline_id in pipeline_ids:
            self.unload(pipeline_id)
    
    def is_loaded(self, pipeline_id: str) -> bool:
        """Check if pipeline is currently loaded"""
        return pipeline_id in self._pipelines
    
    def get_loaded_pipelines(self) -> List[str]:
        """Get list of currently loaded pipeline IDs"""
        return list(self._pipelines.keys())
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory usage information"""
        return {
            "loaded_pipelines": len(self._pipelines),
            "cache_size": self._cache_size,
            "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        }
    
    @contextmanager
    def temporary_pipeline(self, pipeline_id: str, loader_func: Callable[[], P]):
        """
        Context manager for temporary pipeline usage
        
        The pipeline will be automatically unloaded when exiting the context,
        regardless of whether it was already cached.
        """
        was_loaded = self.is_loaded(pipeline_id)
        try:
            pipeline = self.get_or_load(pipeline_id, loader_func)
            yield pipeline
        finally:
            if not was_loaded:
                self.unload(pipeline_id)
    
    def _add_pipeline(self, pipeline_id: str, pipeline: P) -> None:
        """Add pipeline with LRU eviction if needed"""
        # If cache is full, evict least recently used
        if len(self._pipelines) >= self._cache_size and pipeline_id not in self._pipelines:
            oldest_id = self._access_order[0]
            self.unload(oldest_id)
            
        self._pipelines[pipeline_id] = pipeline
        self._update_access_order(pipeline_id)
    
    def _update_access_order(self, pipeline_id: str) -> None:
        """Update LRU order"""
        if pipeline_id in self._access_order:
            self._access_order.remove(pipeline_id)
        self._access_order.append(pipeline_id)
    
    @staticmethod
    def _cleanup_memory() -> None:
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self) -> None:
        """Cleanup on destruction"""
        try:
            self.unload_all()
        except Exception:
            pass  # Avoid errors during cleanup