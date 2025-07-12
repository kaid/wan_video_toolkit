"""
Wan Video Toolkit - A comprehensive toolkit for Wan video generation models

This package provides modular, type-safe components for building video generation
applications using Wan models (I2V, FLF2V, T2V).

Main Components:
- Core modules: PipelineManager, ImageProcessor, GenerationManager
- Model implementations: WanFLF2V, WanI2V
- Adapter support: LoRA, ControlNet (future)
- Type safety: Comprehensive type annotations throughout
"""

from .version import __version__
from .models import WanFLF2VModel
from .core import PipelineManager, ImageProcessor, GenerationManager
from .adapters import LoRAAdapter
from .types import ModelConfig, GenerationParams, LoRAConfig

__all__ = [
    "__version__",
    "WanFLF2VModel", 
    "PipelineManager",
    "ImageProcessor",
    "GenerationManager",
    "LoRAAdapter",
    "ModelConfig",
    "GenerationParams", 
    "LoRAConfig",
]