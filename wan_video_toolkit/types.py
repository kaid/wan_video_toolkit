"""
Comprehensive type definitions for wan-video-toolkit

This module defines all the types used throughout the toolkit to ensure
type safety and provide clear API contracts.
"""

from typing import (
    Dict, List, Optional, Union, Tuple, Callable, Any, 
    Protocol, TypeVar, Generic, Literal
)
from typing_extensions import TypedDict, NotRequired
from pathlib import Path
from PIL import Image
import torch
from enum import Enum

# Core type aliases
VideoPath = Union[str, Path]
ImageInput = Union[Image.Image, str, Path]
DeviceType = Union[str, torch.device]
DType = torch.dtype

# Enums for better type safety
class ModelType(str, Enum):
    """Supported model types"""
    I2V = "i2v"
    FLF2V = "flf2v" 
    T2V = "t2v"

class DeviceStrategy(str, Enum):
    """Device placement strategies"""
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    BALANCED = "balanced"

# TypedDict for configuration validation
class GenerationParams(TypedDict):
    """Parameters for video generation"""
    prompt: str
    negative_prompt: NotRequired[str]
    height: NotRequired[int]
    width: NotRequired[int]
    num_frames: NotRequired[int]
    steps: NotRequired[int]
    guidance_scale: NotRequired[float]
    seed: NotRequired[int]
    randomize_seed: NotRequired[bool]

class LoRAConfig(TypedDict):
    """Configuration for LoRA adapters"""
    repo_id: str
    filename: str
    adapter_name: str
    weight: NotRequired[float]
    subfolder: NotRequired[str]

class ModelConfig(TypedDict):
    """Configuration for model instances"""
    model_id: str
    model_type: ModelType
    dtype: str  # "float16", "bfloat16", etc.
    max_area: int
    default_height: int
    default_width: int
    min_frames: int
    max_frames: int
    fps: int
    use_fast: NotRequired[bool]
    default_loras: NotRequired[List[LoRAConfig]]

# Protocol for pipeline-like objects (duck typing with type safety)
class PipelineProtocol(Protocol):
    """Protocol for diffusers pipeline objects"""
    
    def __call__(self, **kwargs: Any) -> Any: 
        """Generate content with the pipeline"""
        ...
    
    def to(self, device: DeviceType) -> 'PipelineProtocol': 
        """Move pipeline to device"""
        ...
    
    def load_lora_weights(self, path: str, adapter_name: str) -> None: 
        """Load LoRA weights"""
        ...
    
    def set_adapters(self, adapter_names: List[str], adapter_weights: List[float]) -> None:
        """Set adapter configuration"""
        ...
    
    def fuse_lora(self) -> None:
        """Fuse LoRA weights for efficiency"""
        ...

# Generic type variables
T = TypeVar('T')
P = TypeVar('P', bound=PipelineProtocol)

# Result types
class GenerationResult(TypedDict):
    """Result from video generation"""
    video_path: str
    seed: int
    generation_time: NotRequired[float]
    metadata: NotRequired[Dict[str, Any]]

# Error types for better error handling
class WanVideoToolkitError(Exception):
    """Base exception for toolkit errors"""
    pass

class PipelineLoadError(WanVideoToolkitError):
    """Error loading pipeline"""
    pass

class GenerationError(WanVideoToolkitError):
    """Error during generation"""
    pass

class ValidationError(WanVideoToolkitError):
    """Parameter validation error"""
    pass