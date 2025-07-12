"""
LoRA adapter management with type safety

This module provides the LoRAAdapter class for managing LoRA weights
loading, application, and switching with comprehensive validation.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

from ..types import LoRAConfig, PipelineProtocol, ValidationError, PipelineLoadError

class LoRAAdapter:
    """
    Type-safe LoRA adapter management
    
    Features:
    - Download and cache LoRA weights from Hugging Face Hub
    - Apply LoRA to pipelines with weight configuration
    - Support for multiple adapters with different weights
    - Validation of LoRA configurations
    - Easy switching between adapter configurations
    """
    
    def __init__(self, config: LoRAConfig) -> None:
        """
        Initialize LoRA adapter
        
        Args:
            config: LoRA configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = self._validate_config(config)
        self.adapter_path: Optional[str] = None
        self.is_applied = False
        
    def download(self, cache_dir: Optional[Path] = None) -> str:
        """
        Download LoRA weights from Hugging Face Hub
        
        Args:
            cache_dir: Optional cache directory
            
        Returns:
            Path to downloaded LoRA file
            
        Raises:
            PipelineLoadError: If download fails
        """
        try:
            self.adapter_path = hf_hub_download(
                repo_id=self.config["repo_id"],
                filename=self.config["filename"],
                subfolder=self.config.get("subfolder"),
                cache_dir=cache_dir
            )
            return self.adapter_path
        except Exception as e:
            raise PipelineLoadError(
                f"Failed to download LoRA {self.config['adapter_name']} "
                f"from {self.config['repo_id']}/{self.config['filename']}: {e}"
            ) from e
    
    def apply_to_pipeline(self, pipeline: PipelineProtocol) -> None:
        """
        Apply LoRA to pipeline
        
        Args:
            pipeline: Pipeline to apply LoRA to
            
        Raises:
            PipelineLoadError: If application fails
        """
        if not self.adapter_path:
            self.download()
            
        try:
            # Load LoRA weights
            pipeline.load_lora_weights(
                self.adapter_path, 
                adapter_name=self.config["adapter_name"]
            )
            
            # Set adapter weight
            weight = self.config.get("weight", 1.0)
            pipeline.set_adapters(
                [self.config["adapter_name"]], 
                adapter_weights=[weight]
            )
            
            self.is_applied = True
            
        except Exception as e:
            raise PipelineLoadError(
                f"Failed to apply LoRA {self.config['adapter_name']} to pipeline: {e}"
            ) from e
    
    def fuse_to_pipeline(self, pipeline: PipelineProtocol) -> None:
        """
        Fuse LoRA weights into pipeline for efficiency
        
        Args:
            pipeline: Pipeline to fuse LoRA into
            
        Raises:
            PipelineLoadError: If fusing fails
        """
        if not self.is_applied:
            raise ValidationError("LoRA must be applied before fusing")
            
        try:
            pipeline.fuse_lora()
        except Exception as e:
            raise PipelineLoadError(f"Failed to fuse LoRA {self.config['adapter_name']}: {e}") from e
    
    @staticmethod
    def _validate_config(config: LoRAConfig) -> LoRAConfig:
        """
        Validate LoRA configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        # Check required fields
        required_fields = ["repo_id", "filename", "adapter_name"]
        for field in required_fields:
            if field not in config or not config[field]:
                raise ValidationError(f"LoRA config missing required field: {field}")
        
        # Validate repo_id format
        repo_id = config["repo_id"]
        if "/" not in repo_id:
            raise ValidationError(f"Invalid repo_id format: {repo_id}. Should be 'username/repo'")
        
        # Validate filename
        filename = config["filename"]
        if not filename.endswith(('.safetensors', '.bin', '.pt', '.pth')):
            raise ValidationError(f"Invalid LoRA filename: {filename}. Should end with .safetensors, .bin, .pt, or .pth")
        
        # Validate adapter name
        adapter_name = config["adapter_name"]
        if not adapter_name.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(f"Invalid adapter name: {adapter_name}. Should contain only alphanumeric characters, hyphens, and underscores")
        
        # Validate weight if provided
        if "weight" in config:
            weight = config["weight"]
            if not isinstance(weight, (int, float)):
                raise ValidationError(f"LoRA weight must be a number, got {type(weight)}")
            if weight < 0 or weight > 2:
                raise ValidationError(f"LoRA weight should be between 0 and 2, got {weight}")
        
        return config
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this LoRA adapter"""
        return {
            "adapter_name": self.config["adapter_name"],
            "repo_id": self.config["repo_id"], 
            "filename": self.config["filename"],
            "weight": self.config.get("weight", 1.0),
            "subfolder": self.config.get("subfolder"),
            "is_downloaded": self.adapter_path is not None,
            "is_applied": self.is_applied,
            "adapter_path": self.adapter_path
        }

class LoRAManager:
    """
    Manage multiple LoRA adapters
    
    Features:
    - Load and manage multiple LoRA adapters
    - Apply adapter configurations to pipelines
    - Switch between different adapter combinations
    - Validate adapter compatibility
    """
    
    def __init__(self) -> None:
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.active_adapters: List[str] = []
    
    def add_adapter(self, adapter: LoRAAdapter) -> None:
        """
        Add LoRA adapter to manager
        
        Args:
            adapter: LoRAAdapter instance to add
        """
        adapter_name = adapter.config["adapter_name"]
        self.adapters[adapter_name] = adapter
    
    def add_adapter_from_config(self, config: LoRAConfig) -> LoRAAdapter:
        """
        Create and add LoRA adapter from configuration
        
        Args:
            config: LoRA configuration
            
        Returns:
            Created LoRAAdapter instance
        """
        adapter = LoRAAdapter(config)
        self.add_adapter(adapter)
        return adapter
    
    def apply_adapters(
        self, 
        pipeline: PipelineProtocol, 
        adapter_names: List[str],
        weights: Optional[List[float]] = None
    ) -> None:
        """
        Apply multiple adapters to pipeline
        
        Args:
            pipeline: Pipeline to apply adapters to
            adapter_names: List of adapter names to apply
            weights: Optional list of weights for each adapter
            
        Raises:
            ValidationError: If adapter names or weights are invalid
        """
        if not adapter_names:
            return
            
        # Validate adapter names
        for name in adapter_names:
            if name not in self.adapters:
                raise ValidationError(f"Unknown adapter: {name}")
        
        # Validate weights
        if weights is not None:
            if len(weights) != len(adapter_names):
                raise ValidationError(
                    f"Number of weights ({len(weights)}) must match number of adapters ({len(adapter_names)})"
                )
        else:
            weights = [self.adapters[name].config.get("weight", 1.0) for name in adapter_names]
        
        # Download adapters if needed
        for name in adapter_names:
            adapter = self.adapters[name]
            if not adapter.adapter_path:
                adapter.download()
        
        # Apply all adapters
        for name in adapter_names:
            adapter = self.adapters[name]
            if not adapter.is_applied:
                adapter.apply_to_pipeline(pipeline)
        
        # Set combined adapter configuration
        try:
            pipeline.set_adapters(adapter_names, adapter_weights=weights)
            self.active_adapters = adapter_names.copy()
        except Exception as e:
            raise PipelineLoadError(f"Failed to set adapter configuration: {e}") from e
    
    def get_adapter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all managed adapters"""
        return {name: adapter.get_info() for name, adapter in self.adapters.items()}
    
    def get_active_adapters(self) -> List[str]:
        """Get list of currently active adapter names"""
        return self.active_adapters.copy()
    
    def clear_adapters(self) -> None:
        """Clear all adapters"""
        self.adapters.clear()
        self.active_adapters.clear()

# Predefined popular LoRA configurations
POPULAR_LORAS = {
    "causvid": LoRAConfig(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        adapter_name="causvid_lora",
        weight=0.95
    ),
    # Add more popular LoRAs here as they become available
}