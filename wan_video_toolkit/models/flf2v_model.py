"""
Wan FLF2V model implementation using modular components

This module provides the WanFLF2VModel class that composes core components
to create a complete FLF2V (First+Last Frame to Video) generation system.
"""

from typing import Optional, Dict, Any, Tuple, Union
import time
from pathlib import Path
from PIL import Image
import torch
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from transformers import CLIPVisionModel

from ..core.pipeline_manager import PipelineManager
from ..core.image_processor import ImageProcessor
from ..core.generation_manager import GenerationManager
from ..adapters.lora_adapter import LoRAManager, LoRAAdapter
from ..types import (
    ModelConfig,
    GenerationParams,
    VideoPath,
    ImageInput,
    DeviceType,
    ModelType,
    DeviceStrategy,
    LoRAConfig,
    ValidationError,
    GenerationError,
    GenerationResult,
)


class WanFLF2VModel:
    """
    Type-safe FLF2V model implementation using modular components

    Features:
    - Composition-based architecture using core modules
    - Full type safety with comprehensive validation
    - Automatic pipeline management and caching
    - Built-in LoRA support with easy configuration
    - Detailed generation metadata and timing
    - Memory-efficient with proper cleanup
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        device: Optional[DeviceType] = None,
        cache_dir: Optional[Path] = None,
        pipeline_cache_size: int = 1,
    ) -> None:
        """
        Initialize FLF2V model

        Args:
            config: Model configuration (uses default if None)
            device: Device for computation (auto-detected if None)
            cache_dir: Cache directory for models and adapters
            pipeline_cache_size: Number of pipelines to keep in memory
        """
        self.config = config or self._default_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir

        # Core managers - composition over inheritance
        self.pipeline_manager: PipelineManager[WanImageToVideoPipeline] = (
            PipelineManager(cache_size=pipeline_cache_size)
        )
        self.image_processor = ImageProcessor()
        self.generation_manager = GenerationManager()
        self.lora_manager = LoRAManager()

        # Internal state
        self._pipeline: Optional[WanImageToVideoPipeline] = None
        self._is_loaded = False

        # Load default LoRAs if specified
        self._load_default_loras()
        
        # Load pipeline immediately at module level (ZeroGPU compatible)
        self.load()

    def load(self, force_reload: bool = False) -> None:
        """
        Load the FLF2V pipeline into memory

        Args:
            force_reload: Whether to force reloading even if cached

        Raises:
            PipelineLoadError: If loading fails
        """

        def _loader() -> WanImageToVideoPipeline:
            return self._create_pipeline()

        self._pipeline = self.pipeline_manager.get_or_load(
            "flf2v", _loader, force_reload
        )
        self._is_loaded = True

    def generate(
        self,
        first_frame: ImageInput,
        last_frame: ImageInput,
        prompt: str,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate video from first and last frames

        Args:
            first_frame: First frame image
            last_frame: Last frame image
            prompt: Text prompt describing the desired motion
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with video path, seed, and metadata

        Raises:
            ValidationError: If inputs are invalid
            GenerationError: If generation fails
        """
        start_time = time.time()

        try:
            # Validate and process inputs
            first_img = self.image_processor.validate_image(first_frame)
            last_img = self.image_processor.validate_image(last_frame)

            if not prompt.strip():
                raise ValidationError("Prompt cannot be empty")

            # Load pipeline if needed
            if not self._is_loaded:
                self.load()

            assert self._pipeline is not None  # Type narrowing

            # Validate and prepare parameters
            params = self.generation_manager.validate_parameters(
                prompt=prompt, **kwargs
            )

            # Process images with consistent dimensions
            first_resized, last_resized, target_h, target_w = (
                self.image_processor.prepare_image_pair(
                    first_img,
                    last_img,
                    self.config["max_area"],
                    vae_scale_factor=8,  # Standard for diffusion models
                    patch_size=2,  # Standard for transformer patches
                )
            )

            # Prepare generation parameters
            seed = self.generation_manager.prepare_seed(
                params.get("seed"), params.get("randomize_seed", True)
            )

            num_frames = self.generation_manager.calculate_frames(
                kwargs.get("duration_seconds", 2.0),
                self.config["fps"],
                self.config["min_frames"],
                self.config["max_frames"],
            )

            # Create generator
            generator = self.generation_manager.create_generator(seed, self.device)

            # Generate video
            with torch.inference_mode():
                result = self._pipeline(
                    image=first_resized,
                    last_image=last_resized,
                    prompt=prompt,
                    negative_prompt=params.get("negative_prompt"),
                    height=target_h,
                    width=target_w,
                    num_frames=num_frames,
                    num_inference_steps=params.get("steps", 30),
                    guidance_scale=params.get("guidance_scale", 5.5),
                    generator=generator,
                )

            # Export video
            video_path = self.generation_manager.export_video(
                result.frames[0],
                fps=self.config["fps"],
                output_path=kwargs.get("output_path"),
            )

            generation_time = time.time() - start_time

            # Create comprehensive result
            generation_result: GenerationResult = {
                "video_path": video_path,
                "seed": seed,
                "generation_time": generation_time,
                "metadata": self.generation_manager.create_generation_metadata(
                    params, generation_time, seed, self._get_model_info()
                ),
            }

            return generation_result

        except Exception as e:
            generation_time = time.time() - start_time
            if isinstance(e, (ValidationError, GenerationError)):
                raise
            else:
                raise GenerationError(f"Unexpected error during generation: {e}") from e

    def add_lora(self, lora_config: LoRAConfig) -> LoRAAdapter:
        """
        Add LoRA adapter to model

        Args:
            lora_config: LoRA configuration

        Returns:
            Created LoRAAdapter instance
        """
        adapter = self.lora_manager.add_adapter_from_config(lora_config)

        # Apply to pipeline if already loaded
        if self._is_loaded and self._pipeline is not None:
            adapter.apply_to_pipeline(self._pipeline)

        return adapter

    def set_lora_weights(self, adapter_weights: Dict[str, float]) -> None:
        """
        Set weights for loaded LoRA adapters

        Args:
            adapter_weights: Dictionary mapping adapter names to weights
        """
        if not self._is_loaded:
            raise ValidationError("Pipeline must be loaded before setting LoRA weights")

        adapter_names = list(adapter_weights.keys())
        weights = list(adapter_weights.values())

        self.lora_manager.apply_adapters(self._pipeline, adapter_names, weights)

    def unload(self) -> None:
        """Unload pipeline and free memory"""
        if self._is_loaded:
            self.pipeline_manager.unload("flf2v")
            self._pipeline = None
            self._is_loaded = False

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        return {
            "pipeline_manager": self.pipeline_manager.get_memory_info(),
            "is_loaded": self._is_loaded,
            "active_loras": self.lora_manager.get_active_adapters(),
            "device": str(self.device),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and status information"""
        return {
            "model_id": self.config["model_id"],
            "model_type": self.config["model_type"],
            "is_loaded": self._is_loaded,
            "config": dict(self.config),
            "lora_info": self.lora_manager.get_adapter_info(),
            "memory_info": self.get_memory_info(),
        }

    def _create_pipeline(self) -> WanImageToVideoPipeline:
        """Create and configure the FLF2V pipeline"""
        try:
            # Load components following ZeroGPU best practices
            image_encoder = CLIPVisionModel.from_pretrained(
                self.config["model_id"],
                subfolder="image_encoder",
                torch_dtype=torch.float32,  # Always float32 for image encoder
                cache_dir=self.cache_dir,
            )

            vae = AutoencoderKLWan.from_pretrained(
                self.config["model_id"],
                subfolder="vae",
                torch_dtype=getattr(torch, self.config["dtype"]),
                cache_dir=self.cache_dir,
            )

            # Create pipeline with device strategy
            pipeline_kwargs = {
                "image_encoder": image_encoder,
                "vae": vae,
                "torch_dtype": getattr(torch, self.config["dtype"]),
                "cache_dir": self.cache_dir,
            }

            # Add device mapping if specified
            if self.config["device_strategy"] == DeviceStrategy.BALANCED:
                pipeline_kwargs["device_map"] = "balanced"
                pipeline_kwargs["use_fast"] = self.config.get("use_fast", True)

            pipeline = WanImageToVideoPipeline.from_pretrained(
                self.config["model_id"], **pipeline_kwargs
            )

            # Move to device if not using balanced mapping
            if self.config["device_strategy"] != DeviceStrategy.BALANCED:
                pipeline = pipeline.to(self.device)

            # Apply loaded LoRAs
            for adapter_name, adapter in self.lora_manager.adapters.items():
                if not adapter.is_applied:
                    adapter.apply_to_pipeline(pipeline)

            return pipeline
            
        except Exception as e:
            raise GenerationError(f"Failed to create FLF2V pipeline: {e}") from e

    def _load_default_loras(self) -> None:
        """Load default LoRA adapters from configuration"""
        for lora_config in self.config.get("default_loras", []):
            try:
                self.add_lora(lora_config)
            except Exception as e:
                print(
                    f"Warning: Failed to load default LoRA {lora_config.get('adapter_name')}: {e}"
                )

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information for metadata"""
        return {
            "model_id": self.config["model_id"],
            "model_type": self.config["model_type"].value,
            "dtype": self.config["dtype"],
            "device_strategy": self.config["device_strategy"].value,
            "active_loras": self.lora_manager.get_active_adapters(),
        }

    @staticmethod
    def _default_config() -> ModelConfig:
        """Default configuration for FLF2V model"""
        return ModelConfig(
            model_id="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            model_type=ModelType.FLF2V,
            dtype="float16",
            device_strategy=DeviceStrategy.BALANCED,
            max_area=1280 * 720,
            default_height=720,
            default_width=1280,
            min_frames=8,
            max_frames=81,
            fps=24,
            use_fast=True,
            default_loras=[],
        )

    def __del__(self) -> None:
        """Cleanup on destruction"""
        try:
            self.unload()
        except Exception:
            pass  # Avoid errors during cleanup

