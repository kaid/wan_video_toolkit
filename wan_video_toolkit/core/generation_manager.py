"""
Type-safe generation workflow management

This module provides the GenerationManager class for handling video generation
parameters, validation, and export operations with comprehensive type safety.
"""

from typing import Dict, Any, Optional, List, Union
import random
import tempfile
import time
from pathlib import Path
import torch
import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from ..types import GenerationParams, VideoPath, ValidationError, GenerationError

class GenerationManager:
    """
    Type-safe generation workflow management
    
    Features:
    - Comprehensive parameter validation with helpful error messages
    - Seed management for reproducible results
    - Frame count calculation with constraints
    - Video export with format validation
    - Generation timing and metadata tracking
    """
    
    @staticmethod
    def validate_parameters(**kwargs: Any) -> GenerationParams:
        """
        Validate and normalize generation parameters
        
        Args:
            **kwargs: Raw generation parameters
            
        Returns:
            Validated GenerationParams dictionary
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        # Start with required parameters
        if 'prompt' not in kwargs or not kwargs['prompt']:
            raise ValidationError("Prompt is required and cannot be empty")
        
        params: GenerationParams = {
            'prompt': str(kwargs['prompt']).strip(),
        }
        
        # Validate optional parameters with detailed error messages
        if 'negative_prompt' in kwargs and kwargs['negative_prompt']:
            negative = str(kwargs['negative_prompt']).strip()
            if negative:
                params['negative_prompt'] = negative
        
        # Validate dimensions
        if 'height' in kwargs:
            height = int(kwargs['height'])
            if height < 64:
                raise ValidationError(f"Height must be at least 64, got {height}")
            if height > 2048:
                raise ValidationError(f"Height must be at most 2048, got {height}")
            if height % 8 != 0:
                raise ValidationError(f"Height must be divisible by 8, got {height}")
            params['height'] = height
            
        if 'width' in kwargs:
            width = int(kwargs['width'])
            if width < 64:
                raise ValidationError(f"Width must be at least 64, got {width}")
            if width > 2048:
                raise ValidationError(f"Width must be at most 2048, got {width}")
            if width % 8 != 0:
                raise ValidationError(f"Width must be divisible by 8, got {width}")
            params['width'] = width
        
        # Validate frame count
        if 'num_frames' in kwargs:
            frames = int(kwargs['num_frames'])
            if frames < 1:
                raise ValidationError(f"Number of frames must be at least 1, got {frames}")
            if frames > 200:
                raise ValidationError(f"Number of frames must be at most 200, got {frames}")
            params['num_frames'] = frames
        
        # Validate inference steps
        if 'steps' in kwargs:
            steps = int(kwargs['steps'])
            if steps < 1:
                raise ValidationError(f"Steps must be at least 1, got {steps}")
            if steps > 100:
                raise ValidationError(f"Steps must be at most 100, got {steps}")
            params['steps'] = steps
        
        # Validate guidance scale
        if 'guidance_scale' in kwargs:
            guidance = float(kwargs['guidance_scale'])
            if guidance < 0:
                raise ValidationError(f"Guidance scale must be non-negative, got {guidance}")
            if guidance > 20:
                raise ValidationError(f"Guidance scale must be at most 20, got {guidance}")
            params['guidance_scale'] = guidance
        
        # Validate seed
        if 'seed' in kwargs:
            seed = int(kwargs['seed'])
            if seed < 0:
                raise ValidationError(f"Seed must be non-negative, got {seed}")
            if seed > 2**31 - 1:
                raise ValidationError(f"Seed must be at most {2**31 - 1}, got {seed}")
            params['seed'] = seed
        
        # Validate randomize_seed
        if 'randomize_seed' in kwargs:
            params['randomize_seed'] = bool(kwargs['randomize_seed'])
        
        return params
    
    @staticmethod
    def prepare_seed(seed: Optional[int] = None, randomize: bool = True) -> int:
        """
        Handle seed generation with type safety
        
        Args:
            seed: Optional seed value
            randomize: Whether to randomize seed
            
        Returns:
            Valid seed value in range [0, 2^31-1]
        """
        if randomize or seed is None:
            return random.randint(0, 2**31 - 1)
        
        # Ensure seed is in valid range
        return max(0, min(int(seed), 2**31 - 1))
    
    @staticmethod
    def calculate_frames(
        duration_seconds: float, 
        fps: int, 
        min_frames: int = 8, 
        max_frames: int = 200
    ) -> int:
        """
        Calculate frame count with constraints
        
        Args:
            duration_seconds: Desired duration in seconds
            fps: Frames per second
            min_frames: Minimum frame count
            max_frames: Maximum frame count
            
        Returns:
            Frame count within constraints
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if duration_seconds <= 0:
            raise ValidationError(f"Duration must be positive, got {duration_seconds}")
        if fps <= 0:
            raise ValidationError(f"FPS must be positive, got {fps}")
        if min_frames < 1:
            raise ValidationError(f"Min frames must be at least 1, got {min_frames}")
        if max_frames < min_frames:
            raise ValidationError(f"Max frames ({max_frames}) must be >= min frames ({min_frames})")
            
        calculated_frames = int(round(duration_seconds * fps))
        return int(np.clip(calculated_frames, min_frames, max_frames))
    
    @staticmethod
    def export_video(
        frames: List[Image.Image], 
        fps: int = 24,
        output_path: Optional[VideoPath] = None,
        format: str = "mp4"
    ) -> str:
        """
        Export frames to video with comprehensive validation
        
        Args:
            frames: List of PIL Images to export
            fps: Frames per second
            output_path: Optional output file path
            format: Video format (currently only mp4 supported)
            
        Returns:
            Path to exported video file
            
        Raises:
            GenerationError: If export fails
            ValidationError: If parameters are invalid
        """
        if not frames:
            raise ValidationError("Cannot export video: no frames provided")
        if fps <= 0:
            raise ValidationError(f"FPS must be positive, got {fps}")
        if format.lower() != "mp4":
            raise ValidationError(f"Unsupported format: {format}. Only 'mp4' is currently supported")
        
        # Validate all frames are Images
        for i, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                raise ValidationError(f"Frame {i} is not a PIL Image: {type(frame)}")
        
        # Generate output path if not provided
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                output_path = f.name
        else:
            output_path = str(output_path)
            
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            export_to_video(frames, output_path, fps=fps)
            
            # Verify file was created
            if not Path(output_path).exists():
                raise GenerationError(f"Video file was not created at {output_path}")
                
            return output_path
            
        except Exception as e:
            raise GenerationError(f"Failed to export video to {output_path}: {e}") from e
    
    @staticmethod
    def create_generator(seed: int, device: Union[str, torch.device] = "cuda") -> torch.Generator:
        """
        Create a torch Generator with specified seed
        
        Args:
            seed: Seed value
            device: Device for generator
            
        Returns:
            Configured torch Generator
        """
        return torch.Generator(device=device).manual_seed(seed)
    
    @staticmethod
    def estimate_generation_time(
        num_frames: int,
        steps: int,
        model_type: str = "flf2v",
        device: str = "cuda"
    ) -> float:
        """
        Estimate generation time in seconds
        
        Args:
            num_frames: Number of frames to generate
            steps: Number of inference steps
            model_type: Type of model (affects time estimation)
            device: Computing device
            
        Returns:
            Estimated time in seconds
        """
        # Base time per step (rough estimates)
        base_times = {
            "flf2v": 2.0,  # seconds per step
            "i2v": 1.5,    # seconds per step  
            "t2v": 3.0,    # seconds per step
        }
        
        base_time = base_times.get(model_type, 2.0)
        
        # Adjust for device
        device_multiplier = 1.0 if device == "cuda" else 3.0
        
        # Adjust for frame count (more frames = slightly longer per step)
        frame_multiplier = 1.0 + (num_frames - 16) * 0.01
        
        estimated_time = base_time * steps * device_multiplier * frame_multiplier
        return max(1.0, estimated_time)  # Minimum 1 second
    
    @staticmethod
    def create_generation_metadata(
        params: GenerationParams,
        generation_time: float,
        seed_used: int,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for generation
        
        Args:
            params: Generation parameters used
            generation_time: Actual generation time
            seed_used: Seed that was used
            model_info: Optional model information
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "parameters": dict(params),
            "generation_time_seconds": generation_time,
            "seed_used": seed_used,
            "timestamp": time.time(),
        }
        
        if model_info:
            metadata["model_info"] = model_info
            
        return metadata