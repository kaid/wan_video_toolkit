"""
Type-safe image processing utilities

This module provides comprehensive image processing operations needed for
video generation, including resizing, aspect ratio calculations, and validation.
"""

from typing import Tuple, Union, Optional, List
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as TF

from ..types import ImageInput, ValidationError

class ImageProcessor:
    """
    Type-safe image processing operations for video generation
    
    Features:
    - Aspect ratio preserving resize operations
    - Multi-step image processing pipelines
    - Dimension validation and MOD_VALUE alignment
    - Format conversion and preprocessing
    - Comprehensive input validation
    """
    
    @staticmethod
    def validate_image(image: ImageInput) -> Image.Image:
        """
        Validate and convert image input to PIL Image
        
        Args:
            image: Image input (PIL Image, file path, or Path object)
            
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValidationError: If image cannot be loaded or is invalid
        """
        if isinstance(image, (str, Path)):
            try:
                img = Image.open(image)
                # Ensure RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
            except Exception as e:
                raise ValidationError(f"Could not load image from {image}: {e}") from e
        elif isinstance(image, Image.Image):
            # Ensure RGB format
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        else:
            raise ValidationError(f"Unsupported image type: {type(image)}")
    
    @staticmethod
    def calculate_dimensions(
        image: Image.Image, 
        max_area: int, 
        mod_value: int = 32,
        min_size: int = 128,
        max_size: int = 2048
    ) -> Tuple[int, int]:
        """
        Calculate optimal dimensions preserving aspect ratio
        
        Args:
            image: Input PIL Image
            max_area: Maximum pixel area constraint
            mod_value: Dimensions must be multiples of this value
            min_size: Minimum dimension size
            max_size: Maximum dimension size
            
        Returns:
            Tuple of (height, width) as multiples of mod_value
            
        Raises:
            ValidationError: If image dimensions are invalid
        """
        orig_w, orig_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            raise ValidationError(f"Invalid image dimensions: {orig_w}x{orig_h}")
            
        if max_area <= 0:
            raise ValidationError(f"Max area must be positive, got {max_area}")
            
        if mod_value <= 0:
            raise ValidationError(f"Mod value must be positive, got {mod_value}")
            
        aspect_ratio = orig_h / orig_w
        
        # Calculate dimensions that fit within max_area
        calc_h = round(np.sqrt(max_area * aspect_ratio))
        calc_w = round(np.sqrt(max_area / aspect_ratio))
        
        # Ensure multiples of mod_value
        calc_h = max(mod_value, (calc_h // mod_value) * mod_value)
        calc_w = max(mod_value, (calc_w // mod_value) * mod_value)
        
        # Apply size constraints
        new_h = int(np.clip(calc_h, min_size, max_size))
        new_w = int(np.clip(calc_w, min_size, max_size))
        
        # Ensure still multiples of mod_value after clipping
        new_h = (new_h // mod_value) * mod_value
        new_w = (new_w // mod_value) * mod_value
        
        return new_h, new_w
    
    @staticmethod
    def aspect_resize(
        image: Image.Image, 
        max_area: int, 
        vae_scale_factor: int = 8, 
        patch_size: int = 2
    ) -> Tuple[Image.Image, int, int]:
        """
        Resize maintaining aspect ratio within area constraint
        
        Args:
            image: Input PIL Image
            max_area: Maximum pixel area
            vae_scale_factor: VAE scale factor for dimension alignment
            patch_size: Transformer patch size
            
        Returns:
            Tuple of (resized_image, height, width)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if max_area <= 0:
            raise ValidationError(f"Max area must be positive, got {max_area}")
        if vae_scale_factor <= 0:
            raise ValidationError(f"VAE scale factor must be positive, got {vae_scale_factor}")
        if patch_size <= 0:
            raise ValidationError(f"Patch size must be positive, got {patch_size}")
            
        ar = image.height / image.width
        mod = vae_scale_factor * patch_size
        
        h = int(np.sqrt(max_area * ar)) // mod * mod
        w = int(np.sqrt(max_area / ar)) // mod * mod
        
        # Ensure minimum size
        h = max(mod, h)
        w = max(mod, w)
        
        resized_image = image.resize((w, h), Image.Resampling.LANCZOS)
        return resized_image, h, w
    
    @staticmethod
    def center_crop_resize(
        image: Image.Image, 
        target_height: int, 
        target_width: int
    ) -> Image.Image:
        """
        Resize and center crop to exact dimensions
        
        Args:
            image: Input PIL Image
            target_height: Target height in pixels
            target_width: Target width in pixels
            
        Returns:
            Resized and cropped PIL Image
            
        Raises:
            ValidationError: If target dimensions are invalid
        """
        if target_height <= 0 or target_width <= 0:
            raise ValidationError(
                f"Invalid target dimensions: {target_width}x{target_height}"
            )
            
        # Calculate scale to cover target area
        ratio = max(target_width / image.width, target_height / image.height)
        new_width = round(image.width * ratio)
        new_height = round(image.height * ratio)
        
        # Resize to cover target dimensions
        resized_image = image.resize(
            (new_width, new_height), 
            Image.Resampling.LANCZOS
        )
        
        # Center crop to exact target dimensions
        cropped_image = TF.center_crop(resized_image, [target_height, target_width])
        
        return cropped_image
    
    @staticmethod
    def prepare_image_pair(
        first_image: ImageInput,
        second_image: ImageInput,
        max_area: int,
        vae_scale_factor: int = 8,
        patch_size: int = 2
    ) -> Tuple[Image.Image, Image.Image, int, int]:
        """
        Prepare a pair of images with consistent dimensions
        
        Args:
            first_image: First image input
            second_image: Second image input  
            max_area: Maximum pixel area
            vae_scale_factor: VAE scale factor
            patch_size: Transformer patch size
            
        Returns:
            Tuple of (first_processed, second_processed, height, width)
        """
        # Validate inputs
        first_img = ImageProcessor.validate_image(first_image)
        second_img = ImageProcessor.validate_image(second_image)
        
        # Resize first image with aspect ratio preservation
        first_resized, h, w = ImageProcessor.aspect_resize(
            first_img, max_area, vae_scale_factor, patch_size
        )
        
        # Resize second image to match first image dimensions
        second_resized = ImageProcessor.center_crop_resize(second_img, h, w)
        
        return first_resized, second_resized, h, w
    
    @staticmethod
    def batch_process_images(
        images: List[ImageInput],
        target_height: int,
        target_width: int
    ) -> List[Image.Image]:
        """
        Process multiple images to consistent dimensions
        
        Args:
            images: List of image inputs
            target_height: Target height for all images
            target_width: Target width for all images
            
        Returns:
            List of processed PIL Images
        """
        if not images:
            raise ValidationError("No images provided for batch processing")
            
        processed_images = []
        for i, image in enumerate(images):
            try:
                img = ImageProcessor.validate_image(image)
                processed_img = ImageProcessor.center_crop_resize(
                    img, target_height, target_width
                )
                processed_images.append(processed_img)
            except Exception as e:
                raise ValidationError(f"Failed to process image {i}: {e}") from e
                
        return processed_images
    
    @staticmethod
    def get_image_info(image: Union[Image.Image, ImageInput]) -> dict:
        """
        Get information about an image
        
        Args:
            image: Image to analyze
            
        Returns:
            Dictionary with image information
        """
        if not isinstance(image, Image.Image):
            image = ImageProcessor.validate_image(image)
            
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": len(image.tobytes()) if hasattr(image, 'tobytes') else None,
            "aspect_ratio": image.width / image.height,
            "area": image.width * image.height
        }