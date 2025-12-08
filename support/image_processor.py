"""
Image preprocessing and handling utilities
"""
from PIL import Image
import cv2
import numpy as np
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image preprocessing and augmentation for skin disease images
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image processor
        
        Args:
            config: Image configuration dictionary
        """
        self.config = config
        self.image_size = config.get("image_size", (224, 224))
        self.supported_formats = config.get("supported_formats", [".jpg", ".jpeg", ".png"])
        
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded image: {image_path}, Size: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image object
            size: Target size (width, height)
            
        Returns:
            Resized PIL Image
        """
        if size is None:
            size = self.image_size
        
        # Maintain aspect ratio
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Create a new image with padding if needed
        new_image = Image.new("RGB", size, (255, 255, 255))
        paste_position = (
            (size[0] - image.size[0]) // 2,
            (size[1] - image.size[1]) // 2
        )
        new_image.paste(image, paste_position)
        
        return new_image
    
    def preprocess_for_analysis(self, image_path: str) -> Image.Image:
        """
        Preprocess image for VLM analysis
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed PIL Image
        """
        image = self.load_image(image_path)
        
        # Enhance image quality
        image = self.enhance_image(image)
        
        return image
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better analysis
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced)
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate if image is suitable for analysis
        
        Args:
            image_path: Path to image file
            
        Returns:
            Boolean indicating if image is valid
        """
        try:
            from pathlib import Path
            
            # Check file extension
            if not any(image_path.lower().endswith(fmt) for fmt in self.supported_formats):
                logger.warning(f"Unsupported format: {image_path}")
                return False
            
            # Check if file exists
            if not Path(image_path).exists():
                logger.warning(f"File not found: {image_path}")
                return False
            
            # Try to load image
            image = Image.open(image_path)
            
            # Check image size
            if image.size[0] < 50 or image.size[1] < 50:
                logger.warning(f"Image too small: {image.size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False
