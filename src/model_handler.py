"""
VLM Model Handler for Skin Disease Analysis
"""
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMModelHandler:
    """
    Handler for Vision-Language Model to analyze skin conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VLM model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.get("model_name")
        
        logger.info(f"Initializing model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        
    def load_model(self):
        """
        Load the VLM model and processor
        """
        try:
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with optimization
            logger.info("Loading model...")
            if self.config.get("load_in_8bit", False):
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_image(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Analyze a skin condition image
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if self.model is None:
            self.load_model()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Default prompt if none provided
        if prompt is None:
            prompt = """Analyze this skin condition image. Identify:
1. The type of skin condition
2. Severity level
3. Visible symptoms
4. Recommended plant-based treatments (herbs, plants)
5. How to apply/consume the treatment
6. Any precautions"""
        
        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.get("max_length", 512),
                temperature=self.config.get("temperature", 0.7),
                do_sample=True,
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "analysis": response,
            "image_path": image_path,
            "model_used": self.model_name,
        }
    
    def batch_analyze(self, image_paths: list, prompt: str = None) -> list:
        """
        Analyze multiple images
        
        Args:
            image_paths: List of image paths
            prompt: Custom prompt for analysis
            
        Returns:
            List of analysis results
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.analyze_image(img_path, prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {img_path}: {e}")
                results.append({"error": str(e), "image_path": img_path})
        
        return results
