"""
Main application file for Skin Disease - Plant Treatment VLM
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_handler import VLMModelHandler
from src.image_processor import ImageProcessor
from config import MODEL_CONFIG, IMAGE_CONFIG


def main():
    """
    Main function to run the VLM application
    """
    print("=" * 50)
    print("Skin Disease - Plant Treatment VLM System")
    print("=" * 50)
    
    # Initialize components
    print("\n[1/3] Initializing image processor...")
    image_processor = ImageProcessor(IMAGE_CONFIG)
    
    print("[2/3] Loading VLM model...")
    print(f"Model: {MODEL_CONFIG['model_name']}")
    print(f"Device: {MODEL_CONFIG['device']}")
    model_handler = VLMModelHandler(MODEL_CONFIG)
    
    print("[3/3] System ready!")
    print("\nYou can now:")
    print("- Run the Gradio web interface: python app.py")
    print("- Use the API: uvicorn api:app --reload")
    print("- Process images programmatically using model_handler.analyze_image()")
    
    return model_handler, image_processor


if __name__ == "__main__":
    model_handler, image_processor = main()
