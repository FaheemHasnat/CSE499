"""
Configuration file for Skin Disease - Plant Treatment VLM Project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"

# Model directories
MODEL_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Output directories
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PREPROCESSED_DATA_DIR, 
                  MODEL_DIR, CHECKPOINT_DIR, OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "llava-hf/llava-1.5-7b-hf"),
    "device": os.getenv("DEVICE", "cuda"),
    "max_length": int(os.getenv("MAX_LENGTH", "512")),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "load_in_8bit": True,  # For memory efficiency
    "torch_dtype": "float16",
}

# Alternative VLM models to consider
AVAILABLE_MODELS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-13b": "llava-hf/llava-1.5-13b-hf",
    "blip2": "Salesforce/blip2-opt-2.7b",
    "blip2-flan": "Salesforce/blip2-flan-t5-xl",
    "instructblip": "Salesforce/instructblip-vicuna-7b",
}

# Image preprocessing configuration
IMAGE_CONFIG = {
    "image_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"],
}

# Skin disease categories (can be expanded)
SKIN_DISEASE_CATEGORIES = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "Dermatitis",
    "Rosacea",
    "Fungal Infection",
    "Bacterial Infection",
    "Allergic Reaction",
    "Dry Skin",
    "Oily Skin",
    "Pigmentation",
    "Other"
]

# Plant treatment database (will be loaded from CSV)
PLANT_DATABASE_PATH = PREPROCESSED_DATA_DIR / "skincon_preprocessed.csv"

# Prompts for VLM
SYSTEM_PROMPT = """You are an expert dermatologist and herbalist AI assistant. 
Your role is to analyze skin condition images and recommend plant-based treatments.
Provide detailed, evidence-based recommendations including:
1. Skin condition identification
2. Recommended plants/herbs for treatment
3. Method of application (topical/consumption)
4. Precautions and contraindications
5. Expected timeline for improvement
"""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this skin condition image and provide:
1. Identified skin condition
2. Severity level (mild/moderate/severe)
3. Key visible symptoms
4. Recommended plant-based treatments
5. Application method
6. Precautions

Image analysis:"""

# API Configuration
API_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000")),
    "debug": os.getenv("DEBUG", "True").lower() == "true",
}

# API Keys (if using commercial APIs)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
