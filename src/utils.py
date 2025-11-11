"""
Utility functions for the VLM project
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_analysis_result(result: Dict[str, Any], output_dir: str = "./outputs/results"):
    """
    Save analysis result to JSON file
    
    Args:
        result: Analysis result dictionary
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved result to {filepath}")
    return str(filepath)


def parse_vlm_response(response: str) -> Dict[str, Any]:
    """
    Parse VLM response into structured format
    
    Args:
        response: Raw VLM response text
        
    Returns:
        Structured dictionary with parsed information
    """
    # This is a basic parser - can be enhanced with NLP
    parsed = {
        "raw_response": response,
        "condition": "",
        "severity": "",
        "symptoms": [],
        "plant_treatments": [],
        "application_method": "",
        "precautions": "",
    }
    
    # Simple keyword-based parsing
    lines = response.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        if 'condition' in line_lower or 'diagnosis' in line_lower:
            parsed["condition"] = line.strip()
        elif 'severity' in line_lower:
            parsed["severity"] = line.strip()
        elif 'plant' in line_lower or 'herb' in line_lower or 'treatment' in line_lower:
            parsed["plant_treatments"].append(line.strip())
        elif 'application' in line_lower or 'method' in line_lower:
            parsed["application_method"] = line.strip()
        elif 'precaution' in line_lower or 'warning' in line_lower:
            parsed["precautions"] = line.strip()
    
    return parsed


def format_treatment_recommendation(analysis: Dict[str, Any], plant_info: Dict[str, Any] = None) -> str:
    """
    Format treatment recommendation for display
    
    Args:
        analysis: VLM analysis result
        plant_info: Additional plant information from database
        
    Returns:
        Formatted recommendation text
    """
    formatted = f"""
{'='*60}
SKIN CONDITION ANALYSIS & PLANT TREATMENT RECOMMENDATION
{'='*60}

ANALYSIS:
{analysis.get('analysis', 'No analysis available')}

"""
    
    if plant_info:
        formatted += f"""
PLANT DATABASE INFORMATION:
{'-'*60}
{json.dumps(plant_info, indent=2)}
"""
    
    formatted += f"""
{'='*60}
Analyzed with: {analysis.get('model_used', 'Unknown model')}
Image: {analysis.get('image_path', 'Unknown')}
{'='*60}
"""
    
    return formatted


def check_gpu_availability():
    """
    Check if GPU is available for model inference
    
    Returns:
        Dictionary with GPU information
    """
    import torch
    
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
    }
    
    if gpu_info["cuda_available"]:
        gpu_info["current_device"] = torch.cuda.current_device()
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
    
    return gpu_info
