"""
Initialize src module
"""
from pathlib import Path

# Module version
__version__ = "0.1.0"

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Make imports easier
__all__ = [
    'model_handler',
    'image_processor',
    'plant_database',
    'utils'
]
