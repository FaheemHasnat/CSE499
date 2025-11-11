"""
Plant treatment database handler
"""
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDatabase:
    """
    Handler for plant treatment database
    """
    
    def __init__(self, database_path: str = None):
        """
        Initialize plant database
        
        Args:
            database_path: Path to CSV database file
        """
        self.database_path = database_path
        self.data = None
        
        if database_path and Path(database_path).exists():
            self.load_database()
    
    def load_database(self):
        """
        Load plant treatment database from CSV
        """
        try:
            self.data = pd.read_csv(self.database_path)
            logger.info(f"Loaded database with {len(self.data)} entries")
            logger.info(f"Columns: {self.data.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            raise
    
    def search_by_condition(self, condition: str) -> pd.DataFrame:
        """
        Search for plant treatments by skin condition
        
        Args:
            condition: Skin condition name
            
        Returns:
            DataFrame with matching treatments
        """
        if self.data is None:
            logger.warning("Database not loaded")
            return pd.DataFrame()
        
        # Search in relevant columns (adjust based on your data structure)
        mask = self.data.astype(str).apply(
            lambda row: row.str.contains(condition, case=False, na=False).any(),
            axis=1
        )
        
        return self.data[mask]
    
    def search_by_plant(self, plant_name: str) -> pd.DataFrame:
        """
        Search for information about a specific plant
        
        Args:
            plant_name: Name of the plant
            
        Returns:
            DataFrame with matching plant information
        """
        if self.data is None:
            logger.warning("Database not loaded")
            return pd.DataFrame()
        
        mask = self.data.astype(str).apply(
            lambda row: row.str.contains(plant_name, case=False, na=False).any(),
            axis=1
        )
        
        return self.data[mask]
    
    def get_treatment_details(self, plant_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a plant treatment
        
        Args:
            plant_name: Name of the plant
            
        Returns:
            Dictionary with treatment details
        """
        results = self.search_by_plant(plant_name)
        
        if results.empty:
            return {"error": f"No information found for {plant_name}"}
        
        # Return first matching result as dictionary
        return results.iloc[0].to_dict()
    
    def get_all_plants(self) -> List[str]:
        """
        Get list of all plants in database
        
        Returns:
            List of plant names
        """
        if self.data is None:
            return []
        
        # This assumes there's a column with plant names
        # Adjust based on your actual data structure
        plant_columns = [col for col in self.data.columns if 'plant' in col.lower() or 'herb' in col.lower()]
        
        if plant_columns:
            return self.data[plant_columns[0]].unique().tolist()
        
        return []
