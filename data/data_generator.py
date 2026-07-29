"""
Enterprise Data Generator with Scientific Accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
from pathlib import Path
from loguru import logger

class CropCategory(Enum):
    """Crop categories"""
    CEREAL = "Cereal"
    PULSE = "Pulse"
    OILSEED = "Oilseed"
    FIBER = "Fiber"
    VEGETABLE = "Vegetable"
    FRUIT = "Fruit"
    PLANTATION = "Plantation"
    SPICE = "Spice"
    CASH_CROP = "Cash Crop"

class Season(Enum):
    """Growing seasons"""
    KHARIF = "Kharif (June-October)"
    RABI = "Rabi (November-April)"
    ZAID = "Zaid (April-June)"
    ANNUAL = "Annual"
    PERENNIAL = "Perennial"

@dataclass
class CropSpecs:
    """Scientific specifications for each crop"""
    name: str
    category: CropCategory
    season: Season
    n_range: Tuple[float, float]
    p_range: Tuple[float, float]
    k_range: Tuple[float, float]
    temp_range: Tuple[float, float]
    humidity_range: Tuple[float, float]
    ph_range: Tuple[float, float]
    rainfall_range: Tuple[float, float]
    duration: str
    soil_type: str
    water_requirement: str
    description: str = ""
    nutrient_demand: str = "Medium"
    temperature_sensitivity: str = "Medium"
    drought_tolerance: str = "Low"
    optimal_elevation: Tuple[int, int] = (0, 1000)
    
    @property
    def feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'N': self.n_range,
            'P': self.p_range,
            'K': self.k_range,
            'temperature': self.temp_range,
            'humidity': self.humidity_range,
            'ph': self.ph_range,
            'rainfall': self.rainfall_range
        }

class DataGenerator:
    """
    Professional data generator with agricultural scientific accuracy
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.crops = self._initialize_crops()
        
    def _initialize_crops(self) -> List[CropSpecs]:
        """Initialize comprehensive crop database"""
        return [
            CropSpecs(
                name="Rice",
                category=CropCategory.CEREAL,
                season=Season.KHARIF,
                n_range=(80, 120), p_range=(40, 60), k_range=(40, 60),
                temp_range=(22, 32), humidity_range=(70, 90),
                ph_range=(5.5, 6.5), rainfall_range=(150, 250),
                duration="120-150 days",
                soil_type="Clay loam",
                water_requirement="High (1500-2000 mm)",
                description="Staple food crop, requires standing water",
                nutrient_demand="High",
                temperature_sensitivity="Low",
                drought_tolerance="Very Low"
            ),
            CropSpecs(
                name="Wheat",
                category=CropCategory.CEREAL,
                season=Season.RABI,
                n_range=(60, 100), p_range=(30, 50), k_range=(30, 50),
                temp_range=(15, 25), humidity_range=(60, 80),
                ph_range=(6.0, 7.0), rainfall_range=(50, 100),
                duration="110-130 days",
                soil_type="Loam",
                water_requirement="Medium (450-650 mm)",
                description="Winter crop, requires cool climate",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            ),
            CropSpecs(
                name="Maize",
                category=CropCategory.CEREAL,
                season=Season.KHARIF,
                n_range=(70, 110), p_range=(30, 60), k_range=(30, 60),
                temp_range=(20, 30), humidity_range=(65, 85),
                ph_range=(5.8, 7.0), rainfall_range=(80, 150),
                duration="90-110 days",
                soil_type="Well-drained loam",
                water_requirement="Medium (500-800 mm)",
                description="Versatile crop used for food and feed",
                nutrient_demand="High",
                temperature_sensitivity="Medium",
                drought_tolerance="Low"
            ),
            CropSpecs(
                name="Cotton",
                category=CropCategory.FIBER,
                season=Season.KHARIF,
                n_range=(80, 130), p_range=(40, 70), k_range=(40, 70),
                temp_range=(25, 35), humidity_range=(60, 80),
                ph_range=(6.0, 7.5), rainfall_range=(70, 120),
                duration="150-180 days",
                soil_type="Black soil",
                water_requirement="Medium (700-1200 mm)",
                description="Fiber crop, requires warm climate",
                nutrient_demand="High",
                temperature_sensitivity="High",
                drought_tolerance="High"
            ),
            CropSpecs(
                name="Sugarcane",
                category=CropCategory.CASH_CROP,
                season=Season.ANNUAL,
                n_range=(100, 150), p_range=(50, 80), k_range=(50, 80),
                temp_range=(25, 35), humidity_range=(70, 90),
                ph_range=(6.0, 7.5), rainfall_range=(150, 250),
                duration="10-12 months",
                soil_type="Loamy soil",
                water_requirement="High (1500-2500 mm)",
                description="Cash crop for sugar production",
                nutrient_demand="Very High",
                temperature_sensitivity="Medium",
                drought_tolerance="Low"
            ),
            CropSpecs(
                name="Groundnut",
                category=CropCategory.OILSEED,
                season=Season.KHARIF,
                n_range=(20, 40), p_range=(40, 70), k_range=(30, 50),
                temp_range=(20, 30), humidity_range=(60, 80),
                ph_range=(5.5, 7.0), rainfall_range=(50, 80),
                duration="110-140 days",
                soil_type="Sandy loam",
                water_requirement="Medium (500-800 mm)",
                description="Oilseed crop, enriches soil nitrogen",
                nutrient_demand="Low",
                temperature_sensitivity="Medium",
                drought_tolerance="High"
            ),
            CropSpecs(
                name="Potato",
                category=CropCategory.VEGETABLE,
                season=Season.RABI,
                n_range=(60, 100), p_range=(40, 70), k_range=(50, 80),
                temp_range=(15, 25), humidity_range=(70, 85),
                ph_range=(5.0, 6.5), rainfall_range=(60, 100),
                duration="80-100 days",
                soil_type="Loamy sand",
                water_requirement="Medium (400-600 mm)",
                description="Tuber crop, requires well-drained soil",
                nutrient_demand="High",
                temperature_sensitivity="High",
                drought_tolerance="Low"
            ),
            CropSpecs(
                name="Tomato",
                category=CropCategory.VEGETABLE,
                season=Season.RABI,
                n_range=(50, 90), p_range=(40, 70), k_range=(40, 70),
                temp_range=(20, 27), humidity_range=(65, 80),
                ph_range=(5.5, 7.0), rainfall_range=(50, 90),
                duration="60-80 days",
                soil_type="Well-drained loam",
                water_requirement="Medium (600-800 mm)",
                description="Vegetable crop, high market demand",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Low"
            ),
            CropSpecs(
                name="Coffee",
                category=CropCategory.PLANTATION,
                season=Season.PERENNIAL,
                n_range=(80, 120), p_range=(30, 50), k_range=(40, 60),
                temp_range=(18, 25), humidity_range=(70, 85),
                ph_range=(5.0, 6.0), rainfall_range=(150, 200),
                duration="3-4 years",
                soil_type="Well-drained loam",
                water_requirement="High (1500-2000 mm)",
                description="Plantation crop, requires shade",
                nutrient_demand="Medium",
                temperature_sensitivity="High",
                drought_tolerance="Low",
                optimal_elevation=(600, 2000)
            ),
            CropSpecs(
                name="Tea",
                category=CropCategory.PLANTATION,
                season=Season.PERENNIAL,
                n_range=(100, 150), p_range=(30, 50), k_range=(40, 60),
                temp_range=(18, 25), humidity_range=(80, 90),
                ph_range=(4.5, 5.5), rainfall_range=(150, 250),
                duration="50+ years",
                soil_type="Acidic loam",
                water_requirement="High (2000-2500 mm)",
                description="Plantation crop, acidic soil required",
                nutrient_demand="High",
                temperature_sensitivity="High",
                drought_tolerance="Low",
                optimal_elevation=(600, 2000)
            ),
            CropSpecs(
                name="Banana",
                category=CropCategory.FRUIT,
                season=Season.ANNUAL,
                n_range=(80, 120), p_range=(30, 60), k_range=(80, 120),
                temp_range=(25, 35), humidity_range=(75, 90),
                ph_range=(5.5, 7.0), rainfall_range=(100, 200),
                duration="10-12 months",
                soil_type="Well-drained loam",
                water_requirement="High (1500-2000 mm)",
                description="Tropical fruit, high potassium demand",
                nutrient_demand="Very High",
                temperature_sensitivity="High",
                drought_tolerance="Low"
            ),
            CropSpecs(
                name="Orange",
                category=CropCategory.FRUIT,
                season=Season.ANNUAL,
                n_range=(60, 100), p_range=(30, 60), k_range=(40, 80),
                temp_range=(20, 30), humidity_range=(60, 75),
                ph_range=(5.5, 6.5), rainfall_range=(100, 150),
                duration="6-8 months",
                soil_type="Well-drained loam",
                water_requirement="Medium (900-1200 mm)",
                description="Citrus fruit, requires well-drained soil",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            ),
            CropSpecs(
                name="Apple",
                category=CropCategory.FRUIT,
                season=Season.ANNUAL,
                n_range=(50, 90), p_range=(30, 50), k_range=(40, 70),
                temp_range=(10, 20), humidity_range=(65, 80),
                ph_range=(5.5, 6.5), rainfall_range=(80, 120),
                duration="120-150 days",
                soil_type="Well-drained loam",
                water_requirement="Medium (700-1000 mm)",
                description="Temperate fruit, requires chilling",
                nutrient_demand="Medium",
                temperature_sensitivity="High",
                drought_tolerance="Low",
                optimal_elevation=(1500, 2700)
            ),
            CropSpecs(
                name="Barley",
                category=CropCategory.CEREAL,
                season=Season.RABI,
                n_range=(50, 90), p_range=(30, 50), k_range=(30, 50),
                temp_range=(12, 22), humidity_range=(60, 75),
                ph_range=(6.0, 7.5), rainfall_range=(40, 80),
                duration="90-120 days",
                soil_type="Loam",
                water_requirement="Low (350-500 mm)",
                description="Cereal crop, drought tolerant",
                nutrient_demand="Low",
                temperature_sensitivity="Medium",
                drought_tolerance="High"
            ),
            CropSpecs(
                name="Mustard",
                category=CropCategory.OILSEED,
                season=Season.RABI,
                n_range=(40, 80), p_range=(30, 60), k_range=(30, 50),
                temp_range=(10, 25), humidity_range=(60, 75),
                ph_range=(5.5, 7.0), rainfall_range=(40, 70),
                duration="80-100 days",
                soil_type="Sandy loam",
                water_requirement="Low (300-450 mm)",
                description="Oilseed crop, cool season",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="High"
            ),
            CropSpecs(
                name="Soybean",
                category=CropCategory.OILSEED,
                season=Season.KHARIF,
                n_range=(20, 50), p_range=(40, 70), k_range=(30, 60),
                temp_range=(20, 30), humidity_range=(65, 80),
                ph_range=(6.0, 7.0), rainfall_range=(60, 100),
                duration="100-130 days",
                soil_type="Well-drained loam",
                water_requirement="Medium (450-700 mm)",
                description="Protein-rich oilseed, fixes nitrogen",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            ),
            CropSpecs(
                name="Peas",
                category=CropCategory.VEGETABLE,
                season=Season.RABI,
                n_range=(20, 40), p_range=(30, 60), k_range=(30, 50),
                temp_range=(10, 22), humidity_range=(60, 75),
                ph_range=(5.5, 7.0), rainfall_range=(40, 70),
                duration="60-80 days",
                soil_type="Loam",
                water_requirement="Low (300-450 mm)",
                description="Legume vegetable, fixes nitrogen",
                nutrient_demand="Low",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            ),
            CropSpecs(
                name="Onion",
                category=CropCategory.VEGETABLE,
                season=Season.RABI,
                n_range=(50, 80), p_range=(30, 50), k_range=(30, 50),
                temp_range=(13, 24), humidity_range=(60, 70),
                ph_range=(6.0, 7.0), rainfall_range=(50, 80),
                duration="100-120 days",
                soil_type="Well-drained loam",
                water_requirement="Medium (350-550 mm)",
                description="Bulb crop, requires cool climate",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            ),
            CropSpecs(
                name="Garlic",
                category=CropCategory.VEGETABLE,
                season=Season.RABI,
                n_range=(40, 70), p_range=(30, 50), k_range=(30, 50),
                temp_range=(12, 24), humidity_range=(60, 70),
                ph_range=(6.0, 7.0), rainfall_range=(40, 70),
                duration="90-120 days",
                soil_type="Sandy loam",
                water_requirement="Low (300-450 mm)",
                description="Bulb crop, medicinal value",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            ),
            CropSpecs(
                name="Chilli",
                category=CropCategory.SPICE,
                season=Season.KHARIF,
                n_range=(60, 100), p_range=(30, 60), k_range=(30, 60),
                temp_range=(20, 30), humidity_range=(60, 75),
                ph_range=(6.0, 7.0), rainfall_range=(60, 120),
                duration="90-120 days",
                soil_type="Well-drained loam",
                water_requirement="Medium (600-800 mm)",
                description="Spice crop, high value",
                nutrient_demand="Medium",
                temperature_sensitivity="Medium",
                drought_tolerance="Medium"
            )
        ]
    
    def generate_sample(self, crop: CropSpecs, include_noise: bool = True) -> Dict:
        """
        Generate a single sample for a crop
        
        Args:
            crop: Crop specifications
            include_noise: Whether to include random noise
            
        Returns:
            Dictionary with feature values
        """
        sample = {'crop': crop.name}
        
        for feature, (low, high) in crop.feature_ranges.items():
            # Generate value with normal distribution
            mean = (low + high) / 2
            std = (high - low) / 6  # 6 sigma covers 99.7% of range
            value = np.random.normal(mean, std)
            
            # Add noise if requested (simulating measurement errors)
            if include_noise and np.random.random() < 0.1:
                noise_factor = np.random.uniform(0.9, 1.1)
                value *= noise_factor
            
            # Clip to range
            value = np.clip(value, low, high)
            
            sample[feature] = round(value, 2)
        
        return sample
    
    def generate_dataset(self, samples_per_crop: int = 500, 
                        include_noise: bool = True) -> pd.DataFrame:
        """
        Generate complete dataset
        
        Args:
            samples_per_crop: Number of samples per crop
            include_noise: Whether to include noise
            
        Returns:
            DataFrame with generated data
        """
        logger.info(f"Generating dataset with {len(self.crops)} crops, {samples_per_crop} samples each")
        
        data = []
        for crop in self.crops:
            logger.debug(f"Generating {samples_per_crop} samples for {crop.name}")
            for _ in range(samples_per_crop):
                data.append(self.generate_sample(crop, include_noise))
        
        # Add some outlier samples (5% of total)
        n_outliers = int(len(data) * 0.05)
        logger.debug(f"Adding {n_outliers} outlier samples")
        
        for _ in range(n_outliers):
            crop = np.random.choice(self.crops)
            sample = self.generate_sample(crop, include_noise)
            # Add extreme values
            for feature in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
                if np.random.random() < 0.3:
                    low, high = crop.feature_ranges[feature]
                    if np.random.random() < 0.5:
                        sample[feature] = max(0, low - np.random.uniform(0, low * 0.3))
                    else:
                        sample[feature] = high + np.random.uniform(0, high * 0.3)
                    sample[feature] = round(sample[feature], 2)
            data.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        logger.success(f"Dataset generated: {len(df)} samples, {len(df.columns)} features")
        
        # Log distribution
        distribution = df['crop'].value_counts()
        for crop, count in distribution.items():
            logger.debug(f"  {crop}: {count} samples ({count/len(df)*100:.1f}%)")
        
        return df
    
    def generate_time_series(self, crop: CropSpecs, days: int = 365) -> pd.DataFrame:
        """
        Generate time series data for a crop
        
        Args:
            crop: Crop specifications
            days: Number of days to generate
            
        Returns:
            DataFrame with time series data
        """
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        data = []
        for date in dates:
            # Seasonal variations
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            sample = {
                'date': date,
                'crop': crop.name
            }
            
            for feature, (low, high) in crop.feature_ranges.items():
                mean = (low + high) / 2
                std = (high - low) / 6
                
                # Apply seasonal factor
                value = np.random.normal(mean * seasonal_factor, std)
                value = np.clip(value, low * 0.8, high * 1.2)
                
                sample[feature] = round(value, 2)
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def get_crop_info(self, crop_name: str) -> Optional[CropSpecs]:
        """Get crop specifications by name"""
        for crop in self.crops:
            if crop.name.lower() == crop_name.lower():
                return crop
        return None
    
    def get_crops_by_category(self, category: CropCategory) -> List[CropSpecs]:
        """Get crops by category"""
        return [c for c in self.crops if c.category == category]
    
    def get_crops_by_season(self, season: Season) -> List[CropSpecs]:
        """Get crops by season"""
        return [c for c in self.crops if c.season == season]
    
    def get_compatible_crops(self, **conditions) -> List[Tuple[CropSpecs, float]]:
        """
        Get crops compatible with given conditions
        
        Args:
            conditions: Feature conditions
            
        Returns:
            List of (crop, compatibility_score) tuples
        """
        compatible = []
        
        for crop in self.crops:
            score = 1.0
            
            for feature, value in conditions.items():
                if feature in crop.feature_ranges:
                    low, high = crop.feature_ranges[feature]
                    
                    if value < low:
                        score *= max(0, 1 - (low - value) / low)
                    elif value > high:
                        score *= max(0, 1 - (value - high) / high)
            
            if score > 0.5:  # Only include reasonably compatible crops
                compatible.append((crop, score))
        
        return sorted(compatible, key=lambda x: x[1], reverse=True)
    
    def save_to_csv(self, df: pd.DataFrame, path: Path):
        """Save dataset to CSV"""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.success(f"Dataset saved to {path}")
        
    def load_from_csv(self, path: Path) -> pd.DataFrame:
        """Load dataset from CSV"""
        df = pd.read_csv(path)
        logger.success(f"Dataset loaded from {path}: {len(df)} samples")
        return df