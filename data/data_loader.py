"""
Enterprise Data Loader with Validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from datetime import datetime
import json
import yaml
from loguru import logger
import pandera as pa
from pandera import DataFrameSchema, Column, Check

class DataLoader:
    """
    Professional data loader with validation and multiple format support
    """
    
    def __init__(self, validate: bool = True):
        self.validate = validate
        self.schema = self._create_schema()
        
    def _create_schema(self) -> DataFrameSchema:
        """Create data validation schema"""
        return DataFrameSchema({
            'N': Column(float, Check.in_range(0, 200), nullable=False),
            'P': Column(float, Check.in_range(0, 200), nullable=False),
            'K': Column(float, Check.in_range(0, 200), nullable=False),
            'temperature': Column(float, Check.in_range(0, 50), nullable=False),
            'humidity': Column(float, Check.in_range(0, 100), nullable=False),
            'ph': Column(float, Check.in_range(0, 14), nullable=False),
            'rainfall': Column(float, Check.in_range(0, 300), nullable=False),
            'crop': Column(str, nullable=False)
        })
    
    def load_csv(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        path = Path(path)
        logger.info(f"Loading CSV from {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            df = pd.read_csv(path, **kwargs)
            logger.success(f"Loaded {len(df)} rows from {path}")
            
            if self.validate:
                df = self.validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_excel(self, path: Union[str, Path], sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Load data from Excel file"""
        path = Path(path)
        logger.info(f"Loading Excel from {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
            logger.success(f"Loaded {len(df)} rows from {path}")
            
            if self.validate:
                df = self.validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    def load_json(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from JSON file"""
        path = Path(path)
        logger.info(f"Loading JSON from {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            logger.success(f"Loaded {len(df)} rows from {path}")
            
            if self.validate:
                df = self.validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def load_parquet(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from Parquet file"""
        path = Path(path)
        logger.info(f"Loading Parquet from {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            df = pd.read_parquet(path)
            logger.success(f"Loaded {len(df)} rows from {path}")
            
            if self.validate:
                df = self.validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise
    
    def save_csv(self, df: pd.DataFrame, path: Union[str, Path], **kwargs):
        """Save data to CSV file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.validate:
                df = self.validate_data(df)
            
            df.to_csv(path, index=False, **kwargs)
            logger.success(f"Saved {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise
    
    def save_excel(self, df: pd.DataFrame, path: Union[str, Path], sheet_name: str = "Sheet1", **kwargs):
        """Save data to Excel file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.validate:
                df = self.validate_data(df)
            
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)
            
            logger.success(f"Saved {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"Error saving Excel: {e}")
            raise
    
    def save_json(self, df: pd.DataFrame, path: Union[str, Path], orient: str = 'records', **kwargs):
        """Save data to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.validate:
                df = self.validate_data(df)
            
            if orient == 'records':
                data = df.to_dict(orient='records')
            elif orient == 'split':
                data = df.to_dict(orient='split')
            else:
                data = df.to_dict(orient=orient)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, **kwargs)
            
            logger.success(f"Saved {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            raise
    
    def save_parquet(self, df: pd.DataFrame, path: Union[str, Path], **kwargs):
        """Save data to Parquet file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.validate:
                df = self.validate_data(df)
            
            df.to_parquet(path, index=False, **kwargs)
            logger.success(f"Saved {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"Error saving Parquet: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data against schema"""
        logger.info("Validating data...")
        
        try:
            # Check required columns
            required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop']
            missing_columns = set(required_columns) - set(df.columns)
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} duplicate rows")
            
            # Remove rows with missing values
            initial_count = len(df)
            df = df.dropna()
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows with missing values")
            
            # Validate ranges
            for col, (min_val, max_val) in [
                ('N', (0, 200)),
                ('P', (0, 200)),
                ('K', (0, 200)),
                ('temperature', (0, 50)),
                ('humidity', (0, 100)),
                ('ph', (0, 14)),
                ('rainfall', (0, 300))
            ]:
                invalid = df[(df[col] < min_val) | (df[col] > max_val)]
                if not invalid.empty:
                    logger.warning(f"Found {len(invalid)} rows with {col} outside range [{min_val}, {max_val}]")
                    # Clip to range
                    df[col] = df[col].clip(min_val, max_val)
            
            logger.success(f"Data validation passed: {len(df)} valid rows")
            return df
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.1, random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Splitting data: train={1-test_size-val_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
        
        # First split: train+val and test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['crop']
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=random_state, stratify=train_val['crop']
        )
        
        logger.info(f"Train: {len(train)} samples")
        logger.info(f"Validation: {len(val)} samples")
        logger.info(f"Test: {len(test)} samples")
        
        return {
            'train': train,
            'validation': val,
            'test': test
        }
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        summary = {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,  # Exclude target
            'crop_distribution': df['crop'].value_counts().to_dict(),
            'feature_stats': {}
        }
        
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            summary['feature_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                '25%': df[col].quantile(0.25),
                '50%': df[col].median(),
                '75%': df[col].quantile(0.75)
            }
        
        return summary
    
    def sample_data(self, df: pd.DataFrame, n: int, stratified: bool = True) -> pd.DataFrame:
        """Sample data with optional stratification"""
        if stratified and 'crop' in df.columns:
            return df.groupby('crop', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // len(df['crop'].unique())), random_state=42)
            )
        else:
            return df.sample(n=min(n, len(df)), random_state=42)