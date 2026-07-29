"""
Enterprise Data Cleaning Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from loguru import logger

class DataCleaner:
    """
    Professional data cleaning with multiple strategies
    """
    
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = RobustScaler()
        self.cleaning_log = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        
        df_clean = df.copy()
        initial_shape = df_clean.shape
        
        # Step 1: Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Step 2: Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 3: Fix data types
        df_clean = self._fix_data_types(df_clean)
        
        # Step 4: Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Step 5: Validate ranges
        df_clean = self._validate_ranges(df_clean)
        
        # Step 6: Standardize categorical values
        df_clean = self._standardize_categories(df_clean)
        
        final_shape = df_clean.shape
        logger.success(f"Cleaning complete: {initial_shape} -> {final_shape}")
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_count - len(df_clean)
        
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} duplicate rows")
            logger.info(f"Removed {removed} duplicates")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        df_clean = df.copy()
        
        # Check for missing values
        missing = df_clean.isnull().sum()
        if missing.sum() == 0:
            return df_clean
        
        self.cleaning_log.append(f"Found {missing.sum()} missing values")
        
        # Separate numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use median for numeric columns
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                self.cleaning_log.append(f"Filled {col} missing values with median: {median_val:.2f}")
        
        # Handle categorical columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use mode for categorical columns
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                self.cleaning_log.append(f"Filled {col} missing values with mode: {mode_val}")
        
        logger.info(f"Handled {missing.sum()} missing values")
        
        return df_clean
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix incorrect data types"""
        df_clean = df.copy()
        
        # Ensure numeric columns are float
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Ensure categorical columns are string
        if 'crop' in df_clean.columns:
            df_clean['crop'] = df_clean['crop'].astype(str).str.strip()
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """Handle outliers using various methods"""
        df_clean = df.copy()
        
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        outliers_count = 0
        
        for col in numeric_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                outliers_count += len(outliers)
                
                if method == 'cap':
                    # Cap outliers
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    if len(outliers) > 0:
                        self.cleaning_log.append(f"Capped {len(outliers)} outliers in {col}")
                
                elif method == 'remove':
                    # Remove outliers
                    df_clean = df_clean[
                        (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                    ]
                    if len(outliers) > 0:
                        self.cleaning_log.append(f"Removed {len(outliers)} outliers from {col}")
        
        if outliers_count > 0:
            logger.info(f"Handled {outliers_count} outliers using {method} method")
        
        return df_clean
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix values outside acceptable ranges"""
        df_clean = df.copy()
        
        range_config = {
            'N': (0, 200),
            'P': (0, 200),
            'K': (0, 200),
            'temperature': (0, 50),
            'humidity': (0, 100),
            'ph': (0, 14),
            'rainfall': (0, 300)
        }
        
        for col, (min_val, max_val) in range_config.items():
            if col in df_clean.columns:
                # Clip values to range
                before_count = ((df_clean[col] < min_val) | (df_clean[col] > max_val)).sum()
                df_clean[col] = df_clean[col].clip(min_val, max_val)
                
                if before_count > 0:
                    self.cleaning_log.append(f"Clipped {before_count} values in {col} to range [{min_val}, {max_val}]")
        
        return df_clean
    
    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values"""
        df_clean = df.copy()
        
        if 'crop' in df_clean.columns:
            # Convert to title case and strip
            df_clean['crop'] = df_clean['crop'].str.strip().str.title()
            
            # Replace common variations
            replacements = {
                'Rice': ['rice', 'RICE', 'paddy'],
                'Wheat': ['wheat', 'WHEAT', 'gehu'],
                'Maize': ['maize', 'MAIZE', 'corn'],
                'Cotton': ['cotton', 'COTTON', 'kapas']
            }
            
            for standard, variants in replacements.items():
                for variant in variants:
                    df_clean.loc[df_clean['crop'] == variant, 'crop'] = standard
            
            self.cleaning_log.append("Standardized crop names")
        
        return df_clean
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations"""
        return {
            'operations_performed': len(self.cleaning_log),
            'logs': self.cleaning_log,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the data"""
        anomalies = pd.DataFrame()
        
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for col in numeric_cols:
            if col in df.columns:
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_anomalies = df[z_scores > 3]
                
                if not col_anomalies.empty:
                    temp_df = col_anomalies.copy()
                    temp_df['anomaly_feature'] = col
                    temp_df['anomaly_score'] = z_scores[z_scores > 3]
                    anomalies = pd.concat([anomalies, temp_df])
        
        return anomalies
    
    def suggest_corrections(self, df: pd.DataFrame) -> List[str]:
        """Suggest corrections for common data issues"""
        suggestions = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            suggestions.append(f"Handle {missing.sum()} missing values")
        
        # Check for outliers
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                if outliers > 0:
                    suggestions.append(f"Consider handling {outliers} outliers in {col}")
        
        # Check data types
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            if col in df.columns and df[col].dtype not in ['float64', 'int64']:
                suggestions.append(f"Convert {col} to numeric type")
        
        return suggestions