"""
Enterprise Feature Engineering Module
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

from config.logging_config import get_logger

logger = get_logger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering for crop recommendation
    Creates domain-specific features from raw agricultural data
    """
    
    def __init__(self, create_interactions: bool = True, 
                 create_ratios: bool = True,
                 create_polynomials: bool = False,
                 degree: int = 2):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_polynomials = create_polynomials
        self.degree = degree
        self.base_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.derived_features = []
        self.feature_names = None
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, include_bias=False) if create_polynomials else None
        self.fitted = False
        
    def _create_nutrient_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create nutrient ratio features"""
        df = X.copy()
        
        # NPK ratios (with epsilon to avoid division by zero)
        eps = 1e-6
        df['N_P_ratio'] = df['N'] / (df['P'] + eps)
        df['N_K_ratio'] = df['N'] / (df['K'] + eps)
        df['P_K_ratio'] = df['P'] / (df['K'] + eps)
        df['NPK_sum'] = df['N'] + df['P'] + df['K']
        df['NPK_product'] = df['N'] * df['P'] * df['K']
        
        # Nutrient balance indices
        df['N_balance'] = df['N'] / (df['NPK_sum'] + eps)
        df['P_balance'] = df['P'] / (df['NPK_sum'] + eps)
        df['K_balance'] = df['K'] / (df['NPK_sum'] + eps)
        
        self.derived_features.extend([
            'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
            'NPK_sum', 'NPK_product',
            'N_balance', 'P_balance', 'K_balance'
        ])
        
        return df
    
    def _create_environmental_indices(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create environmental condition indices"""
        df = X.copy()
        
        # Temperature-Humidity Index (THI)
        df['THI'] = df['temperature'] - (0.55 - 0.0055 * df['humidity']) * (df['temperature'] - 14.5)
        
        # Aridity Index (simplified)
        df['aridity_index'] = df['rainfall'] / (df['temperature'] + 5)
        
        # Growing Degree Days (simplified)
        df['GDD'] = np.maximum(0, df['temperature'] - 10)
        
        # Heat stress indicator
        df['heat_stress'] = np.maximum(0, df['temperature'] - 35)
        
        # Moisture index
        df['moisture_index'] = df['humidity'] * df['rainfall'] / 100
        
        # pH classification features
        df['pH_acidic'] = (df['ph'] < 5.5).astype(int)
        df['pH_neutral'] = ((df['ph'] >= 5.5) & (df['ph'] <= 7.5)).astype(int)
        df['pH_alkaline'] = (df['ph'] > 7.5).astype(int)
        
        self.derived_features.extend([
            'THI', 'aridity_index', 'GDD', 'heat_stress',
            'moisture_index', 'pH_acidic', 'pH_neutral', 'pH_alkaline'
        ])
        
        return df
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df = X.copy()
        
        # Nutrient × environmental interactions
        df['N_temp'] = df['N'] * df['temperature']
        df['P_temp'] = df['P'] * df['temperature']
        df['K_temp'] = df['K'] * df['temperature']
        
        df['N_humidity'] = df['N'] * df['humidity']
        df['P_humidity'] = df['P'] * df['humidity']
        df['K_humidity'] = df['K'] * df['humidity']
        
        df['N_rain'] = df['N'] * df['rainfall']
        df['P_rain'] = df['P'] * df['rainfall']
        df['K_rain'] = df['K'] * df['rainfall']
        
        df['N_pH'] = df['N'] * df['ph']
        df['P_pH'] = df['P'] * df['ph']
        df['K_pH'] = df['K'] * df['ph']
        
        # Environmental interactions
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['temp_rain'] = df['temperature'] * df['rainfall']
        df['humidity_rain'] = df['humidity'] * df['rainfall']
        df['temp_pH'] = df['temperature'] * df['ph']
        
        self.derived_features.extend([
            'N_temp', 'P_temp', 'K_temp',
            'N_humidity', 'P_humidity', 'K_humidity',
            'N_rain', 'P_rain', 'K_rain',
            'N_pH', 'P_pH', 'K_pH',
            'temp_humidity', 'temp_rain', 'humidity_rain', 'temp_pH'
        ])
        
        return df
    
    def _create_optimality_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create optimality scores based on crop requirements"""
        df = X.copy()
        
        # Optimal ranges for different crops (will be computed from data)
        # This is a simplified version - in production, use domain knowledge
        
        # Calculate z-scores for each feature
        for feature in self.base_features:
            mean = df[feature].mean()
            std = df[feature].std()
            if std > 0:
                df[f'{feature}_zscore'] = (df[feature] - mean) / std
                self.derived_features.append(f'{feature}_zscore')
        
        # Create optimality score (lower z-score is better)
        zscore_cols = [f'{f}_zscore' for f in self.base_features if f'{f}_zscore' in df.columns]
        if zscore_cols:
            df['optimality_score'] = 1 / (1 + np.abs(df[zscore_cols]).mean(axis=1))
            self.derived_features.append('optimality_score')
        
        return df
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        df = X.copy()
        
        # Rolling statistics (if data has temporal order)
        # This is a placeholder - in production, implement properly
        
        # Feature combinations
        df['N_normalized'] = df['N'] / df['N'].max() if df['N'].max() > 0 else df['N']
        df['P_normalized'] = df['P'] / df['P'].max() if df['P'].max() > 0 else df['P']
        df['K_normalized'] = df['K'] / df['K'].max() if df['K'].max() > 0 else df['K']
        
        self.derived_features.extend(['N_normalized', 'P_normalized', 'K_normalized'])
        
        return df
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineer"""
        logger.info("Fitting feature engineer...")
        
        # Store feature names
        self.feature_names = self.base_features.copy()
        
        # Fit polynomial features if enabled
        if self.poly:
            self.poly.fit(X[self.base_features])
        
        # Fit scaler on base features
        self.scaler.fit(X[self.base_features])
        
        self.fitted = True
        logger.info(f"Feature engineer fitted with {len(self.base_features)} base features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        logger.debug("Applying feature engineering...")
        
        # Start with base features
        df = X[self.base_features].copy()
        
        # Apply all feature engineering steps
        if self.create_ratios:
            df = self._create_nutrient_ratios(df)
        
        df = self._create_environmental_indices(df)
        
        if self.create_interactions:
            df = self._create_interaction_features(df)
        
        df = self._create_optimality_scores(df)
        df = self._create_statistical_features(df)
        
        # Add polynomial features if enabled
        if self.poly:
            poly_features = self.poly.transform(X[self.base_features])
            poly_df = pd.DataFrame(
                poly_features,
                columns=[f'poly_{i}' for i in range(poly_features.shape[1])],
                index=X.index
            )
            df = pd.concat([df, poly_df], axis=1)
            self.derived_features.extend(poly_df.columns.tolist())
        
        # Update feature names
        self.feature_names = df.columns.tolist()
        
        logger.debug(f"Feature engineering complete: {len(df.columns)} total features "
                    f"({len(self.base_features)} base + {len(self.derived_features)} derived)")
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names"""
        if not self.fitted:
            return self.base_features
        return self.feature_names
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups"""
        return {
            'base': self.base_features,
            'ratios': [f for f in self.derived_features if 'ratio' in f or 'balance' in f],
            'environmental': [f for f in self.derived_features if any(x in f for x in ['THI', 'aridity', 'GDD', 'stress', 'moisture', 'pH_'])],
            'interactions': [f for f in self.derived_features if '_' in f and not any(x in f for x in ['ratio', 'balance', 'normalized'])],
            'statistical': [f for f in self.derived_features if 'zscore' in f or 'normalized' in f or 'optimality' in f],
            'polynomial': [f for f in self.derived_features if 'poly_' in f]
        }
    
    def get_feature_importance_guide(self) -> Dict[str, str]:
        """Get guide for feature interpretation"""
        return {
            'N_P_ratio': 'Nitrogen to Phosphorus ratio - indicates nutrient balance',
            'N_K_ratio': 'Nitrogen to Potassium ratio - indicates nutrient balance',
            'P_K_ratio': 'Phosphorus to Potassium ratio - indicates nutrient balance',
            'NPK_sum': 'Total NPK nutrients - overall soil fertility',
            'THI': 'Temperature-Humidity Index - measures combined heat and humidity stress',
            'aridity_index': 'Aridity Index - lower values indicate drier conditions',
            'GDD': 'Growing Degree Days - measure of heat accumulation for crop growth',
            'heat_stress': 'Heat Stress indicator - values >0 indicate heat stress',
            'moisture_index': 'Moisture Index - combined measure of humidity and rainfall',
            'optimality_score': 'Overall optimality score - higher is better'
        }