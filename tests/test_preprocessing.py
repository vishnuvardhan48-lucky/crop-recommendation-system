"""
Tests for preprocessing module
"""

import pytest
import pandas as pd
import numpy as np

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.validators import InputValidator, DataValidator
from preprocessing.data_cleaner import DataCleaner

class TestFeatureEngineer:
    """Test feature engineering"""
    
    def test_initialization(self):
        engineer = FeatureEngineer()
        assert engineer.base_features == ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        assert not engineer.fitted
    
    def test_fit_transform(self, sample_data):
        engineer = FeatureEngineer()
        X = sample_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        
        result = engineer.fit_transform(X)
        
        assert result is not None
        assert len(result.columns) > len(X.columns)
        assert engineer.fitted
    
    def test_get_feature_names(self, sample_data):
        engineer = FeatureEngineer()
        X = sample_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        
        engineer.fit(X)
        features = engineer.get_feature_names()
        
        assert len(features) > 0
        assert 'N_P_ratio' in features

class TestInputValidator:
    """Test input validation"""
    
    def setup_method(self):
        self.validator = InputValidator()
    
    def test_valid_range(self):
        is_valid, warnings = self.validator.validate_range('N', 100)
        assert is_valid
        assert warnings is None
    
    def test_invalid_range(self):
        is_valid, warnings = self.validator.validate_range('N', 250)
        assert not is_valid
        assert 'exceeds maximum' in warnings
    
    def test_validate_all_valid(self, sample_input):
        is_valid, warnings = self.validator.validate_all(**sample_input)
        assert is_valid
        assert len(warnings) == 0
    
    def test_validate_all_invalid(self):
        inputs = {
            'N': 250,
            'P': 45,
            'K': 40,
            'temperature': 25,
            'humidity': 70,
            'ph': 6.5,
            'rainfall': 150
        }
        is_valid, warnings = self.validator.validate_all(**inputs)
        assert not is_valid
        assert len(warnings) > 0

class TestDataCleaner:
    """Test data cleaning"""
    
    def setup_method(self):
        self.cleaner = DataCleaner()
    
    def test_clean_data(self, sample_data):
        # Add some dirty data
        dirty_data = sample_data.copy()
        dirty_data.loc[0, 'N'] = np.nan
        dirty_data.loc[1, 'P'] = -10  # Invalid
        
        cleaned = self.cleaner.clean_data(dirty_data)
        
        assert cleaned is not None
        assert cleaned['N'].notna().all()
        assert cleaned['P'].min() >= 0
    
    def test_remove_duplicates(self, sample_data):
        # Add duplicate
        duplicate = pd.concat([sample_data, sample_data.iloc[[0]]])
        
        cleaned = self.cleaner._remove_duplicates(duplicate)
        
        assert len(cleaned) == len(sample_data)
    
    def test_handle_outliers(self, sample_data):
        cleaned = self.cleaner._handle_outliers(sample_data)
        
        # Check that values are within reasonable ranges
        assert cleaned['N'].between(0, 200).all()
        assert cleaned['P'].between(0, 200).all()
        assert cleaned['K'].between(0, 200).all()