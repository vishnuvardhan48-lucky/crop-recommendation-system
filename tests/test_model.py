"""
Unit tests for the preprocessing module.
Tests data validation, cleaning, and feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.validators import InputValidator, DataValidator
from preprocessing.data_cleaner import DataCleaner


# ============================================================================
# Test DataCleaner
# ============================================================================

class TestDataCleaner:
    """Test suite for DataCleaner class."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.cleaner = DataCleaner()
        self.sample_df = pd.DataFrame({
            'N': [90, 85, 75, None, 95],
            'P': [45, 42, 35, 50, None],
            'K': [40, 38, 35, 45, 42],
            'temperature': [25, 24, 18, 28, 26],
            'humidity': [70, 72, 65, 68, 71],
            'ph': [6.5, 6.3, 6.2, 6.8, 6.4],
            'rainfall': [150, 145, 75, 180, 160],
            'crop': ['Rice', 'Maize', 'Wheat', 'Cotton', 'Rice']
        })
    
    def test_initialization(self):
        """Test proper initialization of DataCleaner."""
        assert self.cleaner is not None
        assert hasattr(self.cleaner, 'num_imputer')
        assert hasattr(self.cleaner, 'cat_imputer')
        assert hasattr(self.cleaner, 'scaler')
        assert isinstance(self.cleaner.cleaning_log, list)
        assert len(self.cleaner.cleaning_log) == 0
    
    def test_remove_duplicates(self):
        """Test duplicate removal functionality."""
        # Add duplicates
        df_with_dupes = pd.concat([self.sample_df, self.sample_df.iloc[[0]]], ignore_index=True)
        assert len(df_with_dupes) == len(self.sample_df) + 1
        
        cleaned = self.cleaner._remove_duplicates(df_with_dupes)
        assert len(cleaned) == len(self.sample_df)
        assert len(self.cleaner.cleaning_log) > 0
        assert any('duplicate' in log.lower() for log in self.cleaner.cleaning_log)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Verify missing values exist
        assert self.sample_df['N'].isnull().sum() == 1
        assert self.sample_df['P'].isnull().sum() == 1
        
        cleaned = self.cleaner._handle_missing_values(self.sample_df)
        
        # Check missing values are filled
        assert cleaned['N'].isnull().sum() == 0
        assert cleaned['P'].isnull().sum() == 0
        
        # Check that appropriate logs were created
        assert len(self.cleaner.cleaning_log) > 0
        assert any('missing' in log.lower() for log in self.cleaner.cleaning_log)
    
    def test_handle_outliers_cap_method(self):
        """Test outlier handling with capping method."""
        # Add outliers
        df_with_outliers = self.sample_df.copy()
        df_with_outliers.loc[0, 'N'] = 500  # Extreme outlier
        
        cleaned = self.cleaner._handle_outliers(df_with_outliers, method='cap')
        
        # Check that outlier was capped (not exceeding reasonable range)
        assert cleaned['N'].max() <= 200
        assert len(self.cleaner.cleaning_log) > 0
        assert any('cap' in log.lower() for log in self.cleaner.cleaning_log)
    
    def test_handle_outliers_remove_method(self):
        """Test outlier handling with removal method."""
        # Add outliers
        df_with_outliers = self.sample_df.copy()
        df_with_outliers.loc[0, 'N'] = 500  # Extreme outlier
        
        initial_count = len(df_with_outliers)
        cleaned = self.cleaner._handle_outliers(df_with_outliers, method='remove')
        
        # Check that outlier row was removed
        assert len(cleaned) < initial_count
        assert len(self.cleaner.cleaning_log) > 0
        assert any('remov' in log.lower() for log in self.cleaner.cleaning_log)
    
    def test_validate_ranges(self):
        """Test range validation and clipping."""
        # Add out-of-range values
        df_invalid = self.sample_df.copy()
        df_invalid.loc[0, 'N'] = -10  # Below min
        df_invalid.loc[1, 'P'] = 250  # Above max
        
        cleaned = self.cleaner._validate_ranges(df_invalid)
        
        # Check values were clipped to valid ranges
        assert cleaned['N'].min() >= 0
        assert cleaned['P'].max() <= 200
        assert len(self.cleaner.cleaning_log) > 0
    
    def test_standardize_categories(self):
        """Test category standardization."""
        df_varied = self.sample_df.copy()
        df_varied.loc[0, 'crop'] = 'rice'
        df_varied.loc[1, 'crop'] = 'RICE'
        df_varied.loc[2, 'crop'] = 'WHEAT'
        
        cleaned = self.cleaner._standardize_categories(df_varied)
        
        # Check categories are standardized
        assert cleaned['crop'].iloc[0] == 'Rice'
        assert cleaned['crop'].iloc[1] == 'Rice'
        assert cleaned['crop'].iloc[2] == 'Wheat'
        assert len(self.cleaner.cleaning_log) > 0
    
    def test_full_clean_pipeline(self):
        """Test complete cleaning pipeline."""
        # Create dirty data
        dirty_data = self.sample_df.copy()
        dirty_data.loc[0, 'N'] = np.nan
        dirty_data.loc[1, 'P'] = 250  # Outlier
        dirty_data.loc[2, 'crop'] = 'wheat'  # Wrong case
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0]]])  # Duplicate
        
        initial_count = len(dirty_data)
        cleaned = self.cleaner.clean_data(dirty_data)
        
        # Verify all cleaning operations
        assert cleaned['N'].notna().all()
        assert cleaned['P'].max() <= 200
        assert cleaned['crop'].iloc[2] == 'Wheat'
        assert len(cleaned) < initial_count  # Duplicates removed
        assert len(self.cleaner.cleaning_log) >= 4  # Multiple operations logged
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        df_with_anomalies = self.sample_df.copy()
        df_with_anomalies.loc[0, 'N'] = 1000  # Extreme anomaly
        
        anomalies = self.cleaner.detect_anomalies(df_with_anomalies)
        assert len(anomalies) > 0
        assert 'anomaly_feature' in anomalies.columns
        assert 'anomaly_score' in anomalies.columns
    
    def test_suggest_corrections(self):
        """Test correction suggestions."""
        dirty_data = self.sample_df.copy()
        dirty_data.loc[0, 'N'] = np.nan
        dirty_data.loc[1, 'P'] = 250
        
        suggestions = self.cleaner.suggest_corrections(dirty_data)
        assert len(suggestions) > 0
        assert any('missing' in s.lower() for s in suggestions)
        assert any('outlier' in s.lower() for s in suggestions)
    
    def test_get_cleaning_summary(self):
        """Test cleaning summary generation."""
        # Perform some cleaning
        self.cleaner._remove_duplicates(self.sample_df)
        self.cleaner._handle_missing_values(self.sample_df)
        
        summary = self.cleaner.get_cleaning_summary()
        assert 'operations_performed' in summary
        assert summary['operations_performed'] >= 2
        assert 'logs' in summary
        assert len(summary['logs']) >= 2
        assert 'timestamp' in summary


# ============================================================================
# Test FeatureEngineer
# ============================================================================

class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.engineer = FeatureEngineer()
        self.sample_df = pd.DataFrame({
            'N': [90, 85, 75, 95, 80],
            'P': [45, 42, 35, 50, 40],
            'K': [40, 38, 35, 45, 38],
            'temperature': [25, 24, 18, 28, 22],
            'humidity': [70, 72, 65, 68, 69],
            'ph': [6.5, 6.3, 6.2, 6.8, 6.4],
            'rainfall': [150, 145, 75, 180, 120]
        })
    
    def test_initialization(self):
        """Test proper initialization of FeatureEngineer."""
        assert self.engineer is not None
        assert self.engineer.base_features == ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        assert not self.engineer.fitted
        assert self.engineer.create_interactions is True
        assert self.engineer.create_ratios is True
    
    def test_fit(self):
        """Test fit method."""
        X = self.sample_df[self.engineer.base_features]
        self.engineer.fit(X)
        
        assert self.engineer.fitted
        assert self.engineer.feature_names is not None
        assert len(self.engineer.feature_names) >= len(self.engineer.base_features)
    
    def test_transform_without_fit(self):
        """Test that transform raises error without fit."""
        X = self.sample_df[self.engineer.base_features]
        
        with pytest.raises(ValueError, match="fitted"):
            self.engineer.transform(X)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X = self.sample_df[self.engineer.base_features]
        result = self.engineer.fit_transform(X)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(X.columns)
        assert self.engineer.fitted
    
    def test_nutrient_ratios_creation(self):
        """Test creation of nutrient ratio features."""
        X = self.sample_df[self.engineer.base_features]
        result = self.engineer._create_nutrient_ratios(X)
        
        expected_features = ['N_P_ratio', 'N_K_ratio', 'P_K_ratio', 'NPK_sum', 'NPK_product']
        for feature in expected_features:
            assert feature in result.columns
    
    def test_environmental_indices_creation(self):
        """Test creation of environmental indices."""
        X = self.sample_df[self.engineer.base_features]
        result = self.engineer._create_environmental_indices(X)
        
        expected_features = ['THI', 'aridity_index', 'GDD', 'heat_stress', 'moisture_index']
        for feature in expected_features:
            assert feature in result.columns
    
    def test_interaction_features_creation(self):
        """Test creation of interaction features."""
        X = self.sample_df[self.engineer.base_features]
        result = self.engineer._create_interaction_features(X)
        
        expected_features = ['N_temp', 'P_temp', 'K_temp', 'N_humidity', 'P_humidity']
        for feature in expected_features:
            assert feature in result.columns
    
    def test_optimality_scores_creation(self):
        """Test creation of optimality scores."""
        X = self.sample_df[self.engineer.base_features]
        result = self.engineer._create_optimality_scores(X)
        
        assert 'optimality_score' in result.columns
        for feature in self.engineer.base_features:
            assert f'{feature}_zscore' in result.columns
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        X = self.sample_df[self.engineer.base_features]
        self.engineer.fit(X)
        
        feature_names = self.engineer.get_feature_names()
        assert len(feature_names) > 0
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_get_feature_groups(self):
        """Test getting feature groups."""
        X = self.sample_df[self.engineer.base_features]
        self.engineer.fit_transform(X)
        
        groups = self.engineer.get_feature_groups()
        assert 'base' in groups
        assert 'ratios' in groups
        assert 'environmental' in groups
        assert 'interactions' in groups
        assert len(groups['base']) == len(self.engineer.base_features)
    
    def test_polynomial_features(self):
        """Test polynomial feature creation when enabled."""
        engineer_poly = FeatureEngineer(create_polynomials=True, degree=2)
        X = self.sample_df[self.engineer.base_features]
        result = engineer_poly.fit_transform(X)
        
        assert result.shape[1] > X.shape[1]
        assert any('poly_' in col for col in result.columns)


# ============================================================================
# Test InputValidator
# ============================================================================

class TestInputValidator:
    """Test suite for InputValidator class."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.validator = InputValidator()
    
    def test_initialization(self):
        """Test proper initialization of InputValidator."""
        assert self.validator is not None
        assert 'N' in self.validator.feature_ranges
        assert 'P' in self.validator.feature_ranges
        assert 'K' in self.validator.feature_ranges
        assert len(self.validator.optimal_ranges) > 0
    
    def test_validate_range_valid(self):
        """Test range validation with valid values."""
        is_valid, warning = self.validator.validate_range('N', 100)
        assert is_valid is True
        assert warning is None
    
    def test_validate_range_below_min(self):
        """Test range validation with values below minimum."""
        is_valid, warning = self.validator.validate_range('N', -10)
        assert is_valid is False
        assert warning is not None
        assert 'below minimum' in warning
    
    def test_validate_range_above_max(self):
        """Test range validation with values above maximum."""
        is_valid, warning = self.validator.validate_range('N', 250)
        assert is_valid is False
        assert warning is not None
        assert 'exceeds maximum' in warning
    
    def test_validate_type_valid(self):
        """Test type validation with valid types."""
        is_valid, warning = self.validator.validate_type('N', 100.0)
        assert is_valid is True
        assert warning is None
        
        is_valid, warning = self.validator.validate_type('N', 100)
        assert is_valid is True
        assert warning is None
    
    def test_validate_type_invalid(self):
        """Test type validation with invalid types."""
        is_valid, warning = self.validator.validate_type('N', '100')
        assert is_valid is False
        assert warning is not None
        assert 'must be a number' in warning
        
        is_valid, warning = self.validator.validate_type('N', None)
        assert is_valid is False
        assert warning is not None
    
    def test_validate_soil_balance(self):
        """Test soil balance validation."""
        # Test balanced nutrients
        warnings = self.validator.validate_soil_balance(90, 45, 40)
        assert len(warnings) == 0
        
        # Test high nitrogen
        warnings = self.validator.validate_soil_balance(180, 10, 10)
        assert len(warnings) > 0
        assert any('high Nitrogen' in w for w in warnings)
        
        # Test low nutrients
        warnings = self.validator.validate_soil_balance(20, 10, 10)
        assert len(warnings) > 0
        assert any('Low Nitrogen' in w for w in warnings)
    
    def test_validate_environmental(self):
        """Test environmental validation."""
        # Test normal conditions
        warnings = self.validator.validate_environmental(25, 70, 6.5, 150)
        assert len(warnings) == 0
        
        # Test extreme temperature
        warnings = self.validator.validate_environmental(45, 70, 6.5, 150)
        assert len(warnings) > 0
        assert any('high temperature' in w.lower() for w in warnings)
        
        # Test extreme pH
        warnings = self.validator.validate_environmental(25, 70, 4.0, 150)
        assert len(warnings) > 0
        assert any('acidic' in w.lower() for w in warnings)
    
    def test_validate_all_valid(self, sample_input):
        """Test complete validation with valid inputs."""
        is_valid, warnings = self.validator.validate_all(**sample_input)
        assert is_valid is True
        assert len(warnings) == 0
    
    def test_validate_all_invalid(self, invalid_inputs):
        """Test complete validation with various invalid inputs."""
        for inputs in invalid_inputs:
            is_valid, warnings = self.validator.validate_all(**inputs)
            assert is_valid is False
            assert len(warnings) > 0
    
    def test_get_crop_compatibility(self, sample_input):
        """Test crop compatibility calculation."""
        compatibility = self.validator.get_crop_compatibility(**sample_input)
        
        assert isinstance(compatibility, dict)
        assert len(compatibility) > 0
        assert all(0 <= score <= 100 for score in compatibility.values())
        # Should be sorted by score descending
        scores = list(compatibility.values())
        assert scores == sorted(scores, reverse=True)
    
    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        # Test low nutrients
        suggestions = self.validator.suggest_improvements(N=30, P=20, K=15, ph=6.5)
        assert len(suggestions) > 0
        assert any('nitrogen' in s.lower() for s in suggestions)
        
        # Test low pH
        suggestions = self.validator.suggest_improvements(N=90, P=45, K=40, ph=5.0)
        assert len(suggestions) > 0
        assert any('lime' in s.lower() for s in suggestions)


# ============================================================================
# Test DataValidator
# ============================================================================

class TestDataValidator:
    """Test suite for DataValidator class."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.validator = DataValidator()
        self.sample_df = pd.DataFrame({
            'N': [90, 85, 75, 95, 80],
            'P': [45, 42, 35, 50, 40],
            'K': [40, 38, 35, 45, 38],
            'temperature': [25, 24, 18, 28, 22],
            'humidity': [70, 72, 65, 68, 69],
            'ph': [6.5, 6.3, 6.2, 6.8, 6.4],
            'rainfall': [150, 145, 75, 180, 120],
            'crop': ['Rice', 'Maize', 'Wheat', 'Cotton', 'Rice']
        })
    
    def test_validate_dataframe_valid(self):
        """Test DataFrame validation with valid data."""
        results = self.validator.validate_dataframe(self.sample_df)
        
        assert results['is_valid'] is True
        assert results['total_rows'] == len(self.sample_df)
        assert results['valid_rows'] == len(self.sample_df)
        assert results['invalid_rows'] == 0
        assert 'column_stats' in results
        assert 'crop_distribution' in results
    
    def test_validate_dataframe_missing_columns(self):
        """Test DataFrame validation with missing columns."""
        df_missing = self.sample_df.drop('N', axis=1)
        
        results = self.validator.validate_dataframe(df_missing)
        
        assert results['is_valid'] is False
        assert len(results['errors']) > 0
        assert any('missing' in e.lower() for e in results['errors'])
    
    def test_validate_dataframe_with_invalid_rows(self):
        """Test DataFrame validation with invalid rows."""
        df_invalid = self.sample_df.copy()
        df_invalid.loc[0, 'N'] = -10  # Invalid
        
        results = self.validator.validate_dataframe(df_invalid)
        
        assert results['valid_rows'] < results['total_rows']
        assert results['invalid_rows'] > 0
        assert len(results['warnings']) > 0
    
    def test_count_outliers(self):
        """Test outlier counting."""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        count = self.validator._count_outliers(series)
        assert count == 1
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        report = self.validator.generate_validation_report(self.sample_df)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "VALIDATION REPORT" in report.upper()
        assert "CROP DISTRIBUTION" in report.upper()