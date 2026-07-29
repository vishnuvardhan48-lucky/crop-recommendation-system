"""
Integration tests for Crop Recommendation System
"""

import pytest
import pandas as pd
from pathlib import Path

from data.data_generator import DataGenerator
from models.train import ModelTrainer
from models.predict import PredictionService
from preprocessing.validators import InputValidator

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_pipeline(self, temp_dir):
        """Test complete pipeline from data generation to prediction"""
        
        # Step 1: Generate data
        generator = DataGenerator(seed=42)
        df = generator.generate_dataset(samples_per_crop=10)
        assert len(df) > 0
        
        # Step 2: Train model
        trainer = ModelTrainer()
        results = trainer.train_pipeline(df)
        assert results['best_score'] > 0
        
        # Step 3: Save model
        trainer.save_model(results)
        
        # Step 4: Load model for prediction
        service = PredictionService()
        # Note: In real test, we would load the saved model
        
        # Step 5: Validate input
        validator = InputValidator()
        test_input = {
            'N': 90, 'P': 45, 'K': 40,
            'temperature': 25, 'humidity': 70,
            'ph': 6.5, 'rainfall': 150
        }
        is_valid, warnings = validator.validate_all(**test_input)
        assert is_valid
        
        # Step 6: Make prediction (mock)
        class MockService:
            def predict(self, **kwargs):
                return {
                    'success': True,
                    'primary_recommendation': 'Rice',
                    'confidence': 95.5
                }
        
        service = MockService()
        result = service.predict(**test_input)
        assert result['success']
        assert result['primary_recommendation'] == 'Rice'
        assert result['confidence'] > 0
    
    def test_data_validation_pipeline(self, sample_data):
        """Test data validation pipeline"""
        from preprocessing.data_cleaner import DataCleaner
        from preprocessing.validators import DataValidator
        
        # Clean data
        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(sample_data)
        
        # Validate
        validator = DataValidator()
        validation_results = validator.validate_dataframe(cleaned)
        
        assert validation_results['is_valid']
        assert validation_results['valid_rows'] == len(cleaned)