"""
Pytest configuration and fixtures for the Crop Recommendation System test suite.
Provides reusable test fixtures and setup/teardown hooks.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os
import json
from datetime import datetime
from typing import Dict, Any, Generator

from models.predict import PredictionService
from data.data_generator import DataGenerator
from config.settings import settings

# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_data() -> pd.DataFrame:
    """
    Generate sample data for testing.
    This fixture runs once per test session and provides consistent test data.
    """
    print("\n📊 Generating sample test data...")
    generator = DataGenerator(seed=RANDOM_SEED)
    df = generator.generate_dataset(samples_per_crop=10)
    print(f"✅ Generated {len(df)} test samples")
    return df


@pytest.fixture(scope="function")
def sample_input() -> Dict[str, float]:
    """
    Provide a standard input dictionary for prediction tests.
    This fixture runs for each test function.
    """
    return {
        'N': 90.0,
        'P': 45.0,
        'K': 40.0,
        'temperature': 25.0,
        'humidity': 70.0,
        'ph': 6.5,
        'rainfall': 150.0
    }


@pytest.fixture(scope="function")
def invalid_inputs() -> list:
    """
    Provide a list of invalid inputs for testing validation.
    """
    return [
        {'N': -10, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},  # Negative N
        {'N': 250, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},  # N too high
        {'N': 90, 'P': -5, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},   # Negative P
        {'N': 90, 'P': 250, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},  # P too high
        {'N': 90, 'P': 45, 'K': -10, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150}, # Negative K
        {'N': 90, 'P': 45, 'K': 250, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},  # K too high
        {'N': 90, 'P': 45, 'K': 40, 'temperature': -5, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},   # Temp negative
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 60, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},   # Temp too high
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': -10, 'ph': 6.5, 'rainfall': 150},  # Humidity negative
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 120, 'ph': 6.5, 'rainfall': 150},  # Humidity too high
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': -1, 'rainfall': 150},    # pH negative
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 15, 'rainfall': 150},    # pH too high
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': -10},   # Rainfall negative
        {'N': 90, 'P': 45, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 350},   # Rainfall too high
    ]


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for file operations.
    Automatically cleaned up after the test.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    print(f"📁 Created temporary directory: {temp_path}")
    yield temp_path
    shutil.rmtree(temp_path)
    print(f"🧹 Cleaned up temporary directory: {temp_path}")


@pytest.fixture(scope="session")
def sample_csv_file(temp_dir) -> Path:
    """
    Create a sample CSV file for testing data loading.
    """
    import csv
    
    file_path = temp_dir / "test_data.csv"
    
    # Create sample data
    data = [
        ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop'],
        [90, 45, 40, 25, 70, 6.5, 150, 'Rice'],
        [85, 42, 38, 24, 72, 6.3, 145, 'Maize'],
        [75, 35, 35, 18, 65, 6.2, 75, 'Wheat'],
    ]
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    return file_path


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def mock_prediction_service():
    """
    Create a mock prediction service for testing.
    This avoids loading actual models during unit tests.
    """
    class MockPredictionService:
        def __init__(self):
            self.is_loaded = True
            self.prediction_count = 0
            self.class_names = ['Rice', 'Wheat', 'Maize', 'Cotton']
            
        def predict(self, **kwargs):
            self.prediction_count += 1
            return {
                'success': True,
                'primary_recommendation': 'Rice',
                'confidence': 95.5,
                'top_3': [
                    {'rank': 1, 'crop': 'Rice', 'confidence': 95.5},
                    {'rank': 2, 'crop': 'Maize', 'confidence': 85.2},
                    {'rank': 3, 'crop': 'Wheat', 'confidence': 75.8}
                ],
                'all_probabilities': {
                    'Rice': 95.5,
                    'Maize': 85.2,
                    'Wheat': 75.8,
                    'Cotton': 65.3
                },
                'warnings': [],
                'processing_time_ms': 45.2
            }
        
        def get_model_info(self):
            return {
                'is_loaded': True,
                'model_name': 'Random Forest',
                'num_features': 7,
                'num_classes': 4,
                'classes': self.class_names,
                'prediction_count': self.prediction_count
            }
    
    return MockPredictionService()


@pytest.fixture(scope="session")
def trained_model_path(temp_dir) -> Path:
    """
    Create a mock trained model file for testing.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    model_path = temp_dir / "best_model.pkl"
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(100, 7)
    y = np.random.randint(0, 4, 100)
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    return model_path


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """
    Set mock environment variables for testing.
    """
    monkeypatch.setenv("APP_ENVIRONMENT", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENABLE_CACHING", "false")
    return monkeypatch


@pytest.fixture(scope="function")
def test_settings():
    """
    Provide test-specific settings.
    """
    from types import SimpleNamespace
    
    return SimpleNamespace(
        APP_NAME="Crop Recommendation Test",
        APP_VERSION="1.0.0-test",
        ENVIRONMENT="testing",
        DEBUG=True,
        TEST_MODE=True
    )


# ============================================================================
# Setup and Teardown Hooks
# ============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook.
    Runs before test collection.
    """
    print("\n" + "="*60)
    print("🌱 CROP RECOMMENDATION SYSTEM TEST SUITE")
    print("="*60)
    print(f"Python version: {os.sys.version}")
    print(f"Test root: {os.getcwd()}")
    print("="*60 + "\n")


def pytest_unconfigure(config):
    """
    Pytest unconfiguration hook.
    Runs after all tests.
    """
    print("\n" + "="*60)
    print("✅ TEST SUITE COMPLETE")
    print("="*60 + "\n")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Auto-use fixture that runs before each test.
    Sets up test environment and cleans up afterward.
    """
    # Setup
    print(f"\n🔧 Setting up test: {os.environ.get('PYTEST_CURRENT_TEST', 'Unknown')}")
    
    yield  # Test runs here
    
    # Teardown
    print(f"🧹 Cleaning up test")


# ============================================================================
# Custom Assertions
# ============================================================================

@pytest.fixture
def assert_valid_prediction():
    """
    Custom assertion for prediction results.
    """
    def _assert_valid_prediction(result):
        assert result is not None, "Prediction result should not be None"
        assert 'success' in result, "Result should contain 'success' field"
        if result['success']:
            assert 'primary_recommendation' in result, "Successful prediction should have primary_recommendation"
            assert 'confidence' in result, "Successful prediction should have confidence"
            assert 0 <= result['confidence'] <= 100, f"Confidence {result['confidence']} should be between 0 and 100"
            assert 'top_3' in result, "Successful prediction should have top_3 recommendations"
            assert len(result['top_3']) > 0, "Should have at least one recommendation"
    
    return _assert_valid_prediction


# ============================================================================
# Parameterized Test Data
# ============================================================================

@pytest.fixture(params=[
    {'crop': 'Rice', 'optimal': {'N': 90, 'P': 45, 'K': 40, 'temp': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150}},
    {'crop': 'Wheat', 'optimal': {'N': 75, 'P': 35, 'K': 35, 'temp': 18, 'humidity': 65, 'ph': 6.2, 'rainfall': 75}},
    {'crop': 'Maize', 'optimal': {'N': 85, 'P': 42, 'K': 38, 'temp': 24, 'humidity': 72, 'ph': 6.3, 'rainfall': 145}},
])
def crop_optimal_values(request):
    """
    Parameterized fixture for testing different crop optimal values.
    """
    return request.param