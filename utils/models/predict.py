"""
Enterprise Prediction Service
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import joblib
import json
from functools import lru_cache
import hashlib
import time

from config.settings import settings
from config.logging_config import get_logger
from preprocessing.validators import InputValidator
from utils.cache import CacheManager
from utils.metrics import MetricsCollector

logger = get_logger(__name__)

class PredictionService:
    """
    Professional prediction service with caching and monitoring
    """
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.validator = InputValidator()
        self.cache = CacheManager(ttl=settings.CACHE_TTL) if settings.ENABLE_CACHING else None
        self.metrics = MetricsCollector()
        self.is_loaded = False
        self.feature_names = None
        self.class_names = None
        self.model_name = None
        self.model_version = settings.APP_VERSION
        self.prediction_count = 0
        self.total_response_time = 0
        self.error_count = 0
        
    def load_models(self) -> bool:
        """
        Load all model artifacts
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading model artifacts...")
            
            # Load model
            model_path = settings.MODEL_PATH
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load feature engineer
            engineer_path = settings.FEATURE_ENGINEER_PATH
            if engineer_path.exists():
                self.feature_engineer = joblib.load(engineer_path)
                logger.info("✅ Feature engineer loaded")
            
            # Load metadata
            metadata_path = settings.METADATA_PATH
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names')
                    self.class_names = metadata.get('class_names')
                    self.model_name = metadata.get('model_name')
                    
                logger.info(f"✅ Metadata loaded: {self.model_name}")
                logger.info(f"   Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
                logger.info(f"   Classes: {len(self.class_names) if self.class_names else 'N/A'}")
            
            self.is_loaded = True
            logger.success("✅ All model artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.error_count += 1
            return False
    
    def _get_cache_key(self, inputs: Dict[str, float]) -> str:
        """Generate cache key from inputs"""
        input_str = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _prepare_input(self, N: float, P: float, K: float,
                      temperature: float, humidity: float,
                      ph: float, rainfall: float) -> np.ndarray:
        """
        Prepare input array for prediction
        
        Args:
            Input parameters
            
        Returns:
            Prepared input array
        """
        # Create base array
        input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Apply feature engineering if available
        if self.feature_engineer:
            input_df = pd.DataFrame(
                input_array,
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            )
            input_engineered = self.feature_engineer.transform(input_df)
            return input_engineered.values
        else:
            return input_array
    
    def predict(self, N: float, P: float, K: float,
               temperature: float, humidity: float,
               ph: float, rainfall: float) -> Dict[str, Any]:
        """
        Make prediction for single input
        
        Args:
            N, P, K: Soil nutrients
            temperature, humidity, ph, rainfall: Environmental parameters
            
        Returns:
            Prediction dictionary
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        
        start_time = time.time()
        
        try:
            # Validate inputs
            inputs = {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            is_valid, warnings = self.validator.validate_all(**inputs)
            
            # Check cache
            if self.cache:
                cache_key = self._get_cache_key(inputs)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit for prediction")
                    self.metrics.record_cache_hit()
                    return cached_result
            
            # Prepare input
            input_array = self._prepare_input(
                N, P, K, temperature, humidity, ph, rainfall
            )
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            probabilities = self.model.predict_proba(input_array)[0]
            
            # Get class names
            if self.class_names:
                class_names = self.class_names
            else:
                class_names = [f"Class_{i}" for i in range(len(probabilities))]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = [
                {
                    'rank': i + 1,
                    'crop': class_names[idx],
                    'confidence': float(probabilities[idx] * 100),
                    'probability': float(probabilities[idx])
                }
                for i, idx in enumerate(top_indices)
            ]
            
            # Get all probabilities
            all_probabilities = {
                class_names[i]: float(probabilities[i] * 100)
                for i in range(len(class_names))
            }
            
            # Prepare result
            result = {
                'success': True,
                'primary_recommendation': class_names[prediction],
                'confidence': float(probabilities[prediction] * 100),
                'probability': float(probabilities[prediction]),
                'top_3': top_predictions,
                'all_probabilities': all_probabilities,
                'input_parameters': inputs,
                'warnings': warnings,
                'model_info': {
                    'name': self.model_name,
                    'version': self.model_version
                },
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, result)
            
            # Update metrics
            self.prediction_count += 1
            self.total_response_time += time.time() - start_time
            self.metrics.record_prediction(
                result['primary_recommendation'],
                result['confidence']
            )
            
            logger.debug(f"Prediction made: {result['primary_recommendation']} "
                        f"({result['confidence']:.1f}%)")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'input_parameters': {
                    'N': N, 'P': P, 'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple inputs
        
        Args:
            df: DataFrame with input features
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting batch prediction for {len(df)} samples")
            
            # Validate required columns
            required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Prepare features
            if self.feature_engineer:
                X = self.feature_engineer.transform(df[required_cols])
            else:
                X = df[required_cols].values
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Get class names
            if self.class_names:
                class_names = self.class_names
            else:
                class_names = [f"Class_{i}" for i in range(probabilities.shape[1])]
            
            # Add predictions to dataframe
            results = df.copy()
            results['predicted_crop'] = [class_names[p] for p in predictions]
            results['confidence'] = [
                probabilities[i][p] * 100 
                for i, p in enumerate(predictions)
            ]
            results['prediction_time'] = datetime.now().isoformat()
            
            # Add all probabilities
            for i, crop in enumerate(class_names):
                results[f'prob_{crop}'] = probabilities[:, i] * 100
            
            # Update metrics
            self.prediction_count += len(df)
            self.total_response_time += time.time() - start_time
            self.metrics.record_batch_prediction(len(df))
            
            logger.success(f"Batch prediction complete: {len(df)} samples processed")
            
            return results
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        avg_response_time = (
            self.total_response_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            'is_loaded': self.is_loaded,
            'model_name': self.model_name,
            'model_type': type(self.model).__name__ if self.model else None,
            'model_version': self.model_version,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'num_classes': len(self.class_names) if self.class_names else 0,
            'classes': self.class_names if self.class_names else [],
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'avg_response_time_ms': round(avg_response_time * 1000, 2),
            'cache_enabled': self.cache is not None,
            'cache_stats': self.cache.get_stats() if self.cache else None,
            'metrics': self.metrics.get_summary() if self.metrics else None
        }
    
    def predict_proba(self, **kwargs) -> Dict[str, float]:
        """
        Get prediction probabilities only
        
        Returns:
            Dictionary of crop -> probability
        """
        result = self.predict(**kwargs)
        if result['success']:
            return result['all_probabilities']
        return {}
    
    def predict_top_n(self, n: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """
        Get top N predictions
        
        Args:
            n: Number of top predictions to return
            
        Returns:
            List of top predictions
        """
        result = self.predict(**kwargs)
        if result['success']:
            return result['top_3'][:n]
        return []
    
    def explain_prediction(self, **kwargs) -> Dict[str, Any]:
        """
        Explain prediction (requires SHAP)
        
        Returns:
            Explanation dictionary
        """
        try:
            import shap
            
            # Prepare input
            input_array = self._prepare_input(**kwargs)
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(input_array)
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(
                self.feature_names,
                shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]
            ))
            
            return {
                'shap_values': shap_values.tolist(),
                'feature_importance': feature_importance,
                'base_value': float(explainer.expected_value)
            }
            
        except ImportError:
            logger.warning("SHAP not installed. Cannot explain prediction.")
            return {'error': 'SHAP not available'}
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cache:
            self.cache.clear()
        logger.info("Prediction service cleaned up")