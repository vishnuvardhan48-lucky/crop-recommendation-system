"""
Enterprise Model Training Pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import joblib
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from config.logging_config import get_logger
from preprocessing.feature_engineering import FeatureEngineer
from .evaluator import ModelEvaluator

logger = get_logger(__name__)

class ModelTrainer:
    """
    Professional model training with multiple algorithms and optimization
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.best_params = None
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.training_history = []
        self.feature_names = None
        self.class_names = None
        
    def _initialize_models(self) -> Dict[str, Dict]:
        """Initialize models with hyperparameter grids"""
        return {
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=settings.N_ESTIMATORS,
                    max_depth=settings.MAX_DEPTH,
                    min_samples_split=settings.MIN_SAMPLES_SPLIT,
                    min_samples_leaf=settings.MIN_SAMPLES_LEAF,
                    random_state=settings.RANDOM_SEED,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=settings.N_ESTIMATORS,
                    max_depth=settings.MAX_DEPTH,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=settings.RANDOM_SEED,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(
                    n_estimators=settings.N_ESTIMATORS,
                    max_depth=settings.MAX_DEPTH,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=settings.RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [31, 50, 70, 100],
                    'subsample': [0.6, 0.8, 1.0]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=settings.N_ESTIMATORS,
                    max_depth=settings.MAX_DEPTH,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=settings.RANDOM_SEED
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0]
                }
            },
            'SVM': {
                'model': SVC(
                    probability=True,
                    random_state=settings.RANDOM_SEED,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'degree': [2, 3, 4]  # for poly kernel
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(
                    n_jobs=-1,
                    weights='distance'
                ),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski'],
                    'p': [1, 2]
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(
                    max_depth=settings.MAX_DEPTH,
                    min_samples_split=settings.MIN_SAMPLES_SPLIT,
                    min_samples_leaf=settings.MIN_SAMPLES_LEAF,
                    random_state=settings.RANDOM_SEED,
                    class_weight='balanced'
                ),
                'params': {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=settings.RANDOM_SEED,
                    max_iter=1000,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['saga', 'liblinear'],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # for elasticnet
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(
                    iterations=settings.N_ESTIMATORS,
                    depth=settings.MAX_DEPTH,
                    learning_rate=0.1,
                    random_seed=settings.RANDOM_SEED,
                    verbose=False,
                    thread_count=-1
                ),
                'params': {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 7]
                }
            }
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Preparing data for training...")
        
        # Separate features and target
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[feature_cols].copy()
        y = df['crop'].copy()
        
        # Store class names
        self.class_names = sorted(y.unique())
        
        # Apply feature engineering
        X_engineered = self.feature_engineer.fit_transform(X)
        self.feature_names = self.feature_engineer.get_feature_names()
        
        logger.info(f"Features: {X_engineered.shape[1]} (base: {len(feature_cols)}, engineered: {X_engineered.shape[1] - len(feature_cols)})")
        logger.info(f"Target classes: {len(self.class_names)} crops")
        logger.info(f"Total samples: {len(df)}")
        
        return X_engineered.values, y.values
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, 
                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train models with cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting model training with {cv_folds}-fold CV...")
        
        results = {}
        cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=settings.RANDOM_SEED
        )
        
        for name, config in self.models.items():
            logger.info(f"📌 Training {name}...")
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(
                    config['model'], X, y, 
                    cv=cv_strategy, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                logger.info(f"   CV Accuracy: {mean_score:.4f} (+/- {std_score*2:.4f})")
                
                # Store results
                results[name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'model': config['model']
                }
                
                # Track best model
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = config['model']
                    self.best_model_name = name
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Display results
        logger.info("\n📊 CROSS-VALIDATION RESULTS")
        logger.info("="*60)
        
        sorted_results = sorted(
            [(n, r['mean_score']) for n, r in results.items() if 'mean_score' in r],
            key=lambda x: x[1], reverse=True
        )
        
        for name, score in sorted_results:
            logger.info(f"   {name:20s}: {score:.4f}")
        
        logger.info(f"\n🏆 BEST MODEL: {self.best_model_name} with accuracy {self.best_score:.4f}")
        
        return results
    
    def train_with_optimization(self, X: np.ndarray, y: np.ndarray,
                               cv_folds: int = 5, n_iter: int = 10) -> Dict[str, Any]:
        """
        Train with hyperparameter optimization
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            n_iter: Number of parameter iterations
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {n_iter} iterations...")
        
        results = {}
        cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=settings.RANDOM_SEED
        )
        
        for name, config in self.models.items():
            logger.info(f"📌 Optimizing {name}...")
            
            try:
                # Perform grid search
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv_strategy,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X, y)
                
                logger.info(f"   Best score: {grid_search.best_score_:.4f}")
                logger.info(f"   Best params: {grid_search.best_params_}")
                
                results[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'best_model': grid_search.best_estimator_,
                    'cv_results': grid_search.cv_results_
                }
                
                # Track best model
                if grid_search.best_score_ > self.best_score:
                    self.best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = name
                    self.best_params = grid_search.best_params_
                    
            except Exception as e:
                logger.error(f"Error optimizing {name}: {e}")
                results[name] = {'error': str(e)}
        
        logger.info(f"\n🏆 BEST MODEL: {self.best_model_name}")
        logger.info(f"   Score: {self.best_score:.4f}")
        logger.info(f"   Params: {self.best_params}")
        
        return results
    
    def train_pipeline(self, df: pd.DataFrame, optimize: bool = False) -> Dict[str, Any]:
        """
        Complete training pipeline
        
        Args:
            df: Input DataFrame
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with complete training results
        """
        logger.info("="*60)
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*60)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=settings.TRAIN_TEST_SPLIT,
            random_state=settings.RANDOM_SEED,
            stratify=y
        )
        
        # Train models
        if optimize:
            cv_results = self.train_with_optimization(X_train, y_train)
        else:
            cv_results = self.train_with_cv(X_train, y_train)
        
        # Fit best model on full training data
        logger.info(f"\n🔄 Fitting best model ({self.best_model_name}) on full training data...")
        self.best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        evaluation = self.evaluator.evaluate(
            self.best_model, X_test, y_test, self.class_names
        )
        
        # Get feature importance
        feature_importance = self.evaluator.get_feature_importance(
            self.best_model, self.feature_names
        )
        
        # Compile results
        results = {
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'cv_results': cv_results,
            'evaluation': evaluation,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'n_samples': len(df)
        }
        
        logger.success("\n✅ TRAINING PIPELINE COMPLETE")
        logger.success(f"   Best Model: {self.best_model_name}")
        logger.success(f"   Test Accuracy: {evaluation['metrics']['accuracy']:.4f}")
        logger.success(f"   Test F1 Score: {evaluation['metrics']['f1_score']:.4f}")
        
        return results
    
    def save_model(self, metadata: Dict[str, Any]):
        """
        Save model and all artifacts
        
        Args:
            metadata: Training metadata
        """
        logger.info("Saving model artifacts...")
        
        # Create directory
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = settings.MODEL_DIR / "best_model.pkl"
        joblib.dump(self.best_model, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        
        # Save feature engineer
        engineer_path = settings.MODEL_DIR / "feature_engineer.pkl"
        joblib.dump(self.feature_engineer, engineer_path)
        logger.info(f"✅ Feature engineer saved to {engineer_path}")
        
        # Save metadata
        metadata_path = settings.MODEL_DIR / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved to {metadata_path}")
        
        # Save feature importance plot
        if metadata.get('feature_importance'):
            self.evaluator.plot_feature_importance(
                metadata['feature_importance'],
                metadata['feature_names'],
                save_path=settings.MODEL_DIR / "feature_importance.png"
            )
        
        # Save confusion matrix plot
        if metadata['evaluation'].get('confusion_matrix'):
            self.evaluator.plot_confusion_matrix(
                metadata['evaluation']['confusion_matrix'],
                metadata['class_names'],
                save_path=settings.MODEL_DIR / "confusion_matrix.png"
            )
        
        # Save model info
        info_path = settings.MODEL_DIR / "model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model: {metadata['model_name']}\n")
            f.write(f"Accuracy: {metadata['evaluation']['metrics']['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metadata['evaluation']['metrics']['f1_score']:.4f}\n")
            f.write(f"Training Date: {metadata['training_date']}\n")
            f.write(f"Features: {metadata['n_features']}\n")
            f.write(f"Classes: {metadata['n_classes']}\n")
            f.write(f"Samples: {metadata['n_samples']}\n")
            
            if metadata.get('best_params'):
                f.write(f"\nBest Parameters:\n")
                for param, value in metadata['best_params'].items():
                    f.write(f"  {param}: {value}\n")
        
        logger.success(f"✅ All artifacts saved to {settings.MODEL_DIR}")
    
    def load_model(self) -> bool:
        """
        Load saved model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = settings.MODEL_DIR / "best_model.pkl"
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False
            
            self.best_model = joblib.load(model_path)
            
            engineer_path = settings.MODEL_DIR / "feature_engineer.pkl"
            if engineer_path.exists():
                self.feature_engineer = joblib.load(engineer_path)
            
            metadata_path = settings.MODEL_DIR / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names')
                    self.class_names = metadata.get('class_names')
                    self.best_model_name = metadata.get('model_name')
            
            logger.success(f"✅ Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False