"""
Enterprise Model Evaluator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, log_loss)
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from config.logging_config import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Professional model evaluation with comprehensive metrics
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                class_names: List[str]) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            class_names: Names of classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': precision_score(y_test == i, y_pred == i),
                'recall': recall_score(y_test == i, y_pred == i),
                'f1_score': f1_score(y_test == i, y_pred == i),
                'support': np.sum(y_test == i)
            }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC AUC (for binary or multiclass)
        try:
            if len(class_names) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                # One-vs-Rest ROC AUC
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_test_bin, y_proba, multi_class='ovr', average='weighted'
                )
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Calculate log loss
        try:
            metrics['log_loss'] = log_loss(y_test, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate log loss: {e}")
        
        # Get classification report
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Calculate error analysis
        errors = {
            'total_errors': np.sum(y_test != y_pred),
            'error_rate': np.mean(y_test != y_pred),
            'confusion_pairs': self._get_confusion_pairs(y_test, y_pred, class_names)
        }
        
        results = {
            'metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'errors': errors,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_values': y_test.tolist()
        }
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        
        return results
    
    def _get_confusion_pairs(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str]) -> List[Dict]:
        """Get most confused class pairs"""
        cm = confusion_matrix(y_true, y_pred)
        pairs = []
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i][j] > 0:
                    pairs.append({
                        'true': class_names[i],
                        'predicted': class_names[j],
                        'count': int(cm[i][j])
                    })
        
        return sorted(pairs, key=lambda x: x['count'], reverse=True)
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance if available
        
        Args:
            model: Trained model
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importance
        """
        importance = {}
        
        # Try different importance attributes
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            importance = dict(zip(feature_names, coef))
        else:
            logger.warning("Model does not provide feature importance")
            return {}
        
        # Sort by importance
        importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance
    
    def plot_confusion_matrix(self, cm: List[List[int]], class_names: List[str],
                            save_path: Optional[Path] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Convert to numpy array
        cm_array = np.array(cm)
        
        # Calculate percentages
        cm_percentage = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Crop', fontsize=14)
        plt.ylabel('Actual Crop', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance: Dict[str, float],
                               feature_names: List[str],
                               top_n: int = 20,
                               save_path: Optional[Path] = None):
        """Plot feature importance"""
        # Sort and get top N
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_items)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       class_names: List[str], save_path: Optional[Path] = None):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        y_score = model.predict_proba(X_test)
        
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves by Class', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate multiple metrics
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        results = {}
        
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        
        logger.info(f"Cross-validation results:")
        logger.info(f"  Accuracy: {results['accuracy']['mean']:.4f} (+/- {results['accuracy']['std']*2:.4f})")
        logger.info(f"  F1 Score: {results['f1_weighted']['mean']:.4f}")
        
        return results
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray,
                      y_test: np.ndarray, class_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model name -> model
            X_test: Test features
            y_test: Test targets
            class_names: Names of classes
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(y_test, y_pred, average='weighted')
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('F1 Score', ascending=False)
        
        logger.info("\n📊 MODEL COMPARISON")
        logger.info(df.to_string())
        
        return df