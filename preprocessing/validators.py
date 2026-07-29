"""
Enterprise Input Validation Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import re
from loguru import logger

class InputValidator:
    """
    Professional input validation for agricultural parameters
    """
    
    def __init__(self):
        self.feature_ranges = {
            'N': {'min': 0, 'max': 200, 'unit': 'kg/ha', 'description': 'Nitrogen content'},
            'P': {'min': 0, 'max': 200, 'unit': 'kg/ha', 'description': 'Phosphorus content'},
            'K': {'min': 0, 'max': 200, 'unit': 'kg/ha', 'description': 'Potassium content'},
            'temperature': {'min': 0, 'max': 50, 'unit': '°C', 'description': 'Average temperature'},
            'humidity': {'min': 0, 'max': 100, 'unit': '%', 'description': 'Relative humidity'},
            'ph': {'min': 0, 'max': 14, 'unit': '', 'description': 'Soil pH'},
            'rainfall': {'min': 0, 'max': 300, 'unit': 'mm', 'description': 'Annual rainfall'}
        }
        
        self.optimal_ranges = {
            'Rice': {
                'N': (80, 120), 'P': (40, 60), 'K': (40, 60),
                'temperature': (22, 32), 'humidity': (70, 90),
                'ph': (5.5, 6.5), 'rainfall': (150, 250)
            },
            'Wheat': {
                'N': (60, 100), 'P': (30, 50), 'K': (30, 50),
                'temperature': (15, 25), 'humidity': (60, 80),
                'ph': (6.0, 7.0), 'rainfall': (50, 100)
            },
            'Maize': {
                'N': (70, 110), 'P': (30, 60), 'K': (30, 60),
                'temperature': (20, 30), 'humidity': (65, 85),
                'ph': (5.8, 7.0), 'rainfall': (80, 150)
            },
            'Cotton': {
                'N': (80, 130), 'P': (40, 70), 'K': (40, 70),
                'temperature': (25, 35), 'humidity': (60, 80),
                'ph': (6.0, 7.5), 'rainfall': (70, 120)
            }
        }
    
    def validate_range(self, feature: str, value: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if value is within acceptable range
        
        Returns:
            Tuple of (is_valid, warning_message)
        """
        if feature not in self.feature_ranges:
            return True, None
        
        range_info = self.feature_ranges[feature]
        min_val = range_info['min']
        max_val = range_info['max']
        
        if value < min_val:
            return False, f"{feature} value {value} is below minimum {min_val} {range_info['unit']}"
        elif value > max_val:
            return False, f"{feature} value {value} exceeds maximum {max_val} {range_info['unit']}"
        
        return True, None
    
    def validate_type(self, feature: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate data type"""
        if not isinstance(value, (int, float)):
            return False, f"{feature} must be a number, got {type(value).__name__}"
        
        if np.isnan(value):
            return False, f"{feature} cannot be NaN"
        
        if np.isinf(value):
            return False, f"{feature} cannot be infinite"
        
        return True, None
    
    def validate_soil_balance(self, N: float, P: float, K: float) -> List[str]:
        """Validate soil nutrient balance"""
        warnings = []
        
        # Check for extreme imbalances
        total = N + P + K
        if total > 0:
            if N / total > 0.7:
                warnings.append("Very high Nitrogen ratio - may cause excessive vegetative growth")
            if P / total > 0.5:
                warnings.append("Very high Phosphorus ratio - risk of runoff and water pollution")
            if K / total > 0.6:
                warnings.append("Very high Potassium ratio - may affect uptake of other nutrients")
        
        # Check for deficiencies
        if N < 40:
            warnings.append("Low Nitrogen - plants may show yellowing and stunted growth")
        if P < 20:
            warnings.append("Low Phosphorus - may affect root development and flowering")
        if K < 30:
            warnings.append("Low Potassium - may affect fruit quality and disease resistance")
        
        return warnings
    
    def validate_environmental(self, temperature: float, humidity: float, 
                              ph: float, rainfall: float) -> List[str]:
        """Validate environmental conditions"""
        warnings = []
        
        # Temperature warnings
        if temperature < 10:
            warnings.append("Very low temperature - may cause frost damage to sensitive crops")
        elif temperature > 40:
            warnings.append("Extremely high temperature - may cause heat stress")
        elif temperature > 35:
            warnings.append("High temperature - consider heat-tolerant crops")
        
        # Humidity warnings
        if humidity < 30:
            warnings.append("Very low humidity - may cause water stress")
        elif humidity > 90:
            warnings.append("Extremely high humidity - risk of fungal diseases")
        
        # pH warnings
        if ph < 5.0:
            warnings.append("Very acidic soil - most crops prefer pH 5.5-7.0")
        elif ph > 8.5:
            warnings.append("Very alkaline soil - most crops prefer pH 5.5-7.0")
        elif ph < 5.5:
            warnings.append("Acidic soil - consider lime application")
        elif ph > 7.5:
            warnings.append("Alkaline soil - consider sulfur application")
        
        # Rainfall warnings
        if rainfall < 50:
            warnings.append("Very low rainfall - irrigation required")
        elif rainfall > 250:
            warnings.append("Extremely high rainfall - risk of waterlogging")
        
        return warnings
    
    def validate_all(self, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate all input parameters
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        for feature, value in kwargs.items():
            # Type validation
            valid_type, type_warning = self.validate_type(feature, value)
            if not valid_type:
                is_valid = False
                warnings.append(type_warning)
                continue
            
            # Range validation
            valid_range, range_warning = self.validate_range(feature, value)
            if not valid_range:
                is_valid = False
                warnings.append(range_warning)
        
        # Additional validations if all basic checks pass
        if is_valid:
            if all(f in kwargs for f in ['N', 'P', 'K']):
                warnings.extend(self.validate_soil_balance(
                    kwargs['N'], kwargs['P'], kwargs['K']
                ))
            
            if all(f in kwargs for f in ['temperature', 'humidity', 'ph', 'rainfall']):
                warnings.extend(self.validate_environmental(
                    kwargs['temperature'], kwargs['humidity'],
                    kwargs['ph'], kwargs['rainfall']
                ))
        
        return is_valid, warnings
    
    def get_crop_compatibility(self, **kwargs) -> Dict[str, float]:
        """
        Get compatibility scores for different crops
        
        Returns:
            Dictionary of crop -> compatibility score (0-100)
        """
        scores = {}
        
        for crop, optimal in self.optimal_ranges.items():
            score = 100
            penalties = 0
            
            for feature, value in kwargs.items():
                if feature in optimal:
                    low, high = optimal[feature]
                    
                    # Calculate penalty based on deviation from optimal range
                    if value < low:
                        deviation = (low - value) / low
                        penalty = deviation * 30  # Max 30% penalty
                    elif value > high:
                        deviation = (value - high) / high
                        penalty = deviation * 30
                    else:
                        penalty = 0
                    
                    penalties += penalty
            
            # Apply penalties
            score = max(0, 100 - penalties)
            scores[crop] = round(score, 1)
        
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    def suggest_improvements(self, **kwargs) -> List[str]:
        """
        Suggest improvements based on input values
        """
        suggestions = []
        
        # Soil nutrient suggestions
        if 'N' in kwargs and kwargs['N'] < 50:
            suggestions.append("Apply nitrogen-rich fertilizer (e.g., urea, DAP)")
        elif 'N' in kwargs and kwargs['N'] > 150:
            suggestions.append("Reduce nitrogen application to prevent leaching")
        
        if 'P' in kwargs and kwargs['P'] < 30:
            suggestions.append("Apply phosphorus fertilizer (e.g., superphosphate)")
        
        if 'K' in kwargs and kwargs['K'] < 40:
            suggestions.append("Apply potash fertilizer (e.g., muriate of potash)")
        
        # pH suggestions
        if 'ph' in kwargs:
            ph = kwargs['ph']
            if ph < 5.5:
                suggestions.append("Apply lime to increase soil pH")
            elif ph > 7.5:
                suggestions.append("Apply sulfur or organic matter to decrease soil pH")
        
        # Organic matter suggestion
        if all(kwargs.get(f, 0) < 30 for f in ['N', 'P', 'K']):
            suggestions.append("Consider adding organic manure to improve overall soil health")
        
        return suggestions


class DataValidator:
    """
    Validate entire datasets
    """
    
    def __init__(self):
        self.validator = InputValidator()
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate entire dataframe
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'total_rows': len(df),
            'valid_rows': 0,
            'invalid_rows': 0,
            'warnings': [],
            'errors': [],
            'column_stats': {}
        }
        
        # Check required columns
        required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Validate each row
        for idx, row in df.iterrows():
            row_valid, row_warnings = self.validator.validate_all(
                N=row['N'], P=row['P'], K=row['K'],
                temperature=row['temperature'],
                humidity=row['humidity'],
                ph=row['ph'],
                rainfall=row['rainfall']
            )
            
            if row_valid:
                results['valid_rows'] += 1
            else:
                results['invalid_rows'] += 1
                results['errors'].append(f"Row {idx}: Invalid values")
            
            results['warnings'].extend([f"Row {idx}: {w}" for w in row_warnings])
        
        # Column statistics
        for col in required_cols[:-1]:  # Exclude 'crop'
            results['column_stats'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'missing': int(df[col].isnull().sum()),
                'outliers': int(self._count_outliers(df[col]))
            }
        
        # Crop distribution
        results['crop_distribution'] = df['crop'].value_counts().to_dict()
        
        results['validity_percentage'] = (results['valid_rows'] / results['total_rows']) * 100
        
        return results
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def generate_validation_report(self, df: pd.DataFrame) -> str:
        """Generate human-readable validation report"""
        results = self.validate_dataframe(df)
        
        report = []
        report.append("="*60)
        report.append("DATA VALIDATION REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall status
        status = "✅ PASSED" if results['is_valid'] else "❌ FAILED"
        report.append(f"Status: {status}")
        report.append(f"Total Rows: {results['total_rows']}")
        report.append(f"Valid Rows: {results['valid_rows']} ({results['validity_percentage']:.1f}%)")
        report.append(f"Invalid Rows: {results['invalid_rows']}")
        report.append("")
        
        # Column statistics
        report.append("COLUMN STATISTICS:")
        report.append("-"*40)
        for col, stats in results['column_stats'].items():
            report.append(f"\n{col}:")
            report.append(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
            report.append(f"  Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
            report.append(f"  Missing: {stats['missing']}")
            report.append(f"  Outliers: {stats['outliers']}")
        
        # Crop distribution
        report.append("\nCROP DISTRIBUTION:")
        report.append("-"*40)
        for crop, count in results['crop_distribution'].items():
            percentage = (count / results['total_rows']) * 100
            report.append(f"  {crop}: {count} ({percentage:.1f}%)")
        
        # Warnings
        if results['warnings']:
            report.append("\nWARNINGS:")
            report.append("-"*40)
            for warning in results['warnings'][:10]:  # Show first 10
                report.append(f"  ⚠ {warning}")
            if len(results['warnings']) > 10:
                report.append(f"  ... and {len(results['warnings']) - 10} more")
        
        # Errors
        if results['errors']:
            report.append("\nERRORS:")
            report.append("-"*40)
            for error in results['errors'][:5]:  # Show first 5
                report.append(f"  ❌ {error}")
            if len(results['errors']) > 5:
                report.append(f"  ... and {len(results['errors']) - 5} more")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)