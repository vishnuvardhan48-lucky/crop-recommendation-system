"""
Enterprise Helper Functions
"""

import json
import yaml
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import re
import random
import string
from pathlib import Path

def generate_id(prefix: str = "") -> str:
    """Generate unique ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{random_str}" if prefix else f"{timestamp}_{random_str}"

def hash_string(s: str) -> str:
    """Create hash of string"""
    return hashlib.sha256(s.encode()).hexdigest()

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """Validate phone number"""
    pattern = r'^\+?1?\d{9,15}$'
    return bool(re.match(pattern, phone))

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency"""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
    symbol = symbols.get(currency, "$")
    return f"{symbol}{amount:,.2f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage"""
    return f"{value:.{decimals}f}%"

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with commas"""
    return f"{value:,.{decimals}f}"

def parse_json_file(filepath: Path) -> Dict:
    """Parse JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_yaml_file(filepath: Path) -> Dict:
    """Parse YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_json_file(data: Dict, filepath: Path):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def dataframe_to_json(df: pd.DataFrame, orient: str = 'records') -> List:
    """Convert DataFrame to JSON"""
    return df.to_dict(orient=orient)

def json_to_dataframe(data: List) -> pd.DataFrame:
    """Convert JSON to DataFrame"""
    return pd.DataFrame(data)

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return a / b if b != 0 else default

def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]

def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """Remove outliers from DataFrame"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < 3]
    return df

def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize features to [0, 1] range"""
    result = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            result[col] = (df[col] - min_val) / (max_val - min_val)
    return result

def standardize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Standardize features (mean=0, std=1)"""
    result = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            result[col] = (df[col] - mean) / std
    return result

def get_file_size(filepath: Path) -> str:
    """Get human readable file size"""
    size = filepath.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def retry_function(func: callable, max_retries: int = 3, delay: float = 1.0):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))

class Timer:
    """Context manager for timing code"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            logger.info(f"{self.name} took {self.elapsed:.3f}s")
    
    def get_elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0