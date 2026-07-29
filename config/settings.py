"""
Enterprise Configuration Management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Application
    APP_NAME: str = Field(default="Enterprise Crop Recommendation System")
    APP_VERSION: str = Field(default="3.0.0")
    ENVIRONMENT: str = Field(default="production")
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(default="change-this-in-production")
    
    # Model paths
    MODEL_PATH: Path = Field(default=BASE_DIR / "models" / "saved_models" / "best_model.pkl")
    SCALER_PATH: Path = Field(default=BASE_DIR / "models" / "saved_models" / "scaler.pkl")
    ENCODER_PATH: Path = Field(default=BASE_DIR / "models" / "saved_models" / "label_encoder.pkl")
    METADATA_PATH: Path = Field(default=BASE_DIR / "models" / "saved_models" / "model_metadata.json")
    FEATURE_ENGINEER_PATH: Path = Field(default=BASE_DIR / "models" / "saved_models" / "feature_engineer.pkl")
    
    # Data paths
    DATA_PATH: Path = Field(default=BASE_DIR / "data" / "datasets" / "crop_data.csv")
    TRAIN_TEST_SPLIT: float = Field(default=0.2, ge=0.0, le=1.0)
    VALIDATION_SPLIT: float = Field(default=0.1, ge=0.0, le=1.0)
    RANDOM_SEED: int = Field(default=42, ge=0)
    SAMPLES_PER_CROP: int = Field(default=500, ge=100, le=10000)
    
    # Model parameters
    N_ESTIMATORS: int = Field(default=200, ge=10, le=1000)
    MAX_DEPTH: int = Field(default=20, ge=1, le=100)
    MIN_SAMPLES_SPLIT: int = Field(default=5, ge=2)
    MIN_SAMPLES_LEAF: int = Field(default=2, ge=1)
    
    # API settings
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1024, le=65535)
    API_WORKERS: int = Field(default=4, ge=1, le=32)
    
    # Web settings
    WEB_HOST: str = Field(default="0.0.0.0")
    WEB_PORT: int = Field(default=8501, ge=1024, le=65535)
    
    # Database
    DATABASE_URL: Optional[str] = Field(default=None)
    
    # Redis Cache
    REDIS_URL: Optional[str] = Field(default=None)
    CACHE_TTL: int = Field(default=3600, ge=0)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: Path = Field(default=BASE_DIR / "logs" / "app.log")
    LOG_FORMAT: str = Field(default="json")
    LOG_ROTATION: str = Field(default="1 day")
    LOG_RETENTION: str = Field(default="30 days")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None)
    NEW_RELIC_LICENSE_KEY: Optional[str] = Field(default=None)
    
    # Email
    SMTP_HOST: Optional[str] = Field(default=None)
    SMTP_PORT: Optional[int] = Field(default=None)
    SMTP_USER: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    ALERT_EMAIL: Optional[str] = Field(default=None)
    
    # Feature flags
    ENABLE_CACHING: bool = Field(default=True)
    ENABLE_MONITORING: bool = Field(default=True)
    ENABLE_AUTH: bool = Field(default=False)
    ENABLE_API: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        arbitrary_types_allowed = True
        
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in allowed:
            raise ValueError(f'LOG_LEVEL must be one of {allowed}')
        return v
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.BASE_DIR / "logs",
            self.BASE_DIR / "data" / "datasets",
            self.BASE_DIR / "models" / "saved_models",
            self.BASE_DIR / "reports",
            self.BASE_DIR / "temp",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'n_estimators': self.N_ESTIMATORS,
            'max_depth': self.MAX_DEPTH,
            'min_samples_split': self.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': self.MIN_SAMPLES_LEAF,
            'random_state': self.RANDOM_SEED,
            'n_jobs': -1
        }
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return {
            'data_path': self.DATA_PATH,
            'test_size': self.TRAIN_TEST_SPLIT,
            'val_size': self.VALIDATION_SPLIT,
            'random_state': self.RANDOM_SEED
        }
        
    def is_production(self) -> bool:
        """Check if environment is production"""
        return self.ENVIRONMENT.lower() == 'production'
        
    def is_development(self) -> bool:
        """Check if environment is development"""
        return self.ENVIRONMENT.lower() == 'development'
        
    def is_testing(self) -> bool:
        """Check if environment is testing"""
        return self.ENVIRONMENT.lower() == 'testing'
        
    def get_database_url(self) -> str:
        """Get database URL with fallback"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"sqlite:///{self.BASE_DIR}/crop_system.db"
        
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL"""
        return self.REDIS_URL
        
    def get_sentry_dsn(self) -> Optional[str]:
        """Get Sentry DSN"""
        return self.SENTRY_DSN if self.is_production() else None
        
    def as_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.dict().items()
            if not k.startswith('_')
        }
        
    def save_to_file(self, path: Optional[Path] = None):
        """Save settings to file"""
        if path is None:
            path = self.BASE_DIR / "config" / "settings.json"
        with open(path, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

# Global settings instance
settings = Settings().create_directories()