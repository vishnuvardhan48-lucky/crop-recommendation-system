# create_all_files.ps1
# This script creates all 47 files for the Crop Recommendation System

Write-Host "="*60 -ForegroundColor Green
Write-Host "🌱 CREATING ALL CROP RECOMMENDATION SYSTEM FILES" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

# Function to create file with content
function Create-File {
    param(
        [string]$Path,
        [string]$Content
    )
    
    $fullPath = Join-Path $PWD.Path $Path
    $directory = Split-Path $fullPath -Parent
    
    if (!(Test-Path $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    
    Set-Content -Path $fullPath -Value $Content -Encoding UTF8
    Write-Host "  ✅ Created: $Path" -ForegroundColor Green
}

# ============================================================================
# FILE 1: .env
# ============================================================================
Create-File -Path ".env" -Content @"
# Application Settings
APP_NAME="Enterprise Crop Recommendation System"
APP_VERSION="3.0.0"
ENVIRONMENT="production"
DEBUG=false
SECRET_KEY="your-secret-key-here-change-in-production"

# Model Settings
MODEL_PATH="models/saved_models/best_model.pkl"
SCALER_PATH="models/saved_models/scaler.pkl"
ENCODER_PATH="models/saved_models/label_encoder.pkl"
METADATA_PATH="models/saved_models/model_metadata.json"
FEATURE_ENGINEER_PATH="models/saved_models/feature_engineer.pkl"

# Data Settings
DATA_PATH="data/datasets/crop_data.csv"
TRAIN_TEST_SPLIT=0.2
VALIDATION_SPLIT=0.1
RANDOM_SEED=42
SAMPLES_PER_CROP=500

# Model Parameters
N_ESTIMATORS=200
MAX_DEPTH=20
MIN_SAMPLES_SPLIT=5
MIN_SAMPLES_LEAF=2

# API Settings
API_HOST="0.0.0.0"
API_PORT=8000
API_WORKERS=4

# Web Settings
WEB_HOST="0.0.0.0"
WEB_PORT=8501

# Database
DATABASE_URL="sqlite:///crop_system.db"

# Redis Cache
REDIS_URL="redis://localhost:6379/0"
CACHE_TTL=3600

# Logging
LOG_LEVEL="INFO"
LOG_FILE="logs/app.log"
LOG_FORMAT="json"
LOG_ROTATION="1 day"
LOG_RETENTION="30 days"

# Monitoring
SENTRY_DSN=""
NEW_RELIC_LICENSE_KEY=""

# Email
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER=""
SMTP_PASSWORD=""
ALERT_EMAIL="admin@cropsystem.com"

# Feature Flags
ENABLE_CACHING=true
ENABLE_MONITORING=true
ENABLE_AUTH=false
ENABLE_API=true
"@

# ============================================================================
# FILE 2: .gitignore
# ============================================================================
Create-File -Path ".gitignore" -Content @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
pythonenv.*

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Installer logs
pip-log.txt
pip-delete-this-directory.txt
pip-selfcheck.json

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
logs/
*.log
*.pid
*.seed
*.pid.lock

# Data files
*.csv
*.xlsx
*.parquet
!data/datasets/crop_data.csv

# Model files
*.pkl
*.h5
*.joblib
!models/saved_models/example_model.pkl

# Database
*.db
*.sqlite3

# Secrets
*.key
*.pem
*.crt
secrets/

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# OS
.DS_Store
Thumbs.db
"@

# ============================================================================
# FILE 3: requirements.txt
# ============================================================================
Create-File -Path "requirements.txt" -Content @"
# Core Data Science
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.10.1

# Machine Learning
xgboost==1.7.6
lightgbm==4.0.0
catboost==1.2

# Model Serialization
joblib==1.3.2
cloudpickle==2.2.1

# Web Framework
streamlit==1.28.1
fastapi==0.100.0
uvicorn==0.23.1
python-multipart==0.0.6

# Visualization
plotly==5.17.0
matplotlib==3.7.2
seaborn==0.12.2
wordcloud==1.9.2

# Database
sqlalchemy==2.0.19
alembic==1.11.1
psycopg2-binary==2.9.7

# Caching
redis==4.6.0
cachetools==5.3.1

# Configuration
python-dotenv==1.0.0
pydantic==2.3.0
pydantic-settings==2.0.3

# Logging
loguru==0.7.2
structlog==23.1.0
python-json-logger==2.0.7

# Monitoring
prometheus-client==0.17.1
sentry-sdk==1.28.1

# Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.0
pytest-xdist==3.3.1

# Development
black==23.7.0
isort==5.12.0
flake8==6.1.0
mypy==1.5.0
pre-commit==3.3.3

# API Documentation
mkdocs==1.5.2
mkdocstrings==0.22.0

# Utilities
tqdm==4.65.0
click==8.1.6
pyyaml==6.0
requests==2.31.0
aiohttp==3.8.5

# Data Validation
pandera==0.16.1
great-expectations==0.17.0

# Feature Engineering
feature-engine==1.5.2
category-encoders==2.6.1

# Model Interpretation
shap==0.42.1
eli5==0.13.0
lime==0.2.0.1

# Parallel Processing
dask==2023.7.1
ray==2.6.1

# Email
sendgrid==6.10.0
"@

# ============================================================================
# FILE 4: setup.py
# ============================================================================
Create-File -Path "setup.py" -Content @"
#!/usr/bin/env python
\"\"\"
Setup script for Crop Recommendation System
\"\"\"

from setuptools import setup, find_packages
import os

# Read version from environment or set default
__version__ = os.environ.get("APP_VERSION", "3.0.0")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="crop-recommendation-system",
    version=__version__,
    author="Agricultural Intelligence Team",
    author_email="engineering@agri-intelligence.com",
    description="Enterprise AI-Powered Crop Recommendation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agri-intelligence/crop-recommendation",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Agriculture Industry",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Agriculture :: Precision Farming",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "cudatoolkit>=11.0",
            "cuml>=23.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crop-train=crop_system.cli:train",
            "crop-predict=crop_system.cli:predict",
            "crop-serve=crop_system.cli:serve",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
"@

# ============================================================================
# FILE 5: README.md
# ============================================================================
Create-File -Path "README.md" -Content @"
# 🌱 Enterprise AI Crop Recommendation System

## 🎯 Overview
An enterprise-grade machine learning system that provides intelligent crop recommendations based on soil parameters and environmental conditions. Built with production-ready code, comprehensive error handling, and a beautiful user interface.

## ✨ Features
- **🤖 Advanced ML Models**: Random Forest, XGBoost, LightGBM, SVM, KNN with hyperparameter tuning
- **📊 Real-time Predictions**: Instant crop recommendations with confidence scores
- **🎨 Beautiful UI**: Professional Streamlit interface with custom CSS
- **📈 Comprehensive Analytics**: Feature importance, confusion matrices, performance metrics
- **🔄 Caching**: Intelligent caching for repeated predictions
- **📝 Logging**: Professional logging with rotation and structured output
- **🔧 Configuration**: Environment-based configuration management
- **🧪 Testing**: Comprehensive test suite
- **📚 Documentation**: Full API documentation

## 🏗️ Architecture
