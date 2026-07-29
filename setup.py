#!/usr/bin/env python
"""
Setup script for Crop Recommendation System
"""

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