"""
API module for Crop Recommendation System
"""

from .routes import app
from .middleware import setup_middleware

__all__ = ['app', 'setup_middleware']