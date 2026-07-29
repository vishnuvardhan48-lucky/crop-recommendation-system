"""
Enterprise Streamlit Web Application
Complete professional implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
from pathlib import Path
import hashlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.predict import PredictionService
from config.settings import settings
from config.logging_config import get_logger
from web.components import (
    render_header,
    render_sidebar,
    render_input_form,
    render_recommendation,
    render_crop_info,
    render_analytics,
    render_footer,
    render_historical_predictions
)
from web.styles import load_css

# Initialize logger
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/crop-recommendation',
        'Report a bug': 'https://github.com/your-repo/crop-recommendation/issues',
        'About': '# 🌱 AI-Powered Crop Recommendation System\n\nEnterprise-grade agricultural intelligence platform.'
    }
)

# Load custom CSS
load_css()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.prediction_history = []
    st.session_state.model_loaded = False
    st.session_state.current_page = "Home"
    st.session_state.theme = "light"
    st.session_state.favorites = []
    st.session_state.settings = {
        'show_advanced': False,
        'confidence_threshold': 0.5,
        'num_recommendations': 3,
        'show_probabilities': True,
        'dark_mode': False
    }

# Load model
@st.cache_resource(ttl=3600)
def load_prediction_service():
    """Load prediction service with caching"""
    service = PredictionService()
    if service.load_models():
        logger.info("Model loaded successfully")
        return service
    return None

# Initialize service
prediction_service = load_prediction_service()

# Main app logic
def main():
    """Main application entry point"""
    
    # Render header
    render_header()
    
    # Check model status
    if prediction_service is None:
        st.error("""
            ## ❌ Model Not Loaded
            
            The AI model could not be loaded. Please ensure:
            1. Model files exist in the `models/saved_models` directory
            2. You have run the training script: `python run.py --train`
            3. The model files are not corrupted
            
            For immediate setup, run:
            ```bash
            python run.py --train