"""
Enterprise UI Components for Crop Recommendation System
Reusable professional components
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any, List, Optional

def render_header():
    """Render professional header"""
    st.markdown("""
        <div class="main-header">
            <h1>🌱 AI-Powered Crop Recommendation System</h1>
            <p class="subtitle">Enterprise-Grade Agricultural Intelligence Platform</p>
            <div class="header-badges">
                <span class="badge">v3.0.0</span>
                <span class="badge">Production Ready</span>
                <span class="badge">20+ Crops</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar(service):
    """Render professional sidebar"""
    with st.sidebar:
        # Logo and title
        st.markdown("""
            <div class="sidebar-header">
                <h2>🌾 Control Panel</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### 📍 Navigation")
        pages = {
            "Home": "🏠",
            "Analytics": "📊",
            "History": "📜",
            "About": "ℹ️",
            "Settings": "⚙️"
        }
        
        for page, icon in pages.items():
            if st.button(
                f"{icon} {page}",
                key=f"nav_{page}",
                use_container_width=True,
                help=f"Go to {page} page"
            ):
                st.session_state.current_page = page
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🤖 Model Status")
        info = service.get_model_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Online")
        with col2:
            st.metric("Model", info.get('model_name', 'Random Forest'))
        
        st.metric("Accuracy", f"{info.get('accuracy', 0.95):.1%}")
        st.metric("Predictions", info.get('prediction_count', 0))
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Model", use_container_width=True):
            st.cache_resource.clear()
            st.success("Model cache cleared!")
            st.experimental_rerun()
        
        if st.button("📥 Export Data", use_container_width=True):
            if st.session_state.prediction_history:
                df = pd.DataFrame(st.session_state.prediction_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        
        # System info
        st.markdown("### ℹ️ System Info")
        st.markdown(f"**Version:** {st.session_state.get('version', '3.0.0')}")
        st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**Session:** {st.session_state.get('session_id', 'N/A')}")

def render_input_form() -> Dict[str, float]:
    """Render professional input form"""
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "📊 Quick Presets"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            N = st.number_input(
                "🌱 Nitrogen (N) kg/ha",
                min_value=0.0, max_value=200.0, value=90.0, step=1.0,
                help="Nitrogen content in soil (0-200 kg/ha)",
                key="input_N"
            )
            P = st.number_input(
                "🌱 Phosphorus (P) kg/ha",
                min_value=0.0, max_value=200.0, value=45.0, step=1.0,
                help="Phosphorus content in soil (0-200 kg/ha)",
                key="input_P"
            )
            K = st.number_input(
                "🌱 Potassium (K) kg/ha",
                min_value=0.0, max_value=200.0, value=40.0, step=1.0,
                help="Potassium content in soil (0-200 kg/ha)",
                key="input_K"
            )
        
        with col2:
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=0.0, max_value=50.0, value=25.0, step=0.5,
                help="Average temperature during growing season",
                key="input_temp"
            )
            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0, max_value=100.0, value=70.0, step=1.0,
                help="Relative humidity percentage",
                key="input_humidity"
            )
            ph = st.number_input(
                "🧪 Soil pH",
                min_value=0.0, max_value=14.0, value=6.5, step=0.1,
                help="Soil pH level (0-14)",
                key="input_ph"
            )
            rainfall = st.number_input(
                "☔ Rainfall (mm)",
                min_value=0.0, max_value=300.0, value=150.0, step=5.0,
                help="Annual rainfall in mm",
                key="input_rainfall"
            )
    
    with tab2:
        st.markdown("##### Select a crop to auto-fill optimal parameters:")
        
        presets = {
            "🌾 Rice": {"N": 90, "P": 45, "K": 40, "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 150},
            "🌽 Maize": {"N": 85, "P": 42, "K": 38, "temperature": 24, "humidity": 72, "ph": 6.3, "rainfall": 145},
            "🌾 Wheat": {"N": 75, "P": 35, "K": 35, "temperature": 18, "humidity": 65, "ph": 6.2, "rainfall": 75},
            "🌱 Cotton": {"N": 95, "P": 50, "K": 45, "temperature": 28, "humidity": 65, "ph": 6.8, "rainfall": 85},
            "🎋 Sugarcane": {"N": 110, "P": 55, "K": 55, "temperature": 30, "humidity": 75, "ph": 6.5, "rainfall": 180},
            "🥜 Groundnut": {"N": 30, "P": 50, "K": 35, "temperature": 25, "humidity": 68, "ph": 6.2, "rainfall": 65}
        }
        
        cols = st.columns(2)
        for i, (crop, values) in enumerate(presets.items()):
            with cols[i % 2]:
                if st.button(crop, use_container_width=True, key=f"preset_{i}"):
                    for key, val in values.items():
                        st.session_state[f"input_{key}"] = val
                    st.experimental_rerun()
    
    # Store inputs in session state
    inputs = {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    
    st.session_state.inputs = inputs
    
    return inputs

def render_recommendation(result: Dict[str, Any]):
    """Render professional recommendation card"""
    
    # Main recommendation
    st.markdown(f"""
        <div class="recommendation-card">
            <div class="recommendation-header">
                <span class="emoji">🌾</span>
                <span class="title">AI Recommendation</span>
            </div>
            <div class="recommendation-content">
                <h1 class="crop-name">{result['primary_recommendation']}</h1>
                <div class="confidence-meter">
                    <div class="confidence-bar" style="width: {result['confidence']}%;"></div>
                </div>
                <p class="confidence-text">Confidence: {result['confidence']:.1f}%</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Top 3 recommendations
    st.markdown("### 🏆 Top 3 Recommendations")
    cols = st.columns(3)
    
    for i, rec in enumerate(result.get('top_3', [])):
        with cols[i]:
            st.markdown(f"""
                <div class="top-card {'top-1' if i == 0 else ''}">
                    <div class="rank">#{i+1}</div>
                    <h3>{rec['crop']}</h3>
                    <div class="confidence-small">
                        <div class="confidence-bar-small" style="width: {rec['confidence']}%;"></div>
                    </div>
                    <p>{rec['confidence']:.1f}% confident</p>
                </div>
            """, unsafe_allow_html=True)

def render_crop_info(crop_name: str):
    """Render professional crop information"""
    
    crop_database = {
        "Rice": {
            "description": "Staple food crop, requires standing water",
            "season": "Kharif (June-October)",
            "duration": "120-150 days",
            "soil_type": "Clay loam",
            "water_requirement": "High (1500-2000 mm)",
            "temperature_range": "22-32°C",
            "varieties": ["Basmati", "IR64", "Pusa", "Sona"],
            "pests": ["Stem borer", "Leaf folder", "Blast"],
            "fertilizer": "NPK 120:60:40 kg/ha"
        },
        "Wheat": {
            "description": "Winter crop, requires cool climate",
            "season": "Rabi (November-April)",
            "duration": "110-130 days",
            "soil_type": "Loam",
            "water_requirement": "Medium (450-650 mm)",
            "temperature_range": "15-25°C",
            "varieties": ["HD 2967", "PBW 343", "WH 542"],
            "pests": ["Aphids", "Rust", "Termites"],
            "fertilizer": "NPK 100:50:40 kg/ha"
        },
        "Maize": {
            "description": "Versatile crop used for food and feed",
            "season": "Kharif",
            "duration": "90-110 days",
            "soil_type": "Well-drained loam",
            "water_requirement": "Medium (500-800 mm)",
            "temperature_range": "20-30°C",
            "varieties": ["HM 4", "HM 5", "Bio 9544"],
            "pests": ["Stem borer", "Armyworm", "Earworm"],
            "fertilizer": "NPK 150:75:60 kg/ha"
        }
    }
    
    info = crop_database.get(crop_name, {
        "description": "Suitable crop for given conditions",
        "season": "Varies by region",
        "duration": "Varies",
        "soil_type": "Well-drained soil",
        "water_requirement": "Moderate",
        "temperature_range": "15-35°C",
        "varieties": ["Contact local agricultural office"],
        "pests": ["Monitor regularly"],
        "fertilizer": "Soil test recommended"
    })
    
    st.markdown(f"### 📋 {crop_name} Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="info-card">
                <h4>🌱 Basic Info</h4>
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Season:</strong> {info['season']}</p>
                <p><strong>Duration:</strong> {info['duration']}</p>
                <p><strong>Soil Type:</strong> {info['soil_type']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-card">
                <h4>🌡️ Requirements</h4>
                <p><strong>Temperature:</strong> {info['temperature_range']}</p>
                <p><strong>Water:</strong> {info['water_requirement']}</p>
                <p><strong>Fertilizer:</strong> {info['fertilizer']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="info-card">
                <h4>🌾 Popular Varieties</h4>
                <ul>
                    {''.join([f'<li>{v}</li>' for v in info['varieties'][:5]])}
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-card">
                <h4>🐛 Common Pests</h4>
                <ul>
                    {''.join([f'<li>{p}</li>' for p in info['pests'][:5]])}
                </ul>
            </div>
        """, unsafe_allow_html=True)

def render_analytics(result: Dict[str, Any]):
    """Render professional analytics"""
    
    st.markdown("### 📊 Detailed Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Probabilities", "🎯 Confidence", "📊 Comparison"])
    
    with tab1:
        # Probability distribution chart
        probs = result.get('all_probabilities', {})
        df_probs = pd.DataFrame({
            'Crop': list(probs.keys()),
            'Confidence': list(probs.values())
        }).sort_values('Confidence', ascending=True)
        
        fig = px.bar(
            df_probs,
            x='Confidence',
            y='Crop',
            orientation='h',
            title='Model Confidence by Crop',
            color='Confidence',
            color_continuous_scale='Greens',
            range_color=[0, 100]
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        fig.update_xaxes(title="Confidence (%)", gridcolor='#e0e0e0')
        fig.update_yaxes(title="", gridcolor='#e0e0e0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Confidence gauge for top crop
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['confidence'],
            title={'text': f"Confidence for {result['primary_recommendation']}"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2E7D32"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#ffffcc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Radar chart comparing with optimal values
        categories = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        # Get current values
        current = list(st.session_state.inputs.values())
        
        # Get optimal values for recommended crop
        optimal_map = {
            'Rice': [90, 45, 40, 25, 70, 6.5, 150],
            'Wheat': [75, 35, 35, 18, 65, 6.2, 75],
            'Maize': [85, 42, 38, 24, 72, 6.3, 145],
            'Cotton': [95, 50, 45, 28, 65, 6.8, 85]
        }
        
        optimal = optimal_map.get(result['primary_recommendation'], current)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=current,
            theta=categories,
            fill='toself',
            name='Current',
            line_color='#2E7D32'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=optimal,
            theta=categories,
            fill='toself',
            name='Optimal',
            line_color='#FFA500'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(current), max(optimal)) * 1.1]
                )),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_footer():
    """Render professional footer"""
    st.markdown("---")
    st.markdown(f"""
        <div class="footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h4>🌱 Crop Recommendation System</h4>
                    <p>Enterprise Edition v3.0.0</p>
                    <p>© 2024 Agricultural Intelligence Platform</p>
                </div>
                <div class="footer-section">
                    <h4>Quick Links</h4>
                    <p><a href="#">Documentation</a></p>
                    <p><a href="#">API Reference</a></p>
                    <p><a href="#">Support</a></p>
                </div>
                <div class="footer-section">
                    <h4>Connect</h4>
                    <p>📧 support@agri-intelligence.com</p>
                    <p>📞 +1 (800) 123-4567</p>
                    <p>🌐 www.agri-intelligence.com</p>
                </div>
            </div>
            <div class="footer-bottom">
                <p>Made with ❤️ for farmers worldwide | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_historical_predictions(history: List[Dict]):
    """Render historical predictions"""
    
    if not history:
        st.info("No predictions yet. Make your first prediction!")
        return
    
    st.markdown("### 📜 Recent Predictions")
    
    for i, record in enumerate(reversed(history[-10:])):
        with st.expander(f"📊 Prediction {len(history)-i}: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input Parameters:**")
                for key, value in record['inputs'].items():
                    st.markdown(f"- {key}: {value}")
            
            with col2:
                st.markdown("**Result:**")
                st.markdown(f"- **Crop:** {record['result']['primary_recommendation']}")
                st.markdown(f"- **Confidence:** {record['result']['confidence']:.1f}%")
                
                if 'warnings' in record['result'] and record['result']['warnings']:
                    st.markdown("**Warnings:**")
                    for w in record['result']['warnings'][:3]:
                        st.markdown(f"- ⚠️ {w}")