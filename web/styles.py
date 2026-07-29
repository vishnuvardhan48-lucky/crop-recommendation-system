"""
Professional CSS Styles for Crop Recommendation System
"""

import streamlit as st

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #43A047 100%);
            padding: 3rem;
            border-radius: 30px;
            margin: 2rem 0 3rem 0;
            color: white;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            animation: slideDown 0.8s ease-out;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }
        
        .main-header h1 {
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -1px;
            position: relative;
            z-index: 1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }
        
        .main-header .subtitle {
            font-size: 1.3rem;
            opacity: 0.95;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
            font-weight: 300;
            animation: fadeInUp 0.8s ease-out 0.4s both;
        }
        
        .header-badges {
            margin-top: 2rem;
            position: relative;
            z-index: 1;
            animation: fadeInUp 0.8s ease-out 0.6s both;
        }
        
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            margin: 0 0.5rem;
            font-size: 0.9rem;
            font-weight: 500;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
            display: inline-block;
            transition: all 0.3s;
        }
        
        .badge:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        /* Sidebar Styles */
        .sidebar-header {
            text-align: center;
            padding: 2rem 1rem;
            background: linear-gradient(135deg, #2E7D32, #43A047);
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
        }
        
        .sidebar-header h2 {
            margin: 0;
            font-weight: 600;
        }
        
        /* Navigation Buttons */
        .stButton > button {
            background: white;
            color: #333;
            font-weight: 500;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
            margin: 0.25rem 0;
            text-align: left;
        }
        
        .stButton > button:hover {
            background: #2E7D32;
            color: white;
            border-color: #2E7D32;
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(46,125,50,0.3);
        }
        
        /* Recommendation Card */
        .recommendation-card {
            background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
            color: white;
            padding: 3rem;
            border-radius: 30px;
            margin: 2rem 0;
            text-align: center;
            box-shadow: 0 30px 60px rgba(46,125,50,0.3);
            animation: scaleIn 0.5s ease-out;
        }
        
        .recommendation-header {
            margin-bottom: 2rem;
        }
        
        .recommendation-header .emoji {
            font-size: 4rem;
            display: block;
            animation: bounce 2s infinite;
        }
        
        .recommendation-header .title {
            font-size: 1.5rem;
            opacity: 0.9;
        }
        
        .crop-name {
            font-size: 5rem;
            font-weight: 800;
            margin: 1rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            letter-spacing: -2px;
        }
        
        .confidence-meter {
            width: 80%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            margin: 2rem auto;
            overflow: hidden;
        }
        
        .confidence-bar {
            height: 100%;
            background: linear-gradient(90deg, #ffd700, #ffed4e);
            border-radius: 10px;
            transition: width 1s ease-out;
        }
        
        .confidence-text {
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        /* Top Cards */
        .top-card {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
            position: relative;
            height: 100%;
        }
        
        .top-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(46,125,50,0.2);
            border-color: #2E7D32;
        }
        
        .top-1 {
            border: 3px solid gold;
        }
        
        .rank {
            position: absolute;
            top: -15px;
            left: 20px;
            background: #2E7D32;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2rem;
            box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        }
        
        .top-card h3 {
            margin: 1rem 0;
            color: #333;
            font-size: 1.5rem;
        }
        
        .confidence-small {
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        .confidence-bar-small {
            height: 100%;
            background: linear-gradient(90deg, #2E7D32, #43A047);
            transition: width 1s ease-out;
        }
        
        /* Info Cards */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            margin: 1rem 0;
            transition: all 0.3s;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(46,125,50,0.1);
            border-color: #2E7D32;
        }
        
        .info-card h4 {
            color: #2E7D32;
            margin: 0 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .info-card ul {
            margin: 0;
            padding-left: 1.5rem;
        }
        
        .info-card li {
            margin: 0.5rem 0;
            color: #555;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            transition: all 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(46,125,50,0.15);
        }
        
        .metric-card h3 {
            color: #2E7D32;
            font-size: 2rem;
            margin: 0.5rem 0;
        }
        
        /* Footer */
        .footer {
            margin-top: 4rem;
            padding: 3rem 2rem 1rem;
            background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
            color: white;
            border-radius: 30px 30px 0 0;
        }
        
        .footer-content {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .footer-section h4 {
            color: #81C784;
            margin-bottom: 1rem;
        }
        
        .footer-section p {
            margin: 0.5rem 0;
            color: #ccc;
        }
        
        .footer-section a {
            color: #81C784;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .footer-section a:hover {
            color: #A5D6A7;
            text-decoration: underline;
        }
        
        .footer-bottom {
            text-align: center;
            padding-top: 2rem;
            border-top: 1px solid #444;
            color: #999;
        }
        
        /* Animations */
        @keyframes slideDown {
            from {
                transform: translateY(-50px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeInUp {
            from {
                transform: translateY(30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes scaleIn {
            from {
                transform: scale(0.9);
                opacity: 0;
            }
            to {
                transform: scale(1);
                opacity: 1;
            }
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        /* Loading Animation */
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #e0e0e0;
            border-top: 5px solid #2E7D32;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 2rem auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .crop-name {
                font-size: 3rem;
            }
            
            .footer-content {
                grid-template-columns: 1fr;
            }
            
            .badge {
                display: block;
                margin: 0.5rem;
            }
        }
        
        /* Dark Mode Support */
        @media (prefers-color-scheme: dark) {
            .top-card, .info-card, .metric-card {
                background: #2d2d2d;
                color: white;
                border-color: #444;
            }
            
            .top-card h3, .info-card h4 {
                color: #81C784;
            }
            
            .info-card p, .info-card li {
                color: #ccc;
            }
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #2E7D32;
            border-radius: 5px;
            transition: all 0.3s;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #1B5E20;
        }
        
        /* Success/Error Messages */
        .stAlert {
            border-radius: 12px;
            border: none;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: slideDown 0.3s ease-out;
        }
        
        /* Progress Bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #2E7D32, #43A047);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: white;
            padding: 0.5rem;
            border-radius: 50px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 30px;
            padding: 0.75rem 2rem;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stTabs [aria-selected="true"] {
            background: #2E7D32 !important;
            color: white !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: white;
            border-radius: 12px;
            font-weight: 500;
            border: 1px solid #e0e0e0;
            transition: all 0.3s;
        }
        
        .streamlit-expanderHeader:hover {
            background: #E8F5E9;
            border-color: #2E7D32;
        }
        </style>
    """, unsafe_allow_html=True)