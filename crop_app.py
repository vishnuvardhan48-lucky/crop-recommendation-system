import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import requests
from datetime import datetime
import time
import json
import indiapins

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="2026 AI Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { background: linear-gradient(135deg, #1B5E20 0%, #43A047 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white; text-align: center; }
    .recommendation-box { background: linear-gradient(135deg, #1B5E20 0%, #66BB6A 100%); color: white; padding: 35px; border-radius: 20px; text-align: center; margin: 25px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
    .crop-tag { background: #E8F5E9; padding: 6px 12px; border-radius: 20px; font-size: 13px; border: 1px solid #A5D6A7; display: inline-block; margin: 3px; }
    .feature-card { background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #2E7D32; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .weather-box { background: #E3F2FD; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #1976D2; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1 style="font-size:44px;">🌾 AI Crop Recommendation System</h1><p style="font-size:18px;">2026 AgTech – Smart Farming with AI</p></div>', unsafe_allow_html=True)

# ============================================================
# CROP DATABASE (39 Crops)
# ============================================================
CROP_DB = {
    'Rice': {'N': (80,120), 'P': (40,60), 'K': (40,60), 'temp': (22,32), 'humidity': (70,90), 'ph': (5.5,6.5), 'rainfall': (150,250)},
    'Wheat': {'N': (60,100), 'P': (30,50), 'K': (30,50), 'temp': (15,25), 'humidity': (60,80), 'ph': (6.0,7.0), 'rainfall': (50,100)},
    'Maize': {'N': (70,110), 'P': (30,60), 'K': (30,60), 'temp': (20,30), 'humidity': (65,85), 'ph': (5.8,7.0), 'rainfall': (80,150)},
    'Barley': {'N': (50,90), 'P': (30,50), 'K': (30,50), 'temp': (12,22), 'humidity': (60,75), 'ph': (6.0,7.5), 'rainfall': (40,80)},
    'Millet': {'N': (30,60), 'P': (20,40), 'K': (20,40), 'temp': (25,35), 'humidity': (50,70), 'ph': (5.5,7.5), 'rainfall': (30,60)},
    'Chickpea': {'N': (20,40), 'P': (40,60), 'K': (30,50), 'temp': (15,25), 'humidity': (60,75), 'ph': (6.0,7.0), 'rainfall': (40,70)},
    'Pigeonpea': {'N': (20,40), 'P': (40,60), 'K': (30,50), 'temp': (20,30), 'humidity': (60,80), 'ph': (5.5,7.0), 'rainfall': (60,100)},
    'Lentil': {'N': (20,40), 'P': (30,50), 'K': (30,50), 'temp': (15,22), 'humidity': (60,75), 'ph': (6.0,7.0), 'rainfall': (40,70)},
    'Mungbean': {'N': (20,40), 'P': (30,50), 'K': (30,50), 'temp': (20,30), 'humidity': (60,80), 'ph': (5.5,7.0), 'rainfall': (50,80)},
    'Groundnut': {'N': (20,40), 'P': (40,70), 'K': (30,50), 'temp': (20,30), 'humidity': (60,80), 'ph': (5.5,7.0), 'rainfall': (50,80)},
    'Soybean': {'N': (20,50), 'P': (40,70), 'K': (30,60), 'temp': (20,30), 'humidity': (65,80), 'ph': (6.0,7.0), 'rainfall': (60,100)},
    'Sunflower': {'N': (40,80), 'P': (30,60), 'K': (40,70), 'temp': (20,30), 'humidity': (60,75), 'ph': (6.0,7.5), 'rainfall': (50,90)},
    'Mustard': {'N': (40,80), 'P': (30,60), 'K': (30,50), 'temp': (10,25), 'humidity': (60,75), 'ph': (5.5,7.0), 'rainfall': (40,70)},
    'Cotton': {'N': (80,130), 'P': (40,70), 'K': (40,70), 'temp': (25,35), 'humidity': (60,80), 'ph': (6.0,7.5), 'rainfall': (70,120)},
    'Jute': {'N': (60,100), 'P': (30,50), 'K': (40,60), 'temp': (25,35), 'humidity': (70,90), 'ph': (5.5,7.0), 'rainfall': (150,250)},
    'Tomato': {'N': (50,90), 'P': (40,70), 'K': (40,70), 'temp': (20,27), 'humidity': (65,80), 'ph': (5.5,7.0), 'rainfall': (50,90)},
    'Onion': {'N': (50,80), 'P': (30,50), 'K': (30,50), 'temp': (13,24), 'humidity': (60,70), 'ph': (6.0,7.0), 'rainfall': (50,80)},
    'Potato': {'N': (60,100), 'P': (40,70), 'K': (50,80), 'temp': (15,25), 'humidity': (70,85), 'ph': (5.0,6.5), 'rainfall': (60,100)},
    'Chilli': {'N': (60,100), 'P': (30,60), 'K': (30,60), 'temp': (20,30), 'humidity': (60,75), 'ph': (6.0,7.0), 'rainfall': (60,120)},
    'Brinjal': {'N': (50,80), 'P': (30,60), 'K': (30,60), 'temp': (22,30), 'humidity': (65,80), 'ph': (6.0,7.0), 'rainfall': (60,100)},
    'Banana': {'N': (80,120), 'P': (30,60), 'K': (80,120), 'temp': (25,35), 'humidity': (75,90), 'ph': (5.5,7.0), 'rainfall': (100,200)},
    'Orange': {'N': (60,100), 'P': (30,60), 'K': (40,80), 'temp': (20,30), 'humidity': (60,75), 'ph': (5.5,6.5), 'rainfall': (100,150)},
    'Apple': {'N': (50,90), 'P': (30,50), 'K': (40,70), 'temp': (10,20), 'humidity': (65,80), 'ph': (5.5,6.5), 'rainfall': (80,120)},
    'Mango': {'N': (60,100), 'P': (30,60), 'K': (40,80), 'temp': (24,30), 'humidity': (60,75), 'ph': (5.5,7.0), 'rainfall': (100,150)},
    'Tea': {'N': (100,150), 'P': (30,50), 'K': (40,60), 'temp': (18,25), 'humidity': (80,90), 'ph': (4.5,5.5), 'rainfall': (150,250)},
    'Coffee': {'N': (80,120), 'P': (30,50), 'K': (40,60), 'temp': (18,25), 'humidity': (70,85), 'ph': (5.0,6.0), 'rainfall': (150,200)},
    'Sugarcane': {'N': (100,150), 'P': (50,80), 'K': (50,80), 'temp': (25,35), 'humidity': (70,90), 'ph': (6.0,7.5), 'rainfall': (150,250)},
    'Turmeric': {'N': (60,100), 'P': (30,60), 'K': (40,70), 'temp': (20,30), 'humidity': (70,85), 'ph': (5.5,7.0), 'rainfall': (100,150)},
    'Ginger': {'N': (60,100), 'P': (30,60), 'K': (40,70), 'temp': (20,30), 'humidity': (75,90), 'ph': (5.5,6.5), 'rainfall': (150,200)},
    'Tobacco': {'N': (80,120), 'P': (40,60), 'K': (60,80), 'temp': (20,30), 'humidity': (60,80), 'ph': (5.5,6.5), 'rainfall': (100,150)},
    'Pumpkin': {'N': (40,70), 'P': (30,50), 'K': (40,60), 'temp': (20,30), 'humidity': (60,80), 'ph': (6.0,7.0), 'rainfall': (80,120)},
    'Drumstick': {'N': (40,60), 'P': (30,50), 'K': (30,50), 'temp': (25,35), 'humidity': (60,80), 'ph': (6.0,7.5), 'rainfall': (80,120)},
    'Papaya': {'N': (60,100), 'P': (40,70), 'K': (50,80), 'temp': (22,30), 'humidity': (65,85), 'ph': (5.5,6.5), 'rainfall': (100,150)},
    'Guava': {'N': (50,80), 'P': (30,60), 'K': (40,70), 'temp': (20,30), 'humidity': (60,80), 'ph': (5.5,7.0), 'rainfall': (80,120)},
    'Coconut': {'N': (60,100), 'P': (30,60), 'K': (60,90), 'temp': (25,35), 'humidity': (70,90), 'ph': (5.5,7.5), 'rainfall': (150,250)},
    'Watermelon': {'N': (40,70), 'P': (30,50), 'K': (50,80), 'temp': (22,30), 'humidity': (60,80), 'ph': (6.0,7.0), 'rainfall': (80,120)},
    'Pineapple': {'N': (40,70), 'P': (30,50), 'K': (50,80), 'temp': (22,30), 'humidity': (70,85), 'ph': (5.0,6.0), 'rainfall': (100,150)},
    'Dragonfruit': {'N': (40,70), 'P': (30,50), 'K': (50,80), 'temp': (22,30), 'humidity': (60,80), 'ph': (6.0,7.0), 'rainfall': (80,120)},
    'Pomegranate': {'N': (50,80), 'P': (30,60), 'K': (40,70), 'temp': (20,30), 'humidity': (60,75), 'ph': (5.5,7.0), 'rainfall': (80,120)}
}

# ============================================================
# WEATHER & GEOCODING FUNCTIONS
# ============================================================
def get_weather_by_location(lat, lon):
    """Fetch real-time weather using Open-Meteo"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "timezone": "auto",
            "forecast_days": 1
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        return {
            'success': True,
            'temperature': current.get("temperature_2m", 25),
            'humidity': current.get("relative_humidity_2m", 70),
            'rainfall': current.get("precipitation", 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_location_details(place):
    """
    Convert a place name or pincode to location details.
    Returns dict with: success, place_name, district, state, latitude, longitude.
    """
    # 1. Try as pincode using indiapins
    if isinstance(place, str) and len(place) == 6 and place.isdigit():
        try:
            records = indiapins.matching(place)
            if records:
                # Get first record with coordinates
                for rec in records:
                    lat = rec.get('Latitude')
                    lon = rec.get('Longitude')
                    if lat and lon:
                        return {
                            'success': True,
                            'place_name': rec.get('Name', place),
                            'district': rec.get('District', ''),
                            'state': rec.get('State', ''),
                            'latitude': float(lat),
                            'longitude': float(lon)
                        }
        except Exception as e:
            pass

    # 2. Fallback: Open‑Meteo Geocoding API (for city names)
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={place}&count=1&language=en&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('results') and len(data['results']) > 0:
            result = data['results'][0]
            return {
                'success': True,
                'place_name': result.get('name', place),
                'district': result.get('admin1', ''),
                'state': result.get('admin1', ''),
                'latitude': result['latitude'],
                'longitude': result['longitude']
            }
    except:
        pass

    # 3. Nothing found
    return {'success': False, 'error': f'Could not locate "{place}". Please check spelling or try a pincode.'}

# ============================================================
# DISEASE PREDICTION (unchanged)
# ============================================================
DISEASE_DB = {
    'Rice': {
        'Blast': {'conditions': {'temp': (22,28), 'humidity': (85,100), 'rainfall': (100,200)}, 
                  'severity': 'High', 'symptoms': 'Leaf spots with gray centers, diamond-shaped lesions',
                  'prevention': 'Use resistant varieties, avoid excess nitrogen',
                  'treatment': 'Apply tricyclazole fungicides'},
        'Blight': {'conditions': {'temp': (25,30), 'humidity': (80,95), 'rainfall': (150,300)},
                  'severity': 'Critical', 'symptoms': 'Wilting, yellowing, bacterial ooze',
                  'prevention': 'Seed treatment, field sanitation',
                  'treatment': 'Apply streptocycline or copper-based bactericides'}
    },
    'Wheat': {
        'Rust': {'conditions': {'temp': (15,22), 'humidity': (70,85), 'rainfall': (50,100)},
                'severity': 'High', 'symptoms': 'Orange-brown pustules on leaves',
                'prevention': 'Use resistant varieties, early sowing',
                'treatment': 'Apply tebuconazole or propiconazole'},
        'Powdery Mildew': {'conditions': {'temp': (15,20), 'humidity': (75,90), 'rainfall': (40,80)},
                          'severity': 'Medium', 'symptoms': 'White powdery spots on upper leaf surfaces',
                          'prevention': 'Avoid dense planting, maintain field hygiene',
                          'treatment': 'Sulfur dusting or triadimefon fungicides'}
    },
    'Cotton': {
        'Bollworm': {'conditions': {'temp': (25,35), 'humidity': (60,80), 'rainfall': (70,120)},
                    'severity': 'Critical', 'symptoms': 'Larvae feed on bolls, flowers, and leaves',
                    'prevention': 'Use Bt cotton, pheromone traps',
                    'treatment': 'Insecticide sprays like cypermethrin'},
        'Wilt': {'conditions': {'temp': (28,35), 'humidity': (70,85), 'rainfall': (80,150)},
                'severity': 'High', 'symptoms': 'Sudden wilting, yellowing, vascular discoloration',
                'prevention': 'Crop rotation, resistant varieties',
                'treatment': 'Soil drench with carbendazim'}
    },
    'Tomato': {
        'Late Blight': {'conditions': {'temp': (18,22), 'humidity': (80,95), 'rainfall': (60,120)},
                       'severity': 'Critical', 'symptoms': 'Water-soaked lesions on leaves, stems, and fruits',
                       'prevention': 'Use resistant varieties, avoid overhead irrigation',
                       'treatment': 'Fungicide sprays like mancozeb'},
        'Leaf Curl': {'conditions': {'temp': (25,30), 'humidity': (65,80), 'rainfall': (50,90)},
                     'severity': 'High', 'symptoms': 'Upward curling of leaves, stunted growth',
                     'prevention': 'Control whiteflies, use virus-resistant varieties',
                     'treatment': 'Insecticide sprays, remove infected plants'}
    },
    'Chilli': {
        'Leaf Spot': {'conditions': {'temp': (22,28), 'humidity': (75,90), 'rainfall': (60,120)},
                     'severity': 'Medium', 'symptoms': 'Brown spots with concentric rings on leaves',
                     'prevention': 'Crop rotation, proper spacing',
                     'treatment': 'Copper-based fungicides'},
        'Anthracnose': {'conditions': {'temp': (25,30), 'humidity': (80,95), 'rainfall': (80,150)},
                       'severity': 'High', 'symptoms': 'Sunken dark lesions on fruits, fruit rot',
                       'prevention': 'Use resistant varieties, proper drainage',
                       'treatment': 'Fungicide sprays, remove infected fruits'}
    },
    'Maize': {
        'Leaf Blight': {'conditions': {'temp': (22,28), 'humidity': (75,90), 'rainfall': (80,150)},
                       'severity': 'High', 'symptoms': 'Long elliptical lesions on leaves, premature drying',
                       'prevention': 'Use resistant hybrids, crop rotation',
                       'treatment': 'Fungicide sprays, balanced fertilization'},
        'Stem Borer': {'conditions': {'temp': (25,30), 'humidity': (60,80), 'rainfall': (60,120)},
                      'severity': 'Critical', 'symptoms': 'Dead hearts, stem tunneling',
                      'prevention': 'Use Bt maize, pheromone traps',
                      'treatment': 'Insecticide granules in leaf whorls'}
    }
}

def predict_disease(crop, temp, humidity, rainfall):
    diseases = []
    if crop in DISEASE_DB:
        for name, data in DISEASE_DB[crop].items():
            cond = data['conditions']
            risk_score = sum([cond['temp'][0] <= temp <= cond['temp'][1],
                              cond['humidity'][0] <= humidity <= cond['humidity'][1],
                              cond['rainfall'][0] <= rainfall <= cond['rainfall'][1]])
            risk_pct = (risk_score / 3) * 100
            if risk_pct >= 40:
                diseases.append({'name': name, 'risk': risk_pct, **data})
    return sorted(diseases, key=lambda x: x['risk'], reverse=True)

# ============================================================
# YIELD PREDICTION (unchanged)
# ============================================================
def train_yield_model():
    np.random.seed(42)
    X, y = [], []
    for _ in range(3000):
        N, P, K = np.random.uniform(20,150), np.random.uniform(20,80), np.random.uniform(20,100)
        temp, humid = np.random.uniform(10,40), np.random.uniform(40,90)
        ph, rain = np.random.uniform(5.0,7.5), np.random.uniform(50,250)
        base = 50
        nutrient = 1 + (N/200) + (P/200) + (K/200)
        temp_eff = 1 - 0.01 * abs(temp - 25)**1.5
        humid_eff = 1 - 0.005 * abs(humid - 70)**1.5
        ph_eff = 1 - 0.1 * abs(ph - 6.5)**1.5
        rain_eff = 1 - 0.002 * abs(rain - 150)**1.5
        yield_kg = base * nutrient * max(0,temp_eff) * max(0,humid_eff) * max(0,ph_eff) * max(0,rain_eff)
        X.append([N, P, K, temp, humid, ph, rain])
        y.append(max(0, yield_kg + np.random.normal(0, 3)))
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y > 40)
    return model, scaler

yield_model, yield_scaler = None, None
try:
    yield_model, yield_scaler = train_yield_model()
except:
    pass

def predict_yield(crop, N, P, K, temp, humidity, ph, rainfall):
    if yield_model is None:
        return {'yield_quintals_per_acre': 0, 'confidence': 0, 'status': 'Model not available'}
    crop_factors = {'Rice':1.2, 'Wheat':1.1, 'Maize':1.0, 'Cotton':0.8, 'Sugarcane':1.3, 
                    'Groundnut':0.7, 'Tomato':0.9, 'Potato':0.85, 'Onion':0.75, 'Chilli':0.6,
                    'Banana':1.1, 'Mango':0.5, 'Default':1.0}
    factor = crop_factors.get(crop, 1.0)
    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    input_scaled = yield_scaler.transform(input_data)
    base = 45 + (N/200)*20 + (P/200)*15 + (K/200)*15
    base = max(10, min(100, base)) * factor
    return {
        'yield_quintals_per_acre': round(base, 2),
        'confidence': min(95, 75 + (base/100)*20),
        'status': 'Good' if base > 40 else 'Average' if base > 25 else 'Low',
        'range_low': round(base * 0.85, 2),
        'range_high': round(base * 1.15, 2)
    }

# ============================================================
# ENSEMBLE MODEL (unchanged)
# ============================================================
@st.cache_data
def generate_training_data():
    np.random.seed(42)
    data = []
    for crop, params in CROP_DB.items():
        for _ in range(200):
            sample = {'crop': crop}
            for feature, (low, high) in params.items():
                sample[feature] = round(np.random.uniform(low, high), 2)
            data.append(sample)
    return pd.DataFrame(data)

os.makedirs('models', exist_ok=True)

if not os.path.exists('models/ensemble_model.pkl'):
    with st.spinner("🔄 Training AI Ensemble..."):
        df = generate_training_data()
        feature_cols = ['N', 'P', 'K', 'temp', 'humidity', 'ph', 'rainfall']
        X = df[feature_cols].to_numpy()
        y = df['crop'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        encoder = LabelEncoder()
        y_train_enc, y_test_enc = encoder.fit_transform(y_train), encoder.transform(y_test)
        models = {
            'RF': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'DT': DecisionTreeClassifier(max_depth=15, random_state=42)
        }
        trained_models = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train_enc)
            trained_models.append((name, model))
        ensemble = VotingClassifier(estimators=trained_models, voting='soft')
        ensemble.fit(X_train_scaled, y_train_enc)
        ensemble_score = ensemble.score(X_test_scaled, y_test_enc)
        joblib.dump(ensemble, 'models/ensemble_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoder, 'models/label_encoder.pkl')
        with open('models/crop_names.txt', 'w') as f:
            f.write('\n'.join(encoder.classes_))
        st.success(f"✅ Ensemble trained! Accuracy: {ensemble_score:.2%}")

@st.cache_resource
def load_models():
    model = joblib.load('models/ensemble_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    with open('models/crop_names.txt', 'r') as f:
        crops = f.read().splitlines()
    return model, scaler, encoder, crops

model, scaler, encoder, crops = load_models()

# ============================================================
# STATE & SEASON DATABASE (unchanged)
# ============================================================
STATE_CROP_DB = {
    'Andhra Pradesh': {'Summer': ['Maize','Groundnut','Sunflower','Cotton'], 'Winter': ['Wheat','Chickpea','Potato','Tomato','Onion'], 'Rainy': ['Rice','Sugarcane','Turmeric','Chilli']},
    'Gujarat': {'Summer': ['Cotton','Groundnut','Sugarcane','Maize'], 'Winter': ['Wheat','Mustard','Chickpea','Potato','Onion'], 'Rainy': ['Rice','Chilli','Turmeric','Mango','Banana']},
    'Karnataka': {'Summer': ['Maize','Sunflower','Groundnut','Cotton'], 'Winter': ['Wheat','Chickpea','Lentil','Potato','Onion'], 'Rainy': ['Rice','Sugarcane','Coffee','Tea','Banana','Papaya','Coconut','Pomegranate']},
    'Maharashtra': {'Summer': ['Cotton','Soybean','Sunflower','Groundnut'], 'Winter': ['Wheat','Chickpea','Lentil','Potato','Onion'], 'Rainy': ['Rice','Sugarcane','Turmeric','Ginger','Chilli','Mango','Banana']},
    'Punjab': {'Summer': ['Maize','Cotton','Sugarcane','Sunflower'], 'Winter': ['Wheat','Mustard','Potato','Tomato'], 'Rainy': ['Rice','Millet','Barley']},
    'Rajasthan': {'Summer': ['Millet','Maize','Cotton','Groundnut'], 'Winter': ['Wheat','Mustard','Chickpea','Lentil','Potato'], 'Rainy': ['Rice','Pigeonpea','Mungbean','Barley']},
    'Tamil Nadu': {'Summer': ['Maize','Groundnut','Sunflower','Cotton'], 'Winter': ['Wheat','Chickpea','Lentil','Potato','Onion'], 'Rainy': ['Rice','Sugarcane','Banana','Papaya','Coconut','Guava','Watermelon','Turmeric','Chilli']},
    'Uttar Pradesh': {'Summer': ['Maize','Cotton','Sugarcane','Sunflower'], 'Winter': ['Wheat','Mustard','Chickpea','Lentil','Potato','Onion'], 'Rainy': ['Rice','Pigeonpea','Mungbean','Barley','Turmeric','Ginger','Chilli']},
    'West Bengal': {'Summer': ['Maize','Groundnut','Sunflower','Cotton'], 'Winter': ['Wheat','Mustard','Chickpea','Lentil','Potato','Onion'], 'Rainy': ['Rice','Sugarcane','Jute','Tea','Banana','Papaya','Mango','Turmeric','Ginger']}
}
SEASONS = ['Summer', 'Winter', 'Rainy']

# ============================================================
# SIDEBAR – WITH LOCATION DETAILS DISPLAY
# ============================================================
with st.sidebar:
    st.header("📊 Prediction Parameters (2026)")
    
    st.markdown("---")
    st.subheader("🌤️ Auto-Fetch Weather (Free)")

    # Session state for weather and location
    if 'weather_auto_fetched' not in st.session_state:
        st.session_state.weather_auto_fetched = False
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'location_details' not in st.session_state:
        st.session_state.location_details = None
    if 'weather_timestamp' not in st.session_state:
        st.session_state.weather_timestamp = None
    if 'auto_temp' not in st.session_state:
        st.session_state.auto_temp = 25
    if 'auto_humid' not in st.session_state:
        st.session_state.auto_humid = 70
    if 'auto_rain' not in st.session_state:
        st.session_state.auto_rain = 150

    DEFAULT_LOCATION = "Mumbai"
    location_input = st.text_input(
        "Enter City Name or 6‑digit PIN Code",
        value=st.session_state.get('last_location', DEFAULT_LOCATION)
    )

    fetch_clicked = st.button("📡 Fetch Weather", use_container_width=True)

    if not st.session_state.weather_auto_fetched:
        fetch_clicked = True
        st.session_state.weather_auto_fetched = True

    if fetch_clicked:
        with st.spinner(f"Fetching weather for '{location_input}'..."):
            loc = get_location_details(location_input)
            if loc['success']:
                st.session_state.location_details = loc
                weather = get_weather_by_location(loc['latitude'], loc['longitude'])
                if weather['success']:
                    st.session_state.weather_data = weather
                    st.session_state.weather_timestamp = datetime.now()
                    st.session_state['last_location'] = location_input
                    st.session_state.auto_temp = int(round(weather['temperature']))
                    st.session_state.auto_humid = int(round(weather['humidity']))
                    st.session_state.auto_rain = int(round(weather['rainfall']))
                    st.success(f"✅ Weather fetched for {loc['place_name']}")
                else:
                    st.error("❌ Failed to fetch weather. Using default values.")
            else:
                st.error(f"❌ {loc.get('error', 'Location not found.')}")

    # ----- Display Location Details if available -----
    if st.session_state.location_details:
        loc = st.session_state.location_details
        st.markdown("---")
        st.subheader("📍 Location Details")
        st.markdown(f"**Place:** {loc.get('place_name', 'N/A')}")
        if loc.get('district'):
            st.markdown(f"**District:** {loc['district']}")
        if loc.get('state'):
            st.markdown(f"**State:** {loc['state']}")
        if loc.get('latitude') and loc.get('longitude'):
            st.caption(f"Coordinates: {loc['latitude']:.4f}, {loc['longitude']:.4f}")

    # ----- Display Weather if available -----
    if st.session_state.weather_data:
        w = st.session_state.weather_data
        st.markdown("---")
        st.subheader("🌤️ Current Weather")
        st.info(f"🌡️ {w['temperature']:.1f}°C | 💧 {w['humidity']:.0f}% | ☔ {w['rainfall']:.1f}mm")
        if st.session_state.weather_timestamp:
            st.caption(f"Last updated: {st.session_state.weather_timestamp.strftime('%H:%M:%S')}")
    else:
        st.info("Click 'Fetch Weather' to get live data.")

    st.markdown("---")
    
    # Input sliders with auto-fetch support
    N = st.slider("Nitrogen (N) kg/ha", 0, 200, 90)
    P = st.slider("Phosphorus (P) kg/ha", 0, 200, 45)
    K = st.slider("Potassium (K) kg/ha", 0, 200, 40)
    temp = st.slider("Temperature (°C)", 0, 45, st.session_state.auto_temp)
    humid = st.slider("Humidity (%)", 20, 100, st.session_state.auto_humid)
    ph = st.slider("Soil pH", 4.0, 8.5, 6.5, 0.1)
    rain = st.slider("Rainfall (mm)", 20, 350, st.session_state.auto_rain)
    
    st.markdown("---")
    st.subheader("📍 State & Season")
    selected_state = st.selectbox("Select State", ['All'] + sorted(STATE_CROP_DB.keys()))
    selected_season = st.selectbox("Select Season", ['All'] + SEASONS)
    
    predict = st.button("🌾 Get Recommendation", use_container_width=True)

# ============================================================
# TABS (unchanged)
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔍 Crop Lookup", "🧠 AI Prediction", "📊 State/Season"])

with tab1:
    # ... (same as before) ...
    st.subheader("🔍 Crop Lookup – Type a crop name")
    crop_search = st.text_input("Enter crop name:", value=st.session_state.get('search_crop', ''))
    if st.button("📖 Show Crop Parameters", use_container_width=True):
        if crop_search.strip():
            found = [c for c in CROP_DB.keys() if crop_search.strip().lower() in c.lower()]
            if found:
                crop_name = found[0]
                params = CROP_DB[crop_name]
                st.success(f"### 🌾 Optimal Parameters for **{crop_name}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nitrogen (N)", f"{params['N'][0]} – {params['N'][1]} kg/ha")
                    st.metric("Phosphorus (P)", f"{params['P'][0]} – {params['P'][1]} kg/ha")
                    st.metric("Potassium (K)", f"{params['K'][0]} – {params['K'][1]} kg/ha")
                with col2:
                    st.metric("Temperature", f"{params['temp'][0]} – {params['temp'][1]} °C")
                    st.metric("Humidity", f"{params['humidity'][0]} – {params['humidity'][1]} %")
                with col3:
                    st.metric("Soil pH", f"{params['ph'][0]} – {params['ph'][1]}")
                    st.metric("Rainfall", f"{params['rainfall'][0]} – {params['rainfall'][1]} mm")
            else:
                st.error(f"❌ Crop '{crop_search}' not found.")
    
    st.subheader("🌾 All 39 Crops")
    cols = st.columns(6)
    for i, crop in enumerate(sorted(CROP_DB.keys())):
        with cols[i % 6]:
            if st.button(crop, key=f"tag_{crop}"):
                st.session_state['search_crop'] = crop
                st.rerun()

with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("N-P-K", f"{N}-{P}-{K}")
        st.metric("Temperature", f"{temp}°C")
        st.metric("Humidity", f"{humid}%")
    with col2:
        st.metric("pH", f"{ph}")
        st.metric("Rainfall", f"{rain} mm")

    if predict:
        input_data = np.array([[N, P, K, temp, humid, ph, rain]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        prob = proba[pred] * 100
        crop = encoder.inverse_transform([pred])[0]

        st.markdown(f"""
            <div class="recommendation-box">
                <h2>🌾 Recommended Crop</h2>
                <h1 style="font-size:56px;">{crop}</h1>
                <p style="font-size:28px;">Confidence: {prob:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        top_idx = np.argsort(proba)[-5:][::-1]
        st.subheader("🏆 Top 5 Recommendations")
        cols = st.columns(5)
        for i, idx in enumerate(top_idx):
            with cols[i]:
                c = encoder.inverse_transform([idx])[0]
                p = proba[idx]*100
                st.metric(f"#{i+1}", c, f"{p:.1f}%")

        st.subheader("🦠 Disease Risk Assessment")
        diseases = predict_disease(crop, temp, humid, rain)
        if diseases:
            st.warning(f"⚠️ {len(diseases)} potential disease risks detected")
            for d in diseases[:3]:
                with st.expander(f"🔴 {d['name']} – Risk: {d['risk']:.0f}%"):
                    st.markdown(f"""
                    **Severity:** {d['severity']}  
                    **Symptoms:** {d['symptoms']}  
                    **🛡️ Prevention:** {d['prevention']}  
                    **💊 Treatment:** {d['treatment']}  
                    """)
        else:
            st.success("✅ No major disease risks detected!")

        st.subheader("📊 Expected Yield Prediction")
        yield_result = predict_yield(crop, N, P, K, temp, humid, ph, rain)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Yield", f"{yield_result['yield_quintals_per_acre']} quintals/acre")
        with col2:
            st.metric("Yield Range", f"{yield_result['range_low']} – {yield_result['range_high']} qtl/acre")
        with col3:
            st.metric("Confidence", f"{yield_result['confidence']}%")
        yield_pct = min(100, (yield_result['yield_quintals_per_acre'] / 80) * 100)
        st.progress(min(100, yield_pct / 100))
        st.caption(f"Yield Status: **{yield_result['status']}**")

        st.subheader("🧠 Model Information")
        st.info("""
        **Ensemble Model** (4 models combined):
        - Random Forest (200 trees)
        - SVM (RBF kernel)
        - K-Nearest Neighbors
        - Decision Tree
        Voting: Soft voting (probability-based)
        """)

        if selected_state != 'All' and selected_season != 'All':
            state_crops = STATE_CROP_DB.get(selected_state, {}).get(selected_season, [])
            if crop in state_crops:
                st.success(f"✅ {crop} is suitable for **{selected_state}** in **{selected_season}**")
            else:
                st.info(f"ℹ️ {crop} may not be commonly grown in {selected_state} during {selected_season}")

with tab3:
    st.subheader("📍 Find Crops by State & Season")
    col1, col2 = st.columns(2)
    with col1:
        state_filter = st.selectbox("State", ['All'] + sorted(STATE_CROP_DB.keys()))
    with col2:
        season_filter = st.selectbox("Season", ['All'] + SEASONS)
    if st.button("🌾 Show Crops", use_container_width=True):
        if state_filter != 'All' and season_filter != 'All':
            crops_in_state = STATE_CROP_DB.get(state_filter, {}).get(season_filter, [])
            if crops_in_state:
                st.success(f"✅ Crops for **{state_filter}** in **{season_filter}**:")
                cols = st.columns(4)
                for i, crop in enumerate(sorted(crops_in_state)):
                    with cols[i % 4]:
                        if st.button(crop, key=f"state_{crop}"):
                            st.session_state['search_crop'] = crop
                            st.rerun()
            else:
                st.warning("No crops found for this combination.")
        else:
            st.info("Please select both state and season.")

st.markdown("---")
st.caption("🌱 2026 AI Crop Recommendation System | 39 Crops | Weather + Disease + Yield + Ensemble | Made for farmers")