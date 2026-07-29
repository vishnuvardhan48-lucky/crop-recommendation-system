#!/usr/bin/env python3
"""
Weather by Pincode – Live weather for any Indian pincode or city.
Usage: python weather_by_pincode.py 400001
       python weather_by_pincode.py Mumbai
"""

import sys
import requests
import indiapins
from datetime import datetime

# ============================================================
# 1. GEOCODING – Convert place/pincode to coordinates
# ============================================================

def get_coordinates(place):
    """
    Convert a place name or Indian pincode to (latitude, longitude).
    Returns a dict with success flag and lat/lon or error message.
    """
    # 1. If it's a 6-digit pincode, use indiapins
    if isinstance(place, str) and len(place) == 6 and place.isdigit():
        try:
            records = indiapins.matching(place)
            if records:
                # Take first record with coordinates
                for rec in records:
                    lat = rec.get('Latitude')
                    lon = rec.get('Longitude')
                    if lat and lon:
                        return {
                            'success': True,
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'place_name': rec.get('Name', place),
                            'district': rec.get('District', ''),
                            'state': rec.get('State', '')
                        }
                # If no coordinates, return error
                return {'success': False, 'error': f'Pincode {place} found but no coordinates available.'}
        except ValueError as e:
            return {'success': False, 'error': str(e)}

    # 2. Fallback: Open‑Meteo Geocoding API (for city names)
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={place}&count=1&language=en&format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return {
                    'success': True,
                    'latitude': result['latitude'],
                    'longitude': result['longitude'],
                    'place_name': result.get('name', place),
                    'district': result.get('admin1', ''),
                    'state': result.get('admin1', '')
                }
    except Exception as e:
        pass

    return {'success': False, 'error': f'Could not locate "{place}". Please check spelling or try a pincode.'}

# ============================================================
# 2. WEATHER FETCH – Open‑Meteo (no API key)
# ============================================================

def fetch_weather(lat, lon):
    """
    Fetch current weather from Open‑Meteo for given coordinates.
    Returns dict with weather data or error.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "weather_code"],
            "timezone": "auto",
            "forecast_days": 1
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get('current', {})

        # Map weather codes to text descriptions (simplified)
        weather_code_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        wc = current.get('weather_code')
        weather_desc = weather_code_map.get(wc, f"Unknown ({wc})") if wc is not None else "N/A"

        return {
            'success': True,
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'precipitation': current.get('precipitation'),
            'wind_speed': current.get('wind_speed_10m'),
            'weather_code': wc,
            'weather_description': weather_desc,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================
# 3. MAIN – Command‑line interface
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python weather_by_pincode.py <pincode or place>")
        print("Example: python weather_by_pincode.py 400001")
        print("Example: python weather_by_pincode.py Mumbai")
        sys.exit(1)

    place = sys.argv[1].strip()

    # Step 1: Get coordinates
    loc = get_coordinates(place)
    if not loc['success']:
        print(f"❌ {loc['error']}")
        sys.exit(1)

    lat, lon = loc['latitude'], loc['longitude']
    print(f"📍 Location: {loc.get('place_name', place)}")
    if loc.get('district'):
        print(f"   District: {loc['district']}, State: {loc['state']}")
    print(f"   Coordinates: {lat}, {lon}")

    # Step 2: Fetch weather
    weather = fetch_weather(lat, lon)
    if not weather['success']:
        print(f"❌ Weather error: {weather['error']}")
        sys.exit(1)

    # Step 3: Display
    print("\n🌤️ Live Weather Report")
    print(f"   Temperature: {weather['temperature']}°C")
    print(f"   Humidity: {weather['humidity']}%")
    print(f"   Precipitation: {weather['precipitation']} mm")
    print(f"   Wind Speed: {weather['wind_speed']} km/h")
    print(f"   Conditions: {weather['weather_description']}")
    print(f"   Last updated: {weather['timestamp']}")

if __name__ == "__main__":
    main()