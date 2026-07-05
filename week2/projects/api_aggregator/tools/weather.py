import httpx

CITY_COORDS = {
    "islamabad": (33.7215, 73.0433), "karachi": (24.8607, 67.0011),
    "lahore": (31.5204, 74.3587), "london": (51.5074, -0.1278),
    "dubai": (25.2048, 55.2708), "new york": (40.7128, -74.0060),
    "paris": (48.8566, 2.3522), "tokyo": (35.6762, 139.6503),
    "riyadh": (24.7136, 46.6753), "istanbul": (41.0082, 28.9784),
}

WEATHER_CODES = {
    0: "☀️ Clear sky", 1: "🌤️ Mainly clear", 2: "⛅ Partly cloudy",
    3: "☁️ Overcast", 45: "🌫️ Foggy", 61: "🌦️ Light rain",
    63: "🌧️ Moderate rain", 65: "🌧️ Heavy rain", 80: "🌦️ Showers",
    95: "⛈️ Thunderstorm"
}

def fetch_weather(city: str) -> dict:
    city_lower = city.lower().strip()
    coords = CITY_COORDS.get(city_lower)
    if not coords:
        return {"error": f"City '{city}' not supported. Available: {', '.join(c.title() for c in CITY_COORDS)}"}

    lat, lon = coords
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "forecast_days": 3, "timezone": "auto"
                }
            )
            resp.raise_for_status()
            data = resp.json()

        current = data["current"]
        daily = data["daily"]
        condition = WEATHER_CODES.get(current["weather_code"], f"Code {current['weather_code']}")

        return {
            "city": city.title(),
            "condition": condition,
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "wind_kmh": current["wind_speed_10m"],
            "precipitation_mm": current["precipitation"],
            "forecast_3days": [
                {
                    "day": f"Day {i+1}",
                    "max": daily["temperature_2m_max"][i],
                    "min": daily["temperature_2m_min"][i]
                }
                for i in range(min(3, len(daily["temperature_2m_max"])))
            ]
        }
    except Exception as e:
        return {"error": str(e)}