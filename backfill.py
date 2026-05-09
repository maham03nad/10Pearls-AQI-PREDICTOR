"""
- Fetches past AQI data from OpenWeatherMap Air Pollution History API
- Engineers the same features as feature_pipeline.py
- Stores all historical rows in Hopsworks Feature Store (bulk insert)

Run once before training:
    python backfill.py --days 365
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks


OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "YOUR_OWM_KEY")
HOPSWORKS_KEY   = os.getenv("HOPSWORKS_API_KEY", "YOUR_HW_KEY")
CITY = os.getenv("CITY", "karachi")
LAT  = float(os.getenv("LAT", "24.8607"))
LON  = float(os.getenv("LON", "67.0011"))

def fetch_historical_aqi(lat, lon, start_dt: datetime, end_dt: datetime) -> list:
    """
    OpenWeatherMap Air Pollution History API returns hourly data.
    """
    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={OPENWEATHER_KEY}"
    )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json().get("list", [])
