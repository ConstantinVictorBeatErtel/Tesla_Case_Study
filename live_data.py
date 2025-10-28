import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FRED_API_KEY")
url = "https://api.stlouisfed.org/fred/series/observations"

# Fallback value for Fed Funds Rate (as of January 2025)
DEFAULT_FED_FUNDS_RATE = 4.5


def get_fed_funds_rate():
    """
    Fetches the Federal Funds Rate from FRED API.
    Returns None if the request fails.
    """
    try:
        if not api_key:
            return None

        params = {
            "series_id": "FEDFUNDS",
            "api_key": api_key,
            "file_type": "json",
        }
        # Use a session context manager to ensure proper connection cleanup
        with requests.Session() as session:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
    except Exception as e:
        print(f"Warning: Failed to fetch FRED data: {str(e)}")
        return None


def get_most_recent_fed_funds_rate():
    """
    Gets the most recent Federal Funds Rate from FRED API.
    Falls back to a default value if the API is unavailable.
    """
    try:
        data = get_fed_funds_rate()
        if data and "observations" in data and len(data["observations"]) > 0:
            return float(data["observations"][-1]["value"])
    except Exception as e:
        print(f"Warning: Failed to parse FRED data: {str(e)}")

    # Fallback to default value
    print(f"Using default Fed Funds Rate: {DEFAULT_FED_FUNDS_RATE}%")
    return DEFAULT_FED_FUNDS_RATE
