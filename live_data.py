import os

from dotenv import load_dotenv
from requests import get

load_dotenv()

api_key = os.getenv("FRED_API_KEY")
url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "FEDFUNDS",
    "api_key": api_key,
    "file_type": "json",
}

response = get(url, params=params)


def get_fed_funds_rate():
    response = get(url, params=params)
    return response.json()
