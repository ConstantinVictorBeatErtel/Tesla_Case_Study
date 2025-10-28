"""
Data Fetching Module

Retrieves real-world economic data from FRED (Federal Reserve Economic Data).
This module handles fetching time series data like Producer Price Index (PPI)
and foreign exchange rates that feed into our Bayesian models.
"""

import pandas as pd
import requests

# Simple cache to prevent repeated API calls within the same session
_FRED_CACHE = {}


def fetch_fred_series(series_id: str, months: int = 24) -> pd.Series:
    """
    Fetch economic time series from FRED API.
    Uses proper connection management to prevent resource leaks.
    Results are cached to prevent repeated API calls.
    """
    # Check cache first
    cache_key = (series_id, months)
    if cache_key in _FRED_CACHE:
        return _FRED_CACHE[cache_key]

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        # Use a session context manager to ensure proper connection cleanup
        with requests.Session() as session:
            response = session.get(url, timeout=10)
            response.raise_for_status()

            # Read CSV from the response content
            from io import StringIO

            df = pd.read_csv(StringIO(response.text))

            # Check if we got valid data
            if df.empty or len(df.columns) < 2:
                print(f"⚠️  No data returned for {series_id}")
                result = pd.Series(dtype=float)
                _FRED_CACHE[cache_key] = result
                return result

            # Parse date column (usually 'observation_date')
            date_col = df.columns[0]
            value_col = df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)

            # Convert values to numeric and drop missing
            values = pd.to_numeric(df[value_col], errors="coerce").dropna()

            if len(values) == 0:
                print(f"⚠️  No valid numeric data for {series_id}")
                result = pd.Series(dtype=float)
                _FRED_CACHE[cache_key] = result
                return result

            # Keep only recent months
            cutoff = values.index.max() - pd.DateOffset(months=months)
            values = values[values.index > cutoff]

            # Cache the result
            _FRED_CACHE[cache_key] = values
            return values

    except Exception as e:
        print(f"⚠️  Could not fetch {series_id}: {e}")
        result = pd.Series(dtype=float)
        _FRED_CACHE[cache_key] = result
        return result
