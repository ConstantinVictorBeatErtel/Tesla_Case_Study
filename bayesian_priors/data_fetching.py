"""
Data Fetching Module

Retrieves real-world economic data from FRED (Federal Reserve Economic Data).
This module handles fetching time series data like Producer Price Index (PPI)
and foreign exchange rates that feed into our Bayesian models.
"""

import pandas as pd


def fetch_fred_series(series_id: str, months: int = 24) -> pd.Series:
    """Fetch economic time series from FRED API."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        # FRED uses 'observation_date' as the date column name
        df = pd.read_csv(url)

        # Check if we got valid data
        if df.empty or len(df.columns) < 2:
            print(f"⚠️  No data returned for {series_id}")
            return pd.Series(dtype=float)

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
            return pd.Series(dtype=float)

        # Keep only recent months
        cutoff = values.index.max() - pd.DateOffset(months=months)
        values = values[values.index > cutoff]

        return values

    except Exception as e:
        print(f"⚠️  Could not fetch {series_id}: {e}")
        return pd.Series(dtype=float)
