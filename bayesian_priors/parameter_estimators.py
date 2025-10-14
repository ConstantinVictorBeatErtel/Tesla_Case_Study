"""
Parameter Estimators Module

Converts real-world data into Bayesian posterior distributions.
These functions fetch data and fit it to appropriate probability models.

WHAT THIS DOES:
- Takes baseline values and real economic data
- Returns posterior distributions that capture uncertainty
- Used by samplers to generate realistic cost scenarios
"""

from typing import Callable

import numpy as np

from .data_fetching import fetch_fred_series
from .posterior_models import (
    BetaPosterior,
    NormalPosterior,
    fit_normal_posterior,
)


def estimate_raw_material_posterior(baseline: float) -> NormalPosterior:
    """
    Fit posterior for raw material costs from PPI data.
    Uses plastics PPI as proxy for automotive components.
    """
    ppi = fetch_fred_series("PCU325211325211P", months=24)

    if len(ppi) < 2:
        print("⚠️  No PPI data, using hand-picked parameters")
        # Fallback: create posterior that mimics Normal(baseline, baseline*0.1)
        return NormalPosterior(
            mu=baseline, kappa=100, alpha=50, beta=50 * (baseline * 0.1) ** 2
        )

    # Convert PPI index to dollar values centered at baseline
    ppi_normalized = (ppi / ppi.mean()) * baseline

    return fit_normal_posterior(ppi_normalized, prior_mean=baseline)


def estimate_fx_posterior(country: str) -> Callable[[int], np.ndarray]:
    """
    Fit FX volatility from exchange rate data.
    Returns a MULTIPLIER function: cost → cost * (1 ± FX_change)
    """
    series_map = {
        "Mexico": "DEXMXUS",  # Peso per USD
        "China": "DEXCHUS",  # Yuan per USD
    }

    if country == "US":
        return lambda n: np.ones(n)  # No FX risk

    if country not in series_map:
        print(f"⚠️  No FX mapping for {country}")
        return lambda n: np.ones(n)

    fx_series = fetch_fred_series(series_map[country], months=12)

    if len(fx_series) < 2:
        print(f"⚠️  No FX data for {country}, using zero volatility")
        return lambda n: np.ones(n)

    # Calculate log returns (percent changes)
    returns = np.log(fx_series / fx_series.shift(1)).dropna()

    # Fit posterior to returns
    posterior = fit_normal_posterior(returns, prior_mean=0.0)

    # Return sampler that adds FX volatility
    def fx_multiplier(n: int) -> np.ndarray:
        fx_shocks = posterior.sample_predictive(n)
        return 1.0 + fx_shocks

    return fx_multiplier


def estimate_yield_posterior(
    baseline_yield: float, uncertainty: str = "medium"
) -> BetaPosterior:
    """
    Create Beta posterior for manufacturing yield.

    Args:
        baseline_yield: Expected yield (e.g., 0.80 for 80%)
        uncertainty: 'low' (tight), 'medium', 'high' (wide spread)
                    Use 'high' for Mexico ("questionable skills")
    """
    # Uncertainty maps to total pseudo-observations
    uncertainty_map = {
        "low": 100,  # Very confident (China mature facility)
        "medium": 50,  # Moderate confidence (US new facility)
        "high": 15,  # Low confidence (Mexico skill issues)
    }

    total = uncertainty_map.get(uncertainty, 50)
    alpha = baseline_yield * total
    beta = (1 - baseline_yield) * total

    return BetaPosterior(alpha, beta)

