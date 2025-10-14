"""
Simulation Module

Monte Carlo simulation logic using Bayesian samplers.
This is where we actually run the cost simulations for each country.

HOW IT WORKS:
1. Sample cost components from their posterior distributions
2. Apply manufacturing yield and tariffs
3. Add discrete risk events (disruptions, border delays, etc.)
4. Return array of total costs across all simulation runs
"""

from typing import Dict

import numpy as np

from .samplers import CountrySamplers


def simulate_with_bayesian_priors(
    country: str, samplers: CountrySamplers, risk_params: Dict, n_runs: int
) -> np.ndarray:
    """
    Run Monte Carlo simulation using Bayesian posterior samplers.

    This is a DROP-IN REPLACEMENT for simulate_country() in app.py.
    """
    # Sample from posterior predictives
    raw = samplers.raw_material(n_runs)
    labor = samplers.labor(n_runs)
    logistics = samplers.logistics(n_runs)
    fx_mult = samplers.fx_multiplier(n_runs)
    yield_rate = samplers.yield_rate(n_runs)

    # Constants (no public data)
    indirect = samplers.indirect_cost
    electricity = samplers.electricity_cost
    depreciation = samplers.depreciation_cost
    working = samplers.working_capital

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working
    base = base * fx_mult  # Apply FX volatility

    # Apply yield and tariff
    tariff = risk_params["tariff"]["fixed"] + np.random.normal(
        risk_params["tariff_escal"]["mean"], risk_params["tariff_escal"]["std"], n_runs
    )
    total = base / yield_rate + tariff

    # Add discrete risks (same as original)
    total += (
        np.random.binomial(1, risk_params["disruption_prob"], n_runs)
        * risk_params["disruption_impact"]
    )

    border_time = np.random.normal(
        risk_params["border_mean"], risk_params["border_std"], n_runs
    )
    total += (
        np.maximum(0, border_time - risk_params["border_threshold"])
        * risk_params["border_cost_per_hr"]
    )

    total += (
        np.random.binomial(1, risk_params["damage_prob"], n_runs)
        * risk_params["damage_impact"]
    )

    total *= 1 + np.random.normal(
        risk_params["skills_mean"], risk_params["skills_std"], n_runs
    )

    total += (
        np.random.binomial(1, risk_params["cancellation_prob"], n_runs)
        * risk_params["cancellation_impact"]
    )

    return total

