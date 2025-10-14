"""
Samplers Module

Builds country-specific sampler collections that draw from posterior distributions.
Each sampler function takes a number of runs and returns that many samples.

WHY SAMPLERS?
- Monte Carlo simulation needs random samples
- Each sample represents one possible future scenario
- Drawing from posteriors (not just point estimates) gives realistic uncertainty
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.stats import beta as beta_dist

from .parameter_estimators import (
    estimate_fx_posterior,
    estimate_raw_material_posterior,
    estimate_yield_posterior,
)


@dataclass
class CountrySamplers:
    """
    Collection of sampler functions for a country.
    Each function takes n_runs and returns n_runs samples from posterior predictive.
    """

    raw_material: Callable[[int], np.ndarray]
    labor: Callable[[int], np.ndarray]
    logistics: Callable[[int], np.ndarray]
    fx_multiplier: Callable[[int], np.ndarray]
    yield_rate: Callable[[int], np.ndarray]

    # Keep other params as constants (no good public data)
    # indirect_cost: float
    # electricity_cost: float
    # depreciation_cost: float
    # working_capital: float


def build_samplers_for_country(country: str, baseline_params: Dict) -> CountrySamplers:
    """
    Build posterior samplers from real data for a country.
    Falls back to baseline parameters if data unavailable.
    """
    print(f"\n🔧 Building samplers for {country}...")

    # Raw materials
    try:
        raw_post = estimate_raw_material_posterior(baseline_params["raw"]["mean"])

        def raw_sampler(n):
            return raw_post.sample_predictive(n)

    except Exception:
        raw_mean = baseline_params["raw"]["mean"]
        raw_std = baseline_params["raw"]["std"]

        def raw_sampler(n):
            return np.random.normal(raw_mean, raw_std, n)

    # Labor (no good public data - use baseline)
    labor_mean = baseline_params["labor"]["mean"]
    labor_std = baseline_params["labor"]["std"]
    labor_sampler = lambda n: np.random.normal(labor_mean, labor_std, n)
    print(f"  → Labor: Baseline Normal({labor_mean}, {labor_std})")

    # Logistics (special handling for China lognormal)
    if baseline_params["logistics"]["dist"] == "lognormal":
        log_mean = baseline_params["logistics"]["mean"]
        log_std = baseline_params["logistics"]["std"]
        sigma = np.sqrt(np.log(1 + (log_std**2 / log_mean**2)))
        mu = np.log(log_mean) - (sigma**2 / 2)
        logistics_sampler = lambda n: np.random.lognormal(mu, sigma, n)
        print("  → Logistics: Lognormal (high volatility from case study)")
    else:
        log_mean = baseline_params["logistics"]["mean"]
        log_std = baseline_params["logistics"]["std"]
        logistics_sampler = lambda n: np.random.normal(log_mean, log_std, n)
        print(f"  → Logistics: Normal({log_mean}, {log_std})")

    # FX volatility
    try:
        fx_sampler = estimate_fx_posterior(country)
        print("  ✓ FX: Posterior from FRED exchange rate data")
    except Exception:
        fx_std = baseline_params["currency_std"]

        def fx_sampler(n):
            return 1 + np.random.normal(0, fx_std, n)

        print(f"  ⚠️  FX: Using baseline volatility {fx_std}")

    # Manufacturing yield
    try:
        yield_baseline = baseline_params["yield_params"]["a"] / (
            baseline_params["yield_params"]["a"] + baseline_params["yield_params"]["b"]
        )
        uncertainty_map = {"US": "medium", "Mexico": "high", "China": "low"}
        yield_post = estimate_yield_posterior(yield_baseline, uncertainty_map[country])

        def yield_sampler(n):
            return yield_post.sample_predictive(n).clip(0.01, 0.99)

        print(f"  ✓ Yield: Beta posterior (uncertainty={uncertainty_map[country]})")
    except Exception:
        a = baseline_params["yield_params"]["a"]
        b = baseline_params["yield_params"]["b"]

        def yield_sampler(n):
            return beta_dist.rvs(a, b, size=n)

        print(f"  ⚠️  Yield: Using baseline Beta({a}, {b})")

    return CountrySamplers(
        raw_material=raw_sampler,
        labor=labor_sampler,
        logistics=logistics_sampler,
        fx_multiplier=fx_sampler,
        yield_rate=yield_sampler,
        # indirect_cost=sample_from_spec(baseline_params["indirect"])
        # electricity_cost=sample_from_spec(baseline_params["electricity"]),
        # depreciation_cost=sample_from_spec(baseline_params["depcreciation"]),
        # working_capital=sample_from_spec(baseline_params["working_capital"]),
    )
