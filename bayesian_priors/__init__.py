"""
Bayesian Parameter Estimation for Tesla Sourcing Simulation

WHY BAYESIAN?
- With limited data (24 months), we're uncertain about TRUE parameters
- Naive approach: Use sample mean/std (pretends we know true values)
- Bayesian approach: Account for parameter uncertainty via posterior predictive
- Result: More conservative risk estimates (wider tails = Student-t vs Normal)

WHY THIS INCREASES SUPPLY CHAIN READINESS:
- More realistic tail risk: Fatter tails in student-t distribution = better prepared for extreme cost scenarios
- Data-driven: Uses actual PPI/FX data instead of guesses
- Conservative planning: Wider confidence intervals → less likely to underestimate costs

WHAT THIS DOES:
1. Fetches real data (PPI, FX rates, etc.)
2. Fits Bayesian posterior from data
3. Returns SAMPLER FUNCTIONS that draw from posterior predictive distributions
4. These samplers replace your hand-picked parameters in app.py

VARIABLES WE ARE DOING THIS FOR (with real data):
- [x] raw material (PPI data → Student-t samples)
- [ ] labor 
- [ ] logistics
- [x] fx (FRED exchange rates → Student-t samples)
- [x] yield (uncertainty-adjusted Beta)
- [ ] indirect
- [ ] electricity
- [ ] depreciation
- [ ] working capital

USAGE:
    from bayesian_priors import create_bayesian_simulator
    
    bayesian_sims = create_bayesian_simulator(countries)
    
    # Use instead of simulate_country()
    us_costs = bayesian_sims['US'](n_runs=10000)
"""

from typing import Callable, Dict

import numpy as np

from .samplers import build_samplers_for_country
from .simulation import simulate_with_bayesian_priors

# Expose main API functions
__all__ = ["create_bayesian_simulator"]


def create_bayesian_simulator(countries_dict: Dict) -> Dict[str, Callable]:
    """
    Create Bayesian simulators for all countries.

    Returns:
        Dict mapping country -> simulator function

    Usage in app.py:
        from bayesian_priors import create_bayesian_simulator

        bayesian_sims = create_bayesian_simulator(countries)

        # Use instead of simulate_country()
        us_costs = bayesian_sims['US'](n_runs=10000)
    """
    simulators = {}

    print("\n" + "=" * 60)
    print("BUILDING BAYESIAN POSTERIOR SAMPLERS")
    print("=" * 60)

    for country, params in countries_dict.items():
        samplers = build_samplers_for_country(country, params)

        # Create closure that captures samplers and params
        def make_sim(c=country, s=samplers, p=params):
            return lambda n: simulate_with_bayesian_priors(c, s, p, n)

        simulators[country] = make_sim()

    print("\n✅ All samplers built successfully!")
    print("=" * 60 + "\n")

    return simulators


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Minimal test
    test_params = {
        "US": {
            "raw": {"mean": 40, "std": 4},
            "labor": {"mean": 12, "std": 0.6},
            "indirect": {"mean": 10},
            "logistics": {"dist": "normal", "mean": 9, "std": 0},
            "electricity": {"mean": 4},
            "depreciation": {"mean": 5},
            "working_capital": {"mean": 5},
            "yield_params": {"a": 79, "b": 20},
            "currency_std": 0,
            "tariff": {"fixed": 0},
            "tariff_escal": {"mean": 0, "std": 0},
            "disruption_prob": 0.05,
            "disruption_impact": 10,
            "border_mean": 0,
            "border_std": 0,
            "border_threshold": 2,
            "border_cost_per_hr": 10,
            "damage_prob": 0.01,
            "damage_impact": 20,
            "skills_mean": 0,
            "skills_std": 0,
            "cancellation_prob": 0,
            "cancellation_impact": 50,
        }
    }

    simulators = create_bayesian_simulator(test_params)
    costs = simulators["US"](1000)

    print("\nTest simulation complete:")
    print(f"  Mean: ${costs.mean():.2f}")
    print(f"  Std:  ${costs.std():.2f}")
    print(f"  95th: ${np.percentile(costs, 95):.2f}")

