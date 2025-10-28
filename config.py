from live_data import get_most_recent_fed_funds_rate

MONTE_CARLO_SIMULATIONS = 50000
MODEL_Y_PRICE = 41630
MODEL_Y_MANUFACTURING_COST = 38000
MODEL_Y_PROFIT = MODEL_Y_PRICE - MODEL_Y_MANUFACTURING_COST
WACC = 0.0877
EXPEDITED_SHIPPING_COST_PER_HEADLAMP = 50.71
# Fetch Fed Funds Rate from FRED API (falls back to default if unavailable)
FED_FUNDS_RATE = get_most_recent_fed_funds_rate()

COUNTRIES = {
    "China": {
        "raw": {"dist": "normal", "mean": 30, "std": 3},
        "labor": {
            "dist": "lognormal",
            "mean": 1.379,
            "std": 0.120,
        },  # mean ~4, std ~0.5
        "indirect": {"dist": "gamma", "shape": 16.0, "scale": 0.25},  # mean 4, std 1
        "logistics": {
            "dist": "lognormal",
            "mean": 12,
            "std": 8,
        },  # High volatility from trade disruptions
        "electricity": {"dist": "triangular", "min": 3.60, "mode": 4.00, "max": 4.40},
        "depreciation": {"dist": "normal", "mean": 5, "std": 0.25},
        "working_capital": {"dist": "normal", "mean": 10, "std": 1},
        "yield_params": {
            "dist": "beta",
            "a": 49,
            "b": 3,
        },  # Approx for mean 0.95, std 0.03
        "tariff": {"fixed": 0.25},
        "tariff_escal": 0.15,
        "currency_std": 0.03,
        # "disruption_prob": 0.2,
        # "disruption_impact": 10,
        # "border_mean": 0,
        # "border_std": 0,
        # "border_threshold": 2,
        # "border_cost_per_hr": 10,
        # "damage_prob": 0.02,
        # "damage_impact": 20,
        # "skills_mean": 0,
        # "skills_std": 0,
        # "cancellation_prob": 0.3,  # Updated from recent shipping data (30% cancellations)
        # "cancellation_impact": 50,
        # --- NEW DISCRETE VARIABLES ---
        "disruption_lambda": 0.15,  # NEW: Avg 0.2 disruptive events per shipment
        "disruption_min_impact": 100,
        "disruption_max_impact": 1000,
        "disruption_days_delayed": 10,
        # Border Delay Risks are impossible for China
        "border_delay_lambda": 0,
        "border_min_impact": 100,
        "border_max_impact": 1000,
        "border_days_delayed": 0,
        # Quality Risks (Binomial)
        "damage_probability": 0.02,
        "defective_probability": 0,  # NEW: Added a separate probability for defects
        "quality_days_delayed": 15,  # NEW: A single delay for any quality issue
        # Cancellation Risk (Bernoulli)
        "cancellation_probability": 0.15,
        "cancellation_days_delayed": 90,
        # --- NEW DISCRETE VARIABLES ---
    },
    "US": {
        # Continuous variables: these are the parameters for each plant
        "raw": {"dist": "normal", "mean": 40, "std": 4},
        "labor": {"dist": "lognormal", "mean": 2.48, "std": 0.15},  # mean ~12, std ~2
        "indirect": {"dist": "gamma", "shape": 25.0, "scale": 0.40},  # mean 10, std 2
        "logistics": {"dist": "normal", "mean": 9, "std": 0},
        "electricity": {"dist": "triangular", "min": 3.5, "mode": 4.0, "max": 4.5},
        "depreciation": {"dist": "normal", "mean": 5, "std": 0.25},
        "working_capital": {"dist": "normal", "mean": 5, "std": 0.5},
        "yield_params": {
            "dist": "beta",
            "a": 79,
            "b": 20,
        },  # Approx for mean 0.8, std 0.04
        # Discrete variables: these are the qualitative risks to the supply chain process
        # The USA has no tariffs or currency volatility
        "tariff": {"fixed": 0},
        "tariff_escal": 0,
        "currency_std": 0,
        # "disruption_prob": 0.05,
        # "disruption_impact": 10,
        # "border_mean": 0,
        # "border_std": 0,
        # "border_threshold": 2,
        # "border_cost_per_hr": 10,
        # "damage_prob": 0.01,
        # "damage_impact": 20,
        # "skills_mean": 0,
        # "skills_std": 0,
        # "cancellation_prob": 0,
        # "cancellation_impact": 50,
        # --- NEW DISCRETE VARIABLES ---
        "disruption_lambda": 0.002,  # NEW: Avg 0.2 disruptive events per shipment
        "disruption_min_impact": 5000,  # NEW: Min 5,000 units lost per event
        "disruption_max_impact": 15000,  # NEW: Max 15,000 units lost per event
        "disruption_days_delayed": 20,  # NEW: Specific delay for this risk
        # Border Delay Risks are impossible for US
        "border_delay_lambda": 0.0,
        "border_min_impact": 0,
        "border_max_impact": 0,
        "border_days_delayed": 0,
        # Quality Risks (Binomial)
        "damage_probability": 0.01,
        "defective_probability": 0,
        "quality_days_delayed": 15,
        # Cancellation Risk (Bernoulli)
        "cancellation_probability": 0.0001,
        "cancellation_days_delayed": 90,
        # --- NEW DISCRETE VARIABLES ---
    },
    "Mexico": {
        "raw": {"dist": "normal", "mean": 35, "std": 3.5},
        "labor": {
            "dist": "lognormal",
            "mean": 2.0635,
            "std": 0.1786,
        },  # mean ~8, std ~1.5
        "indirect": {
            "dist": "gamma",
            "shape": 20.66,
            "scale": 0.387,
        },  # mean 8, std 1.75
        "logistics": {"dist": "normal", "mean": 7, "std": 0.056},
        "electricity": {"dist": "triangular", "min": 2.5, "mode": 3.0, "max": 3.5},
        "depreciation": {"dist": "normal", "mean": 1, "std": 0.05},
        "working_capital": {"dist": "normal", "mean": 6, "std": 0.6},
        "yield_params": {
            "dist": "beta",
            "a": 12,
            "b": 1,
        },  # Approx for mean 0.9, std 0.08
        "tariff": {"fixed": 0.25},
        "tariff_escal": 0.1,
        "currency_std": 0.08,
        # "disruption_prob": 0.1,
        # "disruption_impact": 10000,
        # "border_mean": 0.83,
        # "border_std": 0.67,
        # "border_threshold": 2,
        # "border_cost_per_hr": 10,
        # "damage_prob": 0.015,
        # "damage_impact": 20,
        # "skills_mean": 0,
        # "skills_std": 0.05,
        # "cancellation_prob": 0,
        # "cancellation_impact": 50,
        # --- NEW DISCRETE VARIABLES ---
        "disruption_lambda": 0.1,  # NEW: 2 out of a 100 are disrupted
        "disruption_min_impact": 500,  # NEW: Min 5,000 units lost per event
        "disruption_max_impact": 1500,  # NEW: Max 15,000 units lost per event
        "disruption_days_delayed": 5,  # NEW: Specific delay for this risk
        # Border Delay Risks for Mexico
        "border_delay_lambda": 0.83,
        "border_min_impact": 100,
        "border_max_impact": 1000,
        "border_days_delayed": 20,
        # Quality Risks (Binomial)
        "damage_probability": 0.015,
        "defective_probability": 0.05,  # NEW: Added a separate probability for defects
        "quality_days_delayed": 5,  # NEW: A single delay for any quality issue
        # Cancellation Risk (Bernoulli)
        "cancellation_probability": 0.0001,
        "cancellation_days_delayed": 90,
        # --- NEW DISCRETE VARIABLES ---
    },
}
