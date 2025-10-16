import numpy as np
from scipy.optimize import minimize


def sample_from_spec(spec, n):
    dist = spec.get("dist", "normal").lower()
    if dist == "normal":
        return np.random.normal(spec["mean"], spec["std"], n)
    if dist == "lognormal":  # expects log-space μ, σ
        return np.random.lognormal(spec["mean"], spec["std"], n)
    if dist == "triangular":  # expects min, mode, max
        return np.random.triangular(spec["min"], spec["mode"], spec["max"], n)
    if dist == "gamma":  # expects shape k, scale θ
        return np.random.gamma(spec["shape"], spec["scale"], n)
    if dist == "beta":  # expects min, max
        return np.random.beta(spec["a"], spec["b"], n)
    raise ValueError(f"Unsupported dist: {dist}")


def optimize_without_yield(all_costs, lambda_risk, constraints=None):
    """
    Optimizes portfolio allocation to minimize E[Cost] + lambda * SD[Cost]

    Args:
        all_costs: dict of {country: cost_array}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None

    Returns:
        optimal allocations, expected cost, std dev
    """
    countries_list = list(all_costs.keys())
    n_countries = len(countries_list)

    def objective(weights):
        # Calculate portfolio cost for all runs
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        # portfolio_yield = 0

        for i, country in enumerate(countries_list):
            portfolio += weights[i] * all_costs[country]
            # portfolio_yield += weights[i] * yields[country]

        # GUARD: Avoid division by zero if yield is somehow zero
        # if portfolio_yield <= 0:
        #     return np.inf

        cost_per_good_unit = portfolio

        expected_cost = np.mean(cost_per_good_unit)
        std_cost = np.std(cost_per_good_unit)

        return expected_cost + lambda_risk * std_cost

    # Constraints: weights sum to 1, all weights >= 0
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = []

    # Add custom constraints if provided
    if constraints:
        for country in countries_list:
            if country in constraints:
                min_alloc, max_alloc = constraints[country]
                bounds.append((min_alloc, max_alloc))
            else:
                bounds.append((0, 1))
    else:
        bounds = [(0, 1) for _ in range(n_countries)]

    # Initial guess: equal allocation
    x0 = np.ones(n_countries) / n_countries

    # Optimize
    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=[constraint_sum]
    )

    if result.success:
        optimal_weights = result.x
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        for i, country in enumerate(countries_list):
            portfolio += optimal_weights[i] * all_costs[country]

        return {
            "allocations": {
                countries_list[i]: w for i, w in enumerate(optimal_weights)
            },
            "expected_cost": np.mean(portfolio),
            "std_cost": np.std(portfolio),
            "portfolio_costs": portfolio,
        }
    else:
        return None


def optimize_portfolio(all_costs, yields, lambda_risk, constraints=None):
    """
    Optimizes portfolio allocation to minimize E[Cost] + lambda * SD[Cost]

    Args:
        all_costs: dict of {country: cost_array}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None

    Returns:
        optimal allocations, expected cost, std dev
    """
    countries_list = list(all_costs.keys())
    n_countries = len(countries_list)

    def objective(weights):
        # Calculate portfolio cost for all runs
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        portfolio_yield = 0

        for i, country in enumerate(countries_list):
            portfolio += weights[i] * all_costs[country]
            portfolio_yield += weights[i] * yields[country]

        # GUARD: Avoid division by zero if yield is somehow zero
        if portfolio_yield <= 0:
            return np.inf

        cost_per_good_unit = portfolio / portfolio_yield

        expected_cost = np.mean(cost_per_good_unit)
        std_cost = np.std(cost_per_good_unit)

        return expected_cost + lambda_risk * std_cost

    # Constraints: weights sum to 1, all weights >= 0
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = []

    # Add custom constraints if provided
    if constraints:
        for country in countries_list:
            if country in constraints:
                min_alloc, max_alloc = constraints[country]
                bounds.append((min_alloc, max_alloc))
            else:
                bounds.append((0, 1))
    else:
        bounds = [(0, 1) for _ in range(n_countries)]

    # Initial guess: equal allocation
    x0 = np.ones(n_countries) / n_countries

    # Optimize
    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=[constraint_sum]
    )

    if result.success:
        optimal_weights = result.x
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        for i, country in enumerate(countries_list):
            portfolio += optimal_weights[i] * all_costs[country]

        return {
            "allocations": {
                countries_list[i]: w for i, w in enumerate(optimal_weights)
            },
            "expected_cost": np.mean(portfolio),
            "std_cost": np.std(portfolio),
            "portfolio_costs": portfolio,
        }
    else:
        return None
