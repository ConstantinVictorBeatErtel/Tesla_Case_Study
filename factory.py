import numpy as np


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


def factory_costs(params, n_runs):
    # Sample costs from distributions
    raw = sample_from_spec(params["raw"], n_runs)
    labor = sample_from_spec(params["labor"], n_runs)
    indirect = sample_from_spec(params["indirect"], n_runs)
    electricity = sample_from_spec(params["electricity"], n_runs)
    depreciation = sample_from_spec(params["depreciation"], n_runs)
    working = sample_from_spec(params["working_capital"], n_runs)
    yield_ = sample_from_spec(params["yield_params"], n_runs)

    # Handle lognormal distribution for logistics
    if params["logistics"]["dist"] == "lognormal":
        m, s = params["logistics"]["mean"], params["logistics"]["std"]
        sigma = np.sqrt(np.log(1 + (s**2 / m**2)))
        mu = np.log(m) - (sigma**2 / 2)
        logistics = np.random.lognormal(mu, sigma, n_runs)
    else:
        logistics = np.random.normal(
            params["logistics"]["mean"], params["logistics"]["std"], n_runs
        )

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working

    # Apply currency fluctuation
    base *= 1 + np.random.normal(0, params["currency_std"], n_runs)

    # Apply tariff and potential escalation
    if "fixed" in params["tariff_escal"]:
        tariff = np.full(n_runs, params["tariff"]["fixed"]) + np.full(
            n_runs, params["tariff_escal"]["fixed"]
        )
    else:
        tariff = np.full(n_runs, params["tariff"]["fixed"]) + np.random.normal(
            params["tariff_escal"]["mean"], params["tariff_escal"]["std"], n_runs
        )

    # Calculate total cost before discrete risks
    total = base / yield_ + tariff

    return total
