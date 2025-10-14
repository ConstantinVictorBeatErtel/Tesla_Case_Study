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
