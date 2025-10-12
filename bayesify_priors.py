# bayesify_priors.py
# Empirical-Bayes priors & posterior-predictive samplers for your Monte Carlo.
# Minimal deps: numpy, pandas, scipy, requests

import io, math, json, time, warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from scipy.stats import invgamma, t, beta as beta_dist

# ---------- Utilities

def _to_series(x) -> pd.Series:
    s = pd.Series(x).dropna()
    return s.astype(float)

def _rolling_window(series: pd.Series, months: int = 24) -> pd.Series:
    if isinstance(series.index, pd.DatetimeIndex):
        cutoff = series.index.max() - pd.DateOffset(months=months)
        return series[series.index > cutoff]
    return series.tail(months)  # last N points if not dated

# ---------- Normal with unknown μ,σ²  → Normal-Inverse-Gamma posterior → Student-t predictive

@dataclass
class NormalIGPosterior:
    mu_n: float
    kappa_n: float
    alpha_n: float
    beta_n: float

def fit_normal_inverse_gamma(data: pd.Series,
                             mu0: Optional[float] = None,
                             kappa0: float = 1e-6,
                             alpha0: float = 1e-6,
                             beta0: float = 1e-6) -> NormalIGPosterior:
    x = _to_series(data)
    n = len(x)
    if n == 0:
        raise ValueError("No data to fit Normal-Inverse-Gamma.")
    xbar = x.mean()
    s2 = x.var(ddof=1) if n > 1 else 1e-6
    if mu0 is None:
        mu0 = xbar
    kappa_n = kappa0 + n
    mu_n = (kappa0*mu0 + n*xbar) / kappa_n
    alpha_n = alpha0 + n/2
    beta_n = beta0 + 0.5*((n-1)*s2 + (kappa0*n*(xbar-mu0)**2)/kappa_n)
    return NormalIGPosterior(mu_n, kappa_n, alpha_n, beta_n)

def sample_posterior_predictive_normal(post: NormalIGPosterior, size: int) -> np.ndarray:
    # Posterior predictive is Student-t with df=2*alpha_n, mean=mu_n, scale = sqrt(beta_n*(kappa_n+1)/(alpha_n*kappa_n))
    df = 2*post.alpha_n
    scale = math.sqrt(post.beta_n*(post.kappa_n + 1) / (post.alpha_n*post.kappa_n))
    return post.mu_n + t.rvs(df=df, loc=0, scale=scale, size=size)

# ---------- Lognormal: work on ln(x), then exp

def fit_lognormal_post(data: pd.Series) -> NormalIGPosterior:
    x = _to_series(data)
    x = x[x > 0]
    ln = np.log(x)
    return fit_normal_inverse_gamma(pd.Series(ln))

def sample_posterior_predictive_lognormal(post: NormalIGPosterior, size: int) -> np.ndarray:
    y = sample_posterior_predictive_normal(post, size)
    return np.exp(y)

# ---------- Beta for probabilities (yield, cancellations, etc.)

def beta_from_counts(successes: int, trials: int, alpha0: float = 1.0, beta0: float = 1.0) -> Tuple[float, float]:
    return alpha0 + successes, beta0 + (trials - successes)

def beta_from_rate_series(rates: pd.Series, alpha0: float = 1.0, beta0: float = 1.0) -> Tuple[float, float]:
    r = _to_series(rates)
    # Treat each rate observation as “trials=1” (Bernoulli over a period) → sum as successes
    k = int(round((r > 0).sum()*r.mean())) if len(r) else 0  # heuristic if only percentages available
    n = len(r)
    return beta_from_counts(k, n if n > 0 else 1, alpha0, beta0)

def beta_from_moments(mean: float, var: float) -> Tuple[float, float]:
    # method of moments for Beta (simple and robust)
    if var <= 0:  # collapse to a pseudo-count prior if variance unknown
        kappa = 20.0
        return mean*kappa, (1-mean)*kappa
    tmp = mean*(1-mean)/var - 1
    a = mean*tmp
    b = (1-mean)*tmp
    if a <= 0 or b <= 0:
        kappa = 20.0
        a, b = mean*kappa, (1-mean)*kappa
    return a, b

# ---------- Data fetchers (plug your CSVs/paths as needed)

def fred_csv(series_id: str) -> pd.Series:
    # FRED supports direct CSV: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    sname = [c for c in df.columns if c != "DATE"][0]
    s = pd.to_numeric(df[sname], errors="coerce").dropna()
    return s

def freightos_lane_laneid(lane: str) -> pd.Series:
    """
    Example stub for FBX. Many dashboards require login.
    Simplest path: export CSV manually and point here; or scrape with an API key if available.
    Expected CSV columns: date, value.
    """
    # TODO: replace with your CSV path:
    raise FileNotFoundError("Provide local CSV for FBX lane (date,value).")

def ilostat_wages_stub(country: str) -> pd.Series:
    """
    Stub: use ILOSTAT SDMX or a local CSV for manufacturing earnings (monthly/annual).
    Return a numeric series aligned to months. If annual, forward-fill monthly.
    """
    # TODO: replace with your CSV path / SDMX query.
    raise FileNotFoundError("Provide ILOSTAT wages CSV for country.")

def cbp_border_wait_times_stub(port_number: str = "250601", lane: str = "commercial_standard") -> pd.Series:
    """
    CBP BWT API returns XML; historical UI offers download per port. Provide CSV with columns [date,hours].
    """
    # TODO: replace with local CSV
    raise FileNotFoundError("Provide CBP border wait-times CSV for chosen port.")

# ---------- Build samplers per country from data

@dataclass
class Samplers:
    raw: Callable[[int], np.ndarray]
    labor: Callable[[int], np.ndarray]
    indirect: Callable[[int], np.ndarray]
    logistics: Callable[[int], np.ndarray]
    electricity: Callable[[int], np.ndarray]
    depreciation: Callable[[int], np.ndarray]
    working_capital: Callable[[int], np.ndarray]
    yield_sampler: Callable[[int], np.ndarray]
    tariff_draw: Callable[[int], np.ndarray]
    currency_mult: Callable[[int], np.ndarray]
    disruption_flag: Callable[[int], np.ndarray]
    damage_flag: Callable[[int], np.ndarray]
    cancellation_flag: Callable[[int], np.ndarray]
    border_hours: Callable[[int], np.ndarray]
    skills_mult: Callable[[int], np.ndarray]

def _make_normal_sampler_from_series(series: pd.Series) -> Callable[[int], np.ndarray]:
    post = fit_normal_inverse_gamma(series)
    return lambda n: sample_posterior_predictive_normal(post, n)

def _make_lognormal_sampler_from_series(series: pd.Series) -> Callable[[int], np.ndarray]:
    post = fit_lognormal_post(series)
    return lambda n: sample_posterior_predictive_lognormal(post, n)

def _make_beta_draw_from_series(series: pd.Series) -> Callable[[int], np.ndarray]:
    r = _to_series(series)
    if len(r) >= 2:
        a, b = beta_from_moments(r.mean(), r.var(ddof=1))
    elif len(r) == 1:
        a, b = beta_from_moments(float(r.iloc[0]), 0.01)
    else:
        a, b = (2.0, 2.0)  # weakly-informative
    return lambda n: beta_dist.rvs(a, b, size=n)

def _fx_vol_to_multiplier(fx_series: pd.Series) -> Callable[[int], np.ndarray]:
    # fx_series: domestic units per USD; compute daily returns, use σ of returns
    s = _to_series(fx_series)
    rets = np.log(s/s.shift(1)).dropna()
    sigma = rets.std() if len(rets) else 0.0
    return lambda n: (1.0 + np.random.normal(0.0, sigma, n))

def _bernoulli_from_rate_series(rate_series: pd.Series, alpha0=1.0, beta0=1.0) -> Callable[[int], np.ndarray]:
    r = _to_series(rate_series).clip(0, 1)
    if len(r) >= 2:
        a, b = beta_from_moments(r.mean(), r.var(ddof=1))
    elif len(r) == 1:
        a, b = beta_from_moments(float(r.iloc[0]), 0.01)
    else:
        a, b = (1.2, 8.0)  # weak prior (mean ~0.13)
    def draw(n):
        p = beta_dist.rvs(a, b, size=n)
        return np.random.binomial(1, p, size=n)
    return draw

def build_samplers_for_country(country: str) -> Samplers:
    """
    Plug whatever data you have; if missing, we fall back to current constants (you'll inject them below).
    """
    # --------- RAW: Plastics PPI (US) as proxy (scale later per country if desired)
    try:
        raw_series = fred_csv("PCU325211325211P")
        raw_series = _rolling_window(raw_series, months=24)
        raw_sampler = _make_normal_sampler_from_series(raw_series.pct_change().dropna()*100 + 40)  # center near your $40 baseline
    except Exception:
        raw_sampler = None

    # --------- LABOR: ILOSTAT (stub)
    labor_sampler = None
    try:
        wages = ilostat_wages_stub(country)  # returns e.g. USD/hour or index
        wages = _rolling_window(wages, months=24)
        labor_sampler = _make_normal_sampler_from_series(wages.pct_change().dropna()*100 + 10)
    except Exception:
        pass

    # --------- INDIRECT: Use Damodaran SG&A/Sales (use a tight normal around mean)
    try:
        # You can read margin.xls (SG&A/Sales) and pick the row for Auto Parts or desired sector
        # For simplicity, insert a constant; student t predictive around it:
        sga_pct = 10.0  # e.g., 10% as placeholder after you pull margin.xls
        indirect_sampler = lambda n: np.random.normal(sga_pct, 0.5, n)
    except Exception:
        indirect_sampler = None

    # --------- LOGISTICS: FBX lane series (stub; supply CSV)
    logistics_sampler = None
    try:
        if country == "China":
            lane_series = freightos_lane_laneid("FBX01")
        else:
            lane_series = freightos_lane_laneid("WCI_generic_or_regional")
        lane_series = _rolling_window(lane_series, months=12)
        logistics_sampler = _make_lognormal_sampler_from_series(lane_series)
    except Exception:
        pass

    # --------- ELECTRICITY: OECD/IEA (likely CSV/manual grab)
    electricity_sampler = None
    try:
        # Provide series in USD/kWh; we’ll jitter around its mean
        raise FileNotFoundError
    except Exception:
        pass

    # --------- FX volatility multiplier
    fx_mult_sampler = None
    try:
        if country == "Mexico":
            fx = fred_csv("DEXMXUS")
        elif country == "China":
            fx = fred_csv("DEXCHUS")
        else:  # USA baseline ~ no FX effect
            fx = pd.Series([0.0, 0.0], index=pd.date_range("2000-01-01", periods=2))
        fx_mult_sampler = _fx_vol_to_multiplier(_rolling_window(fx, months=12))
    except Exception:
        pass

    # --------- Border wait times (apply to Mexico only, typically)
    border_sampler = None
    try:
        if country == "Mexico":
            bwt = cbp_border_wait_times_stub(port_number="250601")  # example: Otay Mesa
            border_sampler = _make_normal_sampler_from_series(_rolling_window(bwt, months=12))
    except Exception:
        pass

    # --------- Cancellations / disruption flags from cancellation rate
    cancel_sampler = None
    try:
        # Provide weekly cancellation % as series in [0,1]
        raise FileNotFoundError
    except Exception:
        pass

    # --------- Damage prob: supply a rate series or leave prior weak
    damage_sampler = _bernoulli_from_rate_series(pd.Series([], dtype=float))

    # --------- Skills adj: keep ~0 mean, small std unless you have a proxy
    skills_sampler = lambda n: np.random.normal(0.0, 0.02, n)

    # Compose (some may be None; we'll swap in fallbacks later)
    return Samplers(
        raw=raw_sampler or (lambda n: np.random.normal(40, 4, n)),
        labor=labor_sampler or (lambda n: np.random.normal(10, 1, n)),
        indirect=indirect_sampler or (lambda n: np.random.normal(8, 1, n)),
        logistics=logistics_sampler or (lambda n: np.random.lognormal(mean=np.log(12), sigma=0.5, size=n)),
        electricity=electricity_sampler or (lambda n: np.random.normal(4, 0.4, n)),
        depreciation=lambda n: np.random.normal(5, 0.25, n),
        working_capital=lambda n: np.random.normal(6, 0.6, n),
        yield_sampler=(lambda n: beta_dist.rvs(49, 3, size=n)),  # fallback to your current China example
        tariff_draw=lambda n: np.full(n, 0.0),  # you’ll set per country below
        currency_mult=fx_mult_sampler or (lambda n: np.ones(n)),
        disruption_flag=_bernoulli_from_rate_series(pd.Series([0.05], dtype=float)),
        damage_flag=damage_sampler,
        cancellation_flag=cancel_sampler or _bernoulli_from_rate_series(pd.Series([0.10], dtype=float)),
        border_hours=border_sampler or (lambda n: np.random.normal(0.8, 0.7, n)),
        skills_mult=skills_sampler
    )

# ---------- End-to-end simulation using samplers

def simulate_country_bayes(country: str,
                           samplers: Samplers,
                           params_fixed: Dict,
                           n_runs: int) -> np.ndarray:
    raw = samplers.raw(n_runs)
    labor = samplers.labor(n_runs)
    indirect = samplers.indirect(n_runs)
    logistics = samplers.logistics(n_runs)
    electricity = samplers.electricity(n_runs)
    depreciation = samplers.depreciation(n_runs)
    working = samplers.working_capital(n_runs)

    yield_ = samplers.yield_sampler(n_runs).clip(1e-3, 0.999)

    base = raw + labor + indirect + logistics + electricity + depreciation + working
    base = base * samplers.currency_mult(n_runs)

    # tariff: fixed from your case file + escalation noise if provided
    fixed_tariff = params_fixed[country]['tariff']['fixed']
    t_escal_m = params_fixed[country]['tariff_escal']['mean']
    t_escal_s = params_fixed[country]['tariff_escal']['std']
    tariff = fixed_tariff + np.random.normal(t_escal_m, t_escal_s, n_runs)

    total = base / yield_ + tariff

    # Discrete risks
    total += samplers.disruption_flag(n_runs) * params_fixed[country]['disruption_impact']

    border_time = samplers.border_hours(n_runs)
    thr = params_fixed[country]['border_threshold']
    per_hr = params_fixed[country]['border_cost_per_hr']
    total += np.maximum(0, border_time - thr) * per_hr

    total += samplers.damage_flag(n_runs) * params_fixed[country]['damage_impact']

    total *= (1 + samplers.skills_mult(n_runs))
    total += samplers.cancellation_flag(n_runs) * params_fixed[country]['cancellation_impact']

    return total

# ---------- Convenience: wire everything up

def bayesify_countries(countries_dict: Dict) -> Dict[str, Callable[[int], np.ndarray]]:
    """
    Returns a dict country -> function(n_runs) that returns total cost draws using posterior-predictive samplers.
    """
    mapped = {}
    for ctry in countries_dict.keys():
        samps = build_samplers_for_country(ctry)
        mapped[ctry] = (lambda c=ctry, s=samps: (lambda n: simulate_country_bayes(c, s, countries_dict, n)))()
    return mapped
