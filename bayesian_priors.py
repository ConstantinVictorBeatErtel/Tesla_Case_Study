"""
Bayesian Parameter Estimation for Tesla Sourcing Simulation

WHY BAYESIAN?
- With limited data (24 months), we're uncertain about TRUE parameters
- Naive approach: Use sample mean/std (pretends we know true values)
- Bayesian approach: Account for parameter uncertainty via posterior predictive
- Result: More conservative risk estimates (wider tails = Student-t vs Normal)

WHAT THIS DOES:
1. Fetches real data (PPI, FX rates, etc.)
2. Fits Bayesian posterior from data
3. Returns SAMPLER FUNCTIONS that draw from posterior predictive distributions
4. These samplers replace your hand-picked parameters in app.py
"""

import numpy as np
import pandas as pd
import requests
from scipy.stats import invgamma, t, beta as beta_dist
from typing import Callable, Dict, Tuple
from dataclasses import dataclass

# ============================================================================
# STEP 1: DATA FETCHING
# ============================================================================

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
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
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


# ============================================================================
# STEP 2: BAYESIAN POSTERIOR FITTING
# ============================================================================

@dataclass
class NormalPosterior:
    """
    Posterior from Normal-Inverse-Gamma prior.
    Represents: "Based on N observations, parameters are PROBABLY around (μ, σ²)"
    """
    mu: float      # Posterior mean location
    kappa: float   # Confidence in mean (higher = more certain)
    alpha: float   # Shape of variance posterior
    beta: float    # Scale of variance posterior
    
    def sample_predictive(self, size: int) -> np.ndarray:
        """
        Draw from posterior predictive distribution (Student-t).
        This accounts for BOTH data variability AND parameter uncertainty.
        """
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return self.mu + t.rvs(df=df, loc=0, scale=scale, size=size)


def fit_normal_posterior(data: pd.Series,
                        prior_mean: float = None,
                        prior_weight: float = 0.001) -> NormalPosterior:
    """
    Fit Normal-Inverse-Gamma posterior from data.
    
    Args:
        data: Observed values
        prior_mean: Prior belief about mean (default: sample mean)
        prior_weight: How much to trust prior vs data (small = weak prior)
    
    Returns:
        Posterior parameters that capture parameter uncertainty
    """
    if len(data) == 0:
        raise ValueError("Cannot fit posterior with no data")
    
    n = len(data)
    xbar = data.mean()
    s2 = data.var(ddof=1) if n > 1 else 0.01
    
    # Use sample mean as prior if not specified
    if prior_mean is None:
        prior_mean = xbar
    
    # Posterior parameters (combine prior + data)
    kappa_n = prior_weight + n
    mu_n = (prior_weight * prior_mean + n * xbar) / kappa_n
    alpha_n = prior_weight/2 + n/2
    beta_n = prior_weight/2 + 0.5 * ((n-1) * s2 + (prior_weight * n * (xbar - prior_mean)**2) / kappa_n)
    
    return NormalPosterior(mu_n, kappa_n, alpha_n, beta_n)


def fit_lognormal_posterior(data: pd.Series) -> NormalPosterior:
    """
    Fit posterior for lognormally-distributed data.
    Works by fitting Normal posterior to log(data).
    """
    positive_data = data[data > 0]
    if len(positive_data) == 0:
        raise ValueError("Need positive values for lognormal fitting")
    
    log_data = np.log(positive_data)
    return fit_normal_posterior(pd.Series(log_data))


@dataclass
class BetaPosterior:
    """
    Posterior for probabilities/yields (bounded 0-1).
    """
    alpha: float
    beta: float
    
    def sample_predictive(self, size: int) -> np.ndarray:
        """Draw from Beta posterior."""
        return beta_dist.rvs(self.alpha, self.beta, size=size)


def fit_beta_posterior(successes: int, trials: int, 
                      prior_alpha: float = 1.0, prior_beta: float = 1.0) -> BetaPosterior:
    """
    Fit Beta posterior from success/failure counts.
    
    Example: 18 good parts out of 20 → (18, 20, 1, 1) → Beta(19, 3)
    """
    return BetaPosterior(
        alpha=prior_alpha + successes,
        beta=prior_beta + (trials - successes)
    )


def fit_beta_from_rates(rates: pd.Series, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> BetaPosterior:
    """
    Fit Beta posterior from series of observed rates (0-1).
    Treats each observation as Bernoulli trial.
    """
    if len(rates) == 0:
        return BetaPosterior(prior_alpha, prior_beta)
    
    # Convert rates to pseudo-counts
    mean_rate = rates.mean()
    n = len(rates)
    successes = int(round(mean_rate * n))
    
    return fit_beta_posterior(successes, n, prior_alpha, prior_beta)


# ============================================================================
# STEP 3: PARAMETER ESTIMATORS (Data → Posterior)
# ============================================================================

def estimate_raw_material_posterior(baseline: float) -> NormalPosterior:
    """
    Fit posterior for raw material costs from PPI data.
    Uses plastics PPI as proxy for automotive components.
    """
    ppi = fetch_fred_series("PCU325211325211P", months=24)
    
    if len(ppi) < 2:
        print("⚠️  No PPI data, using hand-picked parameters")
        # Fallback: create posterior that mimics Normal(baseline, baseline*0.1)
        return NormalPosterior(mu=baseline, kappa=100, alpha=50, beta=50*(baseline*0.1)**2)
    
    # Convert PPI index to dollar values centered at baseline
    ppi_normalized = (ppi / ppi.mean()) * baseline
    
    return fit_normal_posterior(ppi_normalized, prior_mean=baseline)


def estimate_fx_posterior(country: str) -> Callable[[int], np.ndarray]:
    """
    Fit FX volatility from exchange rate data.
    Returns a MULTIPLIER function: cost → cost * (1 ± FX_change)
    """
    series_map = {
        'Mexico': 'DEXMXUS',  # Peso per USD
        'China': 'DEXCHUS'     # Yuan per USD
    }
    
    if country == 'US':
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


def estimate_yield_posterior(baseline_yield: float, uncertainty: str = 'medium') -> BetaPosterior:
    """
    Create Beta posterior for manufacturing yield.
    
    Args:
        baseline_yield: Expected yield (e.g., 0.80 for 80%)
        uncertainty: 'low' (tight), 'medium', 'high' (wide spread)
                    Use 'high' for Mexico ("questionable skills")
    """
    # Uncertainty maps to total pseudo-observations
    uncertainty_map = {
        'low': 100,    # Very confident (China mature facility)
        'medium': 50,  # Moderate confidence (US new facility)
        'high': 15     # Low confidence (Mexico skill issues)
    }
    
    total = uncertainty_map.get(uncertainty, 50)
    alpha = baseline_yield * total
    beta = (1 - baseline_yield) * total
    
    return BetaPosterior(alpha, beta)


# ============================================================================
# STEP 4: BUILD SAMPLERS FOR EACH COUNTRY
# ============================================================================

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
    indirect_cost: float
    electricity_cost: float
    depreciation_cost: float
    working_capital: float


def build_samplers_for_country(country: str, baseline_params: Dict) -> CountrySamplers:
    """
    Build posterior samplers from real data for a country.
    Falls back to baseline parameters if data unavailable.
    """
    print(f"\n🔧 Building samplers for {country}...")
    
    # Raw materials
    try:
        raw_post = estimate_raw_material_posterior(baseline_params['raw']['mean'])
        raw_sampler = lambda n: raw_post.sample_predictive(n)
        print(f"  ✓ Raw materials: Posterior from PPI data")
    except Exception as e:
        raw_mean = baseline_params['raw']['mean']
        raw_std = baseline_params['raw']['std']
        raw_sampler = lambda n: np.random.normal(raw_mean, raw_std, n)
        print(f"  ⚠️  Raw materials: Using baseline Normal({raw_mean}, {raw_std})")
    
    # Labor (no good public data - use baseline)
    labor_mean = baseline_params['labor']['mean']
    labor_std = baseline_params['labor']['std']
    labor_sampler = lambda n: np.random.normal(labor_mean, labor_std, n)
    print(f"  → Labor: Baseline Normal({labor_mean}, {labor_std})")
    
    # Logistics (special handling for China lognormal)
    if baseline_params['logistics']['dist'] == 'lognormal':
        log_mean = baseline_params['logistics']['mean']
        log_std = baseline_params['logistics']['std']
        sigma = np.sqrt(np.log(1 + (log_std**2 / log_mean**2)))
        mu = np.log(log_mean) - (sigma**2 / 2)
        logistics_sampler = lambda n: np.random.lognormal(mu, sigma, n)
        print(f"  → Logistics: Lognormal (high volatility from case study)")
    else:
        log_mean = baseline_params['logistics']['mean']
        log_std = baseline_params['logistics']['std']
        logistics_sampler = lambda n: np.random.normal(log_mean, log_std, n)
        print(f"  → Logistics: Normal({log_mean}, {log_std})")
    
    # FX volatility
    try:
        fx_sampler = estimate_fx_posterior(country)
        print(f"  ✓ FX: Posterior from FRED exchange rate data")
    except Exception as e:
        fx_std = baseline_params['currency_std']
        fx_sampler = lambda n: (1 + np.random.normal(0, fx_std, n))
        print(f"  ⚠️  FX: Using baseline volatility {fx_std}")
    
    # Manufacturing yield
    try:
        yield_baseline = baseline_params['yield_params']['a'] / (
            baseline_params['yield_params']['a'] + baseline_params['yield_params']['b']
        )
        uncertainty_map = {'US': 'medium', 'Mexico': 'high', 'China': 'low'}
        yield_post = estimate_yield_posterior(yield_baseline, uncertainty_map[country])
        yield_sampler = lambda n: yield_post.sample_predictive(n).clip(0.01, 0.99)
        print(f"  ✓ Yield: Beta posterior (uncertainty={uncertainty_map[country]})")
    except Exception as e:
        a = baseline_params['yield_params']['a']
        b = baseline_params['yield_params']['b']
        yield_sampler = lambda n: beta_dist.rvs(a, b, size=n)
        print(f"  ⚠️  Yield: Using baseline Beta({a}, {b})")
    
    return CountrySamplers(
        raw_material=raw_sampler,
        labor=labor_sampler,
        logistics=logistics_sampler,
        fx_multiplier=fx_sampler,
        yield_rate=yield_sampler,
        indirect_cost=baseline_params['indirect']['mean'],
        electricity_cost=baseline_params['electricity']['mean'],
        depreciation_cost=baseline_params['depreciation']['mean'],
        working_capital=baseline_params['working_capital']['mean']
    )


# ============================================================================
# STEP 5: RUN SIMULATION WITH BAYESIAN SAMPLERS
# ============================================================================

def simulate_with_bayesian_priors(country: str,
                                 samplers: CountrySamplers,
                                 risk_params: Dict,
                                 n_runs: int) -> np.ndarray:
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
    tariff = risk_params['tariff']['fixed'] + np.random.normal(
        risk_params['tariff_escal']['mean'],
        risk_params['tariff_escal']['std'],
        n_runs
    )
    total = base / yield_rate + tariff
    
    # Add discrete risks (same as original)
    total += np.random.binomial(1, risk_params['disruption_prob'], n_runs) * risk_params['disruption_impact']
    
    border_time = np.random.normal(risk_params['border_mean'], risk_params['border_std'], n_runs)
    total += np.maximum(0, border_time - risk_params['border_threshold']) * risk_params['border_cost_per_hr']
    
    total += np.random.binomial(1, risk_params['damage_prob'], n_runs) * risk_params['damage_impact']
    
    total *= (1 + np.random.normal(risk_params['skills_mean'], risk_params['skills_std'], n_runs))
    
    total += np.random.binomial(1, risk_params['cancellation_prob'], n_runs) * risk_params['cancellation_impact']
    
    return total


# ============================================================================
# STEP 6: CONVENIENCE FUNCTION FOR APP.PY
# ============================================================================

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
    
    print("\n" + "="*60)
    print("BUILDING BAYESIAN POSTERIOR SAMPLERS")
    print("="*60)
    
    for country, params in countries_dict.items():
        samplers = build_samplers_for_country(country, params)
        
        # Create closure that captures samplers and params
        def make_sim(c=country, s=samplers, p=params):
            return lambda n: simulate_with_bayesian_priors(c, s, p, n)
        
        simulators[country] = make_sim()
    
    print("\n✅ All samplers built successfully!")
    print("="*60 + "\n")
    
    return simulators


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Minimal test
    test_params = {
        'US': {
            'raw': {'mean': 40, 'std': 4},
            'labor': {'mean': 12, 'std': 0.6},
            'indirect': {'mean': 10},
            'logistics': {'dist': 'normal', 'mean': 9, 'std': 0},
            'electricity': {'mean': 4},
            'depreciation': {'mean': 5},
            'working_capital': {'mean': 5},
            'yield_params': {'a': 79, 'b': 20},
            'currency_std': 0,
            'tariff': {'fixed': 0},
            'tariff_escal': {'mean': 0, 'std': 0},
            'disruption_prob': 0.05,
            'disruption_impact': 10,
            'border_mean': 0,
            'border_std': 0,
            'border_threshold': 2,
            'border_cost_per_hr': 10,
            'damage_prob': 0.01,
            'damage_impact': 20,
            'skills_mean': 0,
            'skills_std': 0,
            'cancellation_prob': 0,
            'cancellation_impact': 50
        }
    }
    
    simulators = create_bayesian_simulator(test_params)
    costs = simulators['US'](1000)
    
    print(f"\nTest simulation complete:")
    print(f"  Mean: ${costs.mean():.2f}")
    print(f"  Std:  ${costs.std():.2f}")
    print(f"  95th: ${np.percentile(costs, 95):.2f}")