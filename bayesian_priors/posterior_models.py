"""
Posterior Models Module

Contains the core Bayesian model classes and fitting functions.
These models capture uncertainty about parameters when we have limited data.

WHY BAYESIAN?
- With only 24 months of data, we can't be certain about true parameter values
- Naive approach: Use sample mean/std (pretends we know true values)
- Bayesian approach: Account for parameter uncertainty via posterior predictive
- Result: More conservative risk estimates (wider tails)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import t


@dataclass
class NormalPosterior:
    """
    Captures uncertainty about a Normal distribution's mean & variance from limited data.
    
    WHY NOT JUST USE SAMPLE MEAN/STD?
    With only 24 months of data, you can't be certain those ARE the true parameters.
    This class accounts for that uncertainty.
    
    HOW IT WORKS:
    - Uses Normal-Inverse-Gamma prior (models mean & variance uncertainty together)
    - Conjugate prior = closed-form math updates (fast, stable)
    - Gives Student-t predictive distribution (fatter tails than Normal for small data)
    
    PARAMETERS:
    - mu, kappa: Best guess for mean, and confidence in that guess
    - alpha, beta: Control uncertainty about the variance
    """

    mu: float  # Posterior mean location
    kappa: float  # Confidence in mean (higher = more certain)
    alpha: float  # Shape of variance posterior
    beta: float  # Scale of variance posterior

    def sample_predictive(self, size: int) -> np.ndarray:
        """
        Draw from posterior predictive distribution (Student-t).
        This accounts for BOTH data variability AND parameter uncertainty.
        """
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return self.mu + t.rvs(df=df, loc=0, scale=scale, size=size)


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


def fit_normal_posterior(
    data: pd.Series, prior_mean: float = None, prior_weight: float = 0.001
) -> NormalPosterior:
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
    alpha_n = prior_weight / 2 + n / 2
    beta_n = prior_weight / 2 + 0.5 * (
        (n - 1) * s2 + (prior_weight * n * (xbar - prior_mean) ** 2) / kappa_n
    )

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


def fit_beta_posterior(
    successes: int, trials: int, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> BetaPosterior:
    """
    Fit Beta posterior from success/failure counts.

    Example: 18 good parts out of 20 → (18, 20, 1, 1) → Beta(19, 3)
    """
    return BetaPosterior(
        alpha=prior_alpha + successes, beta=prior_beta + (trials - successes)
    )


def fit_beta_from_rates(
    rates: pd.Series, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> BetaPosterior:
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

