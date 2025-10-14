from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class DiscreteRiskSimulation:
    delayed_units: NDArray[int]
    costs: NDArray[float]


@dataclass
class DiscreteRisks:
    disruptions: DiscreteRiskSimulation
    border_delays: DiscreteRiskSimulation
    last_minute_cancellations: DiscreteRiskSimulation
    damaged: DiscreteRiskSimulation
    defectives: DiscreteRiskSimulation


@dataclass
class DiscreteRisksParams:
    disruption_lambda: float  # disruption ~ poisson
    border_delay_lambda: float  # border delay ~ poisson
    damage_probability: float  # damaged ~ binomial
    defective_probability: float  # defectives ~ binomial
    cancellation_probability: float  # cancellation ~ bernoulli
    days_delayed: int  # the number of days we have to wait because of the delay
    # note that the above might be different per country, per risk?
