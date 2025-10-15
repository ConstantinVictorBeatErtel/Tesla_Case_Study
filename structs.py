from dataclasses import dataclass


@dataclass
class DiscreteRiskSimulation:
    delayed_units: int
    cost: float


@dataclass
class DiscreteRisks:
    disruptions: DiscreteRiskSimulation
    border_delays: DiscreteRiskSimulation
    last_minute_cancellations: DiscreteRiskSimulation
    damaged: DiscreteRiskSimulation
    defectives: DiscreteRiskSimulation
    tariff_cost: float


@dataclass
class DiscreteRisksParams:
    disruption_lambda: float  # disruption ~ poisson
    disruption_min: int
    disruption_max: int
    disruption_days_delayed: int
    border_delay_lambda: float  # border delay ~ poisson
    border_delay_min: int
    border_delay_max: int
    border_delay_days_delayed: int
    cancellation_probability: float  # cancellation ~ bernoulli
    cancellation_days_delayed: int
    # Next two parameters are for quality risks, which have the same delay
    damage_probability: float  # damaged ~ binomial
    defective_probability: float  # defectives ~ binomial
    quality_days_delayed: int
    order_size: int
    tariff_escalation: float  # probability that there was a tariff escalation
