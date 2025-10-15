from numpy import sum
from numpy.random import binomial, choice, poisson, uniform

from config import (
    EXPEDITED_SHIPPING_COST_PER_HEADLAMP,
    FED_FUNDS_RATE,
    MODEL_Y_MANUFACTURING_COST,
    MODEL_Y_PROFIT,
    WACC,
)
from structs import DiscreteRiskSimulation, DiscreteRisksParams

# --- Model Functions ---


def opportunity_cost(delayed_units: int):
    return MODEL_Y_PROFIT * delayed_units * ((1 + FED_FUNDS_RATE) / 365)


def expedited_shipping_cost(delayed_units: int):
    return EXPEDITED_SHIPPING_COST_PER_HEADLAMP * delayed_units


def carry_cost(days_delayed: int):
    return WACC * MODEL_Y_MANUFACTURING_COST * days_delayed


def total_cost(delayed_units: int, days_delayed: int):
    return (
        opportunity_cost(delayed_units)
        + expedited_shipping_cost(delayed_units)
        + carry_cost(days_delayed)
    )


# --- Distribution Generators ---
def generate_tariff_escalation(tariff_escalation_probability: float) -> float:
    tariff_escalation = binomial(1, tariff_escalation_probability, 1)

    if tariff_escalation == 0:
        return 0
    else:
        return choice([0.25, 0.50, 1], 1)[0]


def generate_disruption_risk(
    disruption_lambda: float,
    min_impact: int,
    max_impact: int,
    disruption_days_delayed: int,
) -> DiscreteRiskSimulation:
    """Based on the total number of disruptions, estimate the # impacted units"""
    disruptions = poisson(disruption_lambda)

    if disruptions == 0:
        return DiscreteRiskSimulation(0, 0)

    severity = uniform(low=min_impact, high=max_impact, size=disruptions)
    total_lost = int(sum(severity))

    return DiscreteRiskSimulation(
        total_lost, total_cost(total_lost, disruption_days_delayed)
    )


def generate_border_delay_risk(
    border_delay_lambda: float,
    min_impact: int,
    max_impact: int,
    border_delay_days_delayed: int,
) -> DiscreteRiskSimulation:
    """Total number of border delays"""
    border_delays = poisson(border_delay_lambda)
    if border_delays == 0:
        return DiscreteRiskSimulation(0, 0)

    severity = uniform(low=min_impact, high=max_impact, size=border_delays)
    total_lost = int(sum(severity))
    return DiscreteRiskSimulation(
        total_lost, total_cost(total_lost, border_delay_days_delayed)
    )


def generate_damaged_risk(
    order_size: int, damage_probability: float, damage_days_delayed: int
) -> DiscreteRiskSimulation:
    """Total number of damaged units"""
    damaged_units = binomial(order_size, damage_probability)
    return DiscreteRiskSimulation(
        damaged_units, total_cost(damaged_units, damage_days_delayed)
    )


def generate_defective_risk(
    order_size: int, defective_probability: float, defective_days_delayed: int
) -> DiscreteRiskSimulation:
    """Total number of defective units"""
    defective_units = binomial(order_size, defective_probability)
    return DiscreteRiskSimulation(
        defective_units, total_cost(defective_units, defective_days_delayed)
    )


def generate_last_minute_cancellation_risk(
    cancellation_probability: float, order_size: int, cancellation_days_delayed: int
) -> DiscreteRiskSimulation:
    """They either cancel or they don't"""
    cancelled = binomial(1, cancellation_probability)
    return DiscreteRiskSimulation(
        # since they cancel the entire order, we need to multiply by the order size
        cancelled * order_size,
        total_cost(cancelled * order_size, cancellation_days_delayed),
    )


def create_params_from_dict(country_dict: dict, order_size: int) -> DiscreteRisksParams:
    """
    Reads a dictionary of parameters for a country and creates a
    structured DiscreteRisksParams object.
    """
    print(country_dict)
    return DiscreteRisksParams(
        order_size=order_size,
        disruption_lambda=country_dict["disruption_lambda"],
        disruption_min=country_dict["disruption_min_impact"],
        disruption_max=country_dict["disruption_max_impact"],
        disruption_days_delayed=country_dict["disruption_days_delayed"],
        border_delay_lambda=country_dict["border_delay_lambda"],
        border_delay_min=country_dict["border_min_impact"],
        border_delay_max=country_dict["border_max_impact"],
        border_delay_days_delayed=country_dict["border_days_delayed"],
        damage_probability=country_dict["damage_probability"],
        defective_probability=country_dict["defective_probability"],
        quality_days_delayed=country_dict["quality_days_delayed"],
        cancellation_probability=country_dict["cancellation_probability"],
        cancellation_days_delayed=country_dict["cancellation_days_delayed"],
        tariff_escalation=country_dict["tariff_escal"],
    )
