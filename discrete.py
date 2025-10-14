from numpy import sum
from numpy.random import binomial, poisson, uniform

from live_data import get_most_recent_fed_funds_rate
from structs import DiscreteRisks, DiscreteRiskSimulation, DiscreteRisksParams

# --- Model Parameters ---
# these parameters are to explain how a delayed part can be mapped to a financial cost

model_y_price = 41630
model_y_manufacturing_cost = 38000
model_y_profit = model_y_price - model_y_manufacturing_cost
wacc = 0.0877
expedited_shipping_cost_per_unit = 50.71
fed_funds_rate = get_most_recent_fed_funds_rate()

# --- Model Functions ---
# below calculate the costs associated with a delayed part


def opportunity_cost(delayed_units: int):
    return model_y_profit * delayed_units * ((1 + fed_funds_rate) / 365)


def expedited_shipping_cost(delayed_units: int):
    return expedited_shipping_cost_per_unit * delayed_units


def carry_cost(days_delayed: int):
    return wacc * model_y_manufacturing_cost * days_delayed


def total_cost(delayed_units: int, days_delayed: int):
    return (
        opportunity_cost(delayed_units)
        + expedited_shipping_cost(delayed_units)
        + carry_cost(days_delayed)
    )


# --- Distribution Generators ---
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


# --- Main Function ---
def generate_discrete_risks(params: DiscreteRisksParams) -> DiscreteRisks:
    damaged = generate_damaged_risk(
        params.order_size, params.damage_probability, params.quality_days_delayed
    )
    defective = generate_defective_risk(
        params.order_size, params.defective_probability, params.quality_days_delayed
    )
    cancelled = generate_last_minute_cancellation_risk(
        params.cancellation_probability,
        params.order_size,
        params.cancellation_days_delayed,
    )
    border_delay = generate_border_delay_risk(
        params.border_delay_lambda,
        params.border_delay_min,
        params.border_delay_max,
        params.border_delay_days_delayed,
    )
    disruption = generate_disruption_risk(
        params.disruption_lambda,
        params.disruption_min,
        params.disruption_max,
        params.disruption_days_delayed,
    )

    return DiscreteRisks(
        disruptions=disruption,
        border_delays=border_delay,
        damaged=damaged,
        defectives=defective,
        last_minute_cancellations=cancelled,
    )


def create_params_from_dict(country_dict: dict) -> DiscreteRisksParams:
    """
    Reads a dictionary of parameters for a country and creates a
    structured DiscreteRisksParams object.
    """
    return DiscreteRisksParams(
        order_size=country_dict["order_size"],
        disruption_lambda=country_dict["disruption_lambda"],
        disruption_min_impact=country_dict["disruption_min_impact"],
        disruption_max_impact=country_dict["disruption_max_impact"],
        disruption_days_delayed=country_dict["disruption_days_delayed"],
        border_delay_lambda=country_dict["border_delay_lambda"],
        border_min_impact=country_dict["border_min_impact"],
        border_max_impact=country_dict["border_max_impact"],
        border_days_delayed=country_dict["border_days_delayed"],
        damage_probability=country_dict["damage_probability"],
        defective_probability=country_dict["defective_probability"],
        quality_days_delayed=country_dict["quality_days_delayed"],
        cancellation_probability=country_dict["cancellation_probability"],
        cancellation_days_delayed=country_dict["cancellation_days_delayed"],
    )
