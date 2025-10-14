from numpy.random import binomial, poisson
from numpy.typing import NDArray

from live_data import get_most_recent_fed_funds_rate
from structs import DiscreteRisks, DiscreteRiskSimulation, DiscreteRisksParams

# --- Model Parameters ---
# these parameters are to explain how a delayed part can be mapped to a financial cost

model_y_price = 41630
# SOURCE: https://www.motor1.com/news/775008/tesla-model-y-standard-pricing-range-details/
model_y_manufacturing_cost = 38000
# SOURCE: https://www.bitauto.com/wiki/100132610482/#:~:text=The%20battery%20pack%20accounts%20for,for%20approximately%2015%25%2D20%25.
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
def generate_disruption_risk(disruption_lambda: float, n_runs: int) -> NDArray[int]:
    travel_disruptions = poisson(disruption_lambda)
    # TODO:

    # from these disruptions, how many units got impacted?    # from these disruptions, how many units got impacted?


def generate_border_delay_risk(border_delay_lambda: float, n_runs: int) -> NDArray[int]:
    # TODO
    pass


def generate_damaged_risk(
    order_size: int, damage_probability: float, n_runs: int
) -> NDArray[int]:
    return binomial(order_size, damage_probability, n_runs)


def generate_defective_risk(
    order_size: int, defective_probability: float, n_runs: int
) -> NDArray[int]:
    return binomial(order_size, defective_probability, n_runs)


def generate_last_minute_cancellation_risk(
    cancellation_probability: float, n_runs: int
) -> NDArray[int]:
    # they either cancel or they don't, so no order size here
    return binomial(1, cancellation_probability, n_runs)


# --- Main Function ---
def generate_discrete_risks(params: DiscreteRisksParams, n_runs: int) -> DiscreteRisks:
    disruptions = generate_disruption_risk(params.disruption_lambda, n_runs)
    disruption_costs = total_cost(disruptions.delayed_units, params.days_delayed)

    border_delays = generate_border_delay_risk(params.border_delay_lambda, n_runs)
    border_delay_costs = total_cost(border_delays.delayed_units, params.days_delayed)

    damaged = generate_damaged_risk(params.damage_probability, n_runs)
    damaged_costs = total_cost(damaged.delayed_units, params.days_delayed)

    defectives = generate_defective_risk(params.defective_probability, n_runs)
    defectives_costs = total_cost(defectives.delayed_units, params.days_delayed)

    last_minute_cancellations = generate_last_minute_cancellation_risk(
        params.cancellation_probability, n_runs
    )
    last_minute_cancellation_costs = total_cost(
        last_minute_cancellations.delayed_units, params.days_delayed
    )

    return DiscreteRisks(
        DiscreteRiskSimulation(disruptions.delayed_units, disruption_costs),
        DiscreteRiskSimulation(border_delays.delayed_units, border_delay_costs),
        DiscreteRiskSimulation(
            last_minute_cancellations.delayed_units, last_minute_cancellation_costs
        ),
        DiscreteRiskSimulation(damaged.delayed_units, damaged_costs),
        DiscreteRiskSimulation(defectives.delayed_units, defectives_costs),
    )
