import numpy as np

from bayesian_priors.samplers import CountrySamplers, build_samplers_for_country
from config import MONTE_CARLO_SIMULATIONS
from discrete import (
    create_params_from_dict,
    generate_border_delay_risk,
    generate_damaged_risk,
    generate_defective_risk,
    generate_disruption_risk,
    generate_last_minute_cancellation_risk,
    generate_tariff_escalation,
)
from structs import DiscreteRisks, DiscreteRisksParams
from utils import sample_from_spec


def factory_costs_with_bayesian_priors(
    samplers: CountrySamplers, risk_params: dict, n_runs: int
) -> np.ndarray:
    """Run Monte Carlo simulation using Bayesian posterior samplers."""
    # Sample from posterior predictives
    raw = samplers.raw_material(n_runs)
    labor = samplers.labor(n_runs)
    logistics = samplers.logistics(n_runs)
    fx_mult = samplers.fx_multiplier(n_runs)
    yield_rate = samplers.yield_rate(n_runs)

    # Constants (no public data)
    indirect = sample_from_spec(risk_params["indirect"], n_runs)
    electricity = sample_from_spec(risk_params["electricity"], n_runs)
    depreciation = sample_from_spec(risk_params["depreciation"], n_runs)
    working = sample_from_spec(risk_params["working_capital"], n_runs)

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working
    base = base * fx_mult  # Apply FX volatility

    # Apply yield and tariff
    tariff = risk_params["tariff"]["fixed"]
    # if "fixed" in risk_params["tariff_escal"]:
    #     tariff += risk_params["tariff_escal"]["fixed"]
    # # TODO: modify
    # else:
    #     tariff += np.random.normal(
    #         risk_params["tariff_escal"]["mean"], risk_params["tariff_escal"]["std"]
    #     )
    total = base / yield_rate * (1+tariff)

    return total


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
    tariffs = generate_tariff_escalation(params.tariff_escalation)

    return DiscreteRisks(
        disruptions=disruption,
        border_delays=border_delay,
        damaged=damaged,
        defectives=defective,
        last_minute_cancellations=cancelled,
        tariff_cost=tariffs,
    )


def run_monte_carlo(country: str, params: dict, order_size: int) -> np.ndarray:
    """
    The main orchestrator for running simulations.
    Returns a distribution of the TOTAL COST for an entire order.
    """
    samplers = build_samplers_for_country(country, params)
    # 1. GENERATE THE DISTRIBUTION OF TOTAL BASE COSTS
    # this is PER-UNIT costs, one for each simulation run.
    base_cost_per_unit_dist = factory_costs_with_bayesian_priors(
        samplers, params, MONTE_CARLO_SIMULATIONS
    )

    # Then, scale it by the order size to get the TOTAL base cost for the order
    # for each of the simulation runs.
    total_base_cost_dist = base_cost_per_unit_dist * order_size

    # 2. GENERATE THE DISTRIBUTION OF TOTAL RISK COSTS
    risk_params = create_params_from_dict(params, order_size)
    risk_costs_for_order = []
    lost_units_for_order = []
    tariff_escalations_for_order = []

    for _ in range(MONTE_CARLO_SIMULATIONS):
        risk_scenario = generate_discrete_risks(risk_params)

        tariff_escalation = risk_scenario.tariff_cost
        # print(tariff_escalation)
        tariff_escalations_for_order.append(tariff_escalation)

        total_risk_cost = (
            risk_scenario.disruptions.cost
            + risk_scenario.border_delays.cost
            + risk_scenario.damaged.cost
            + risk_scenario.defectives.cost
            + risk_scenario.last_minute_cancellations.cost
        )
        risk_costs_for_order.append(total_risk_cost)

        total_lost_units = (
            risk_scenario.disruptions.delayed_units
            + risk_scenario.border_delays.delayed_units
            + risk_scenario.damaged.delayed_units
            + risk_scenario.defectives.delayed_units
            + risk_scenario.last_minute_cancellations.delayed_units
        )
        lost_units_for_order.append(total_lost_units)

    total_risk_cost_dist = np.array(risk_costs_for_order)
    total_lost_units_dist = np.array(lost_units_for_order)
    total_tariff_escalation_dist = np.array(tariff_escalations_for_order)

    # 3. COMBINE THE DISTRIBUTIONS
    # Add the two arrays element-wise. Each element represents one
    # complete, simulated future (one base cost scenario + one risk scenario).
    total_order_cost_dist = (
        total_base_cost_dist * (1 + total_tariff_escalation_dist)
    ) + total_risk_cost_dist

    return {
        "total_cost": total_order_cost_dist,
        "lost_units": total_lost_units_dist,
    }
