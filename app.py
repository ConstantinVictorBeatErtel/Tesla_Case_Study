import copy

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import minimize

from config import COUNTRIES
from simulation import run_monte_carlo


# --- Portfolio Optimization Function ---
def optimize_portfolio(all_costs, lambda_risk, constraints=None):
    """
    Optimizes portfolio allocation to minimize E[Cost] + lambda * SD[Cost]

    Args:
        all_costs: dict of {country: cost_array}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None

    Returns:
        optimal allocations, expected cost, std dev
    """
    countries_list = list(all_costs.keys())
    n_countries = len(countries_list)

    def objective(weights):
        # Calculate portfolio cost for all runs
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        for i, country in enumerate(countries_list):
            portfolio += weights[i] * all_costs[country]

        expected_cost = np.mean(portfolio)
        std_cost = np.std(portfolio)

        return expected_cost + lambda_risk * std_cost

    # Constraints: weights sum to 1, all weights >= 0
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = []

    # Add custom constraints if provided
    if constraints:
        for country in countries_list:
            if country in constraints:
                min_alloc, max_alloc = constraints[country]
                bounds.append((min_alloc, max_alloc))
            else:
                bounds.append((0, 1))
    else:
        bounds = [(0, 1) for _ in range(n_countries)]

    # Initial guess: equal allocation
    x0 = np.ones(n_countries) / n_countries

    # Optimize
    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=[constraint_sum]
    )

    if result.success:
        optimal_weights = result.x
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        for i, country in enumerate(countries_list):
            portfolio += optimal_weights[i] * all_costs[country]

        return {
            "allocations": {
                countries_list[i]: w for i, w in enumerate(optimal_weights)
            },
            "expected_cost": np.mean(portfolio),
            "std_cost": np.std(portfolio),
            "portfolio_costs": portfolio,
        }
    else:
        return None


# --- Sensitivity Analysis Function ---
def run_sensitivity_analysis(base_params, factors_to_test, n_runs, swing=0.20):
    """
    Runs a one-at-a-time sensitivity analysis for a given set of factors.
    """
    results = []

    # Calculate baseline mean cost
    # base_costs = simulate_country(base_params, n_runs)
    base_costs = run_monte_carlo(base_params, n_runs)
    baseline_mean = np.mean(base_costs)

    for factor_name, param_path in factors_to_test:
        params_low = copy.deepcopy(base_params)
        params_high = copy.deepcopy(base_params)

        # Get the base value using the path
        base_value = base_params
        for key in param_path:
            base_value = base_value[key]

        low_value = base_value * (1 - swing)
        high_value = base_value * (1 + swing)

        # Set the low and high values in the copied params
        temp_low = params_low
        temp_high = params_high
        for i, key in enumerate(param_path):
            if i == len(param_path) - 1:
                temp_low[key] = low_value
                temp_high[key] = high_value
            else:
                temp_low = temp_low[key]
                temp_high = temp_high[key]

        # Simulate with low and high values
        # mean_low = np.mean(simulate_country(params_low, n_runs))
        mean_low = np.mean(run_monte_carlo(params_low, n_runs))
        mean_high = np.mean(run_monte_carlo(params_high, n_runs))

        results.append(
            {
                "Factor": factor_name,
                "Low Cost": mean_low,
                "High Cost": mean_high,
                "Impact": mean_high - mean_low,
            }
        )

    return pd.DataFrame(results), baseline_mean


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Tesla Headlamp Supplier Evaluation")
st.write(
    "Assuming we need to deliver a number of headlamps, what is the optimal procurement strategy across US, Mexico, and China?"
)

order_size = st.number_input(
    "Number of Headlamps",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000,
)
# Toggle for Bayesian approach
use_bayesian = st.checkbox(
    "Use Bayesian Priors (fetch real data from FRED)",
    value=False,
    help="Enable to use real economic data (PPI, FX rates) with Bayesian posterior estimation. This accounts for parameter uncertainty and gives more conservative risk estimates.",
)

# --- Global Simulation Section ---
if st.button("Run Global Simulation"):
    with st.spinner("Running simulations for all countries..."):
        # This now stores a dictionary of dictionaries, e.g., {'US': {'total_cost': [...], 'lost_units': [...]}}
        all_results = {
            country: run_monte_carlo(country, params, order_size)
            for country, params in COUNTRIES.items()
        }

        st.subheader("Summary Statistics")
        cols = st.columns(len(COUNTRIES))

        for i, (country, results) in enumerate(all_results.items()):
            total_cost_dist = results["total_cost"]
            lost_units_dist = results["lost_units"]
            costs_per_lamp = total_cost_dist / order_size

            with cols[i]:
                st.markdown(f"### {country}")

                # METRIC 1: COST PER HEADLAMP (Calculated from total cost)
                expected_cost_per_lamp = np.mean(costs_per_lamp)
                st.metric(
                    label="Expected Cost per Lamp",
                    value=f"${expected_cost_per_lamp:.2f}",
                )

                # METRIC 2: RECOMMENDED ORDER SIZE (Calculated from lost units)
                expected_lost_units = np.mean(lost_units_dist)
                recommended_order_size = order_size + int(expected_lost_units)
                st.metric(
                    label="Recommended Order Size",
                    value=f"{recommended_order_size:,}",
                    help=f"To meet a target of {order_size:,}, you should order this many units to account for expected losses of {int(expected_lost_units):,} units.",
                )

                st.metric(
                    label="Standard Deviation", value=f"${np.std(costs_per_lamp):.2f}"
                )

                st.metric(
                    label="95th Percentile Cost",
                    value=f"${np.percentile(costs_per_lamp, 95):.2f}",
                )

        # Create the per-lamp cost dictionary for plotting
        all_costs_per_lamp = {
            country: results["total_cost"] / order_size
            for country, results in all_results.items()
        }

    # --- START: NEW SECTIONS ---

    st.markdown("---")  # Visual separator

    ## 1. MONTE CARLO CONVERGENCE PLOT
    st.subheader("Monte Carlo Convergence Plot")
    selected_country_conv = st.selectbox(
        "Select Country to View Convergence:", list(COUNTRIES.keys())
    )

    if selected_country_conv:
        costs = all_costs_per_lamp[selected_country_conv]
        # Calculate the running average
        running_mean = np.cumsum(costs) / np.arange(1, len(costs) + 1)

        df_conv = pd.DataFrame(
            {
                "Simulation Number": np.arange(1, len(costs) + 1),
                "Running Average Cost per Lamp": running_mean,
            }
        )

        fig_conv = px.line(
            df_conv,
            x="Simulation Number",
            y="Running Average Cost per Lamp",
            title=f"Convergence of Average Cost for {selected_country_conv}",
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        st.caption(
            "This chart shows how the average cost stabilizes as more simulations are run. A flat line towards the end indicates that the simulation has converged and the number of runs is sufficient."
        )

    st.markdown("---")  # Visual separator

    ## 2. SENSITIVITY ANALYSIS
    st.subheader("Sensitivity Analysis")
    sens_col1, sens_col2 = st.columns([1, 2])

    with sens_col1:
        st.markdown("#### Configure Analysis")
        country_to_analyze = st.selectbox(
            "Select Country to Analyze:", list(COUNTRIES.keys()), key="sens_country"
        )

        # Define parameters that can be tested. You can add more here.
        testable_params = [
            "disruption_lambda",
            "cancellation_probability",
            "damage_probability",
            "raw_mean",  # Example of a factory cost parameter
        ]
        param_to_test = st.selectbox("Select Parameter to Test:", testable_params)

        sensitivity_range = st.slider(
            "Select Variation Range (%):",
            min_value=-100,
            max_value=100,
            value=(-50, 50),
            step=10,
        )
        SENSITIVITY_STEPS = 10  # Number of points to calculate in the range

    with sens_col2:
        with st.spinner(
            f"Running sensitivity analysis on '{param_to_test}' for {country_to_analyze}..."
        ):
            base_params = copy.deepcopy(COUNTRIES[country_to_analyze])

            # Find the original value of the parameter
            if param_to_test == "raw_mean":  # Handle nested parameters
                original_value = base_params["raw"]["mean"]
            else:
                original_value = base_params[param_to_test]

            param_values = np.linspace(
                original_value * (1 + sensitivity_range[0] / 100),
                original_value * (1 + sensitivity_range[1] / 100),
                SENSITIVITY_STEPS,
            )

            expected_costs = []
            for val in param_values:
                temp_params = copy.deepcopy(base_params)
                # Update the parameter in the temporary dictionary
                if param_to_test == "raw_mean":
                    temp_params["raw"]["mean"] = val
                else:
                    temp_params[param_to_test] = val

                # Run a smaller simulation for speed
                results = run_monte_carlo(country_to_analyze, temp_params, order_size)
                avg_total_cost = np.mean(results["total_cost"])
                expected_costs.append(avg_total_cost / order_size)

            df_sens = pd.DataFrame(
                {
                    "Parameter Value": param_values,
                    "Expected Cost per Lamp": expected_costs,
                }
            )

            fig_sens = px.line(
                df_sens,
                x="Parameter Value",
                y="Expected Cost per Lamp",
                title=f"Impact of '{param_to_test}' on Cost in {country_to_analyze}",
                markers=True,
            )
            fig_sens.update_layout(
                xaxis_title=f"Value of {param_to_test}",
                yaxis_title="Expected Cost ($/Lamp)",
            )
            st.plotly_chart(fig_sens, use_container_width=True)
            st.caption(
                "This chart shows how the final expected cost changes as a single input parameter is varied. A steep slope indicates the model is highly sensitive to that parameter."
            )

    # --- END: NEW SECTIONS ---

    st.markdown("---")


st.markdown("---")

# --- Portfolio Optimization UI ---
# st.header("Portfolio Optimization (Mean-Variance)")
# st.write(
#     "Find the optimal sourcing allocation that minimizes: **E[Cost] + λ × SD[Cost]**"
# )
# st.write(
#     "This balances expected cost against risk (volatility). Higher λ means you care more about reducing risk."
# )

# opt_col1, opt_col2 = st.columns([1, 2])
# with opt_col1:
#     st.subheader("Optimization Settings")

#     lambda_risk = st.slider(
#         "Risk Aversion (λ)",
#         min_value=0.0,
#         max_value=5.0,
#         value=1.0,
#         step=0.1,
#         help="λ=0: minimize expected cost only. Higher λ: care more about reducing risk.",
#     )

#     st.markdown("##### Optional Constraints")
#     add_constraints = st.checkbox("Add min/max allocation constraints")

#     constraints = {}
#     if add_constraints:
#         for country in countries.keys():
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 min_val = st.number_input(
#                     f"{country} Min %", 0, 100, 0, 5, key=f"min_{country}"
#                 )
#             with col_b:
#                 max_val = st.number_input(
#                     f"{country} Max %", 0, 100, 100, 5, key=f"max_{country}"
#                 )
#             constraints[country] = (min_val / 100, max_val / 100)

#     run_optimization = st.button("🎯 Run Optimization", type="primary")

# if run_optimization:
#     with st.spinner("Running optimization..."):
#         # Run simulations
#         all_costs = {
#             country: run_monte_carlo(params, n_runs)
#             for country, params in countries.items()
#         }

#         # Optimize
#         opt_constraints = constraints if add_constraints else None
#         result = optimize_portfolio(all_costs, lambda_risk, opt_constraints)

#         if result:
#             with opt_col2:
#                 st.subheader("Optimal Portfolio")
#                 st.metric("Expected Cost", f"${result['expected_cost']:.2f}")
#                 st.metric("Standard Deviation", f"${result['std_cost']:.2f}")
#                 st.metric(
#                     "Objective Value",
#                     f"${result['expected_cost'] + lambda_risk * result['std_cost']:.2f}",
#                 )

#                 st.markdown("##### Optimal Allocations")
#                 alloc_df = pd.DataFrame(
#                     [
#                         {
#                             "Country": country,
#                             "Allocation": f"{weight * 100:.1f}%",
#                             "Weight": weight,
#                         }
#                         for country, weight in result["allocations"].items()
#                     ]
#                 ).sort_values("Weight", ascending=False)
#                 st.dataframe(alloc_df[["Country", "Allocation"]], hide_index=True)

#                 # Pie chart of allocations
#                 fig_pie = px.pie(
#                     alloc_df,
#                     values="Weight",
#                     names="Country",
#                     title="Optimal Allocation Mix",
#                 )
#                 st.plotly_chart(fig_pie, use_container_width=True)
#         else:
#             st.error(
#                 "Optimization failed. Try relaxing constraints or adjusting parameters."
#             )

# st.markdown("---")

# # --- Sensitivity Analysis UI ---
# st.header("Sensitivity Analysis")
# st.write(
#     "Analyze which factors have the biggest impact on the total cost for a specific country. This chart shows how a +/- 20% change in each input variable affects the expected total cost."
# )

# sa_col1, sa_col2 = st.columns([1, 3])
# with sa_col1:
#     sa_country = st.selectbox(
#         "Select Country to Analyze", list(countries.keys()), key="sa_country"
#     )
#     run_sa = st.button("Run Sensitivity Analysis")

# if run_sa:
#     with st.spinner(f"Running sensitivity analysis for {sa_country}..."):
#         base_params = countries[sa_country]

#         # Define factors to test for each country
#         factors = {
#             "US": [
#                 ("Raw Material Mean", ("raw", "mean")),
#                 ("Labor Mean", ("labor", "mean")),
#                 ("Working Capital Mean", ("working_capital", "mean")),
#                 ("Manufacturing Yield", ("yield_params", "a")),
#                 ("Disruption Probability", ("disruption_prob",)),
#             ],
#             "Mexico": [
#                 ("Raw Material Mean", ("raw", "mean")),
#                 ("Labor Mean", ("labor", "mean")),
#                 ("Tariff (Fixed)", ("tariff", "fixed")),
#                 ("Currency Volatility", ("currency_std",)),
#                 ("Border Crossing Time", ("border_mean",)),
#                 ("Manufacturing Yield", ("yield_params", "a")),
#             ],
#             "China": [
#                 ("Raw Material Mean", ("raw", "mean")),
#                 ("Logistics Mean", ("logistics", "mean")),
#                 ("Logistics Volatility", ("logistics", "std")),
#                 ("Cancellation Probability", ("cancellation_prob",)),
#                 ("Manufacturing Yield", ("yield_params", "a")),
#                 ("Tariff (Fixed)", ("tariff", "fixed")),
#             ],
#         }

#         factors_to_test = factors[sa_country]
#         sa_results, baseline_mean = run_sensitivity_analysis(
#             base_params, factors_to_test, n_runs
#         )

#         sa_results = sa_results.sort_values(by="Impact", ascending=True)

#         # Create Tornado Plot
#         fig = go.Figure()
#         fig.add_trace(
#             go.Bar(
#                 y=sa_results["Factor"],
#                 x=sa_results["High Cost"] - baseline_mean,
#                 name="High Estimate (Input +20%)",
#                 orientation="h",
#                 marker_color="indianred",
#             )
#         )
#         fig.add_trace(
#             go.Bar(
#                 y=sa_results["Factor"],
#                 x=sa_results["Low Cost"] - baseline_mean,
#                 name="Low Estimate (Input -20%)",
#                 orientation="h",
#                 marker_color="lightblue",
#             )
#         )

#         fig.update_layout(
#             title=f"Tornado Plot for {sa_country} (Baseline Cost: ${baseline_mean:.2f})",
#             xaxis_title="Impact on Total Cost ($/lamp)",
#             yaxis_title="Sensitivity Factor",
#             barmode="relative",
#             yaxis_autorange="reversed",
#             legend=dict(x=0.01, y=0.01, traceorder="normal"),
#             margin=dict(l=150),  # Add left margin for long factor names
#         )

#         with sa_col2:
#             st.plotly_chart(fig, use_container_width=True)
