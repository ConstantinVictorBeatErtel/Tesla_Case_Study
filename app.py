import copy

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import COUNTRIES
from simulation import run_monte_carlo
from utils import optimize_without_yield

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

# --- Global Simulation Section ---
if st.button("Run Global Simulation"):
    with st.spinner("Running simulations for all countries..."):
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

    # st.markdown("---")  # Visual separator

    # ## 2. SENSITIVITY ANALYSIS
    # st.subheader("Sensitivity Analysis")
    # sens_col1, sens_col2 = st.columns([1, 2])

    # with sens_col1:
    #     st.markdown("#### Configure Analysis")
    #     country_to_analyze = st.selectbox(
    #         "Select Country to Analyze:", list(COUNTRIES.keys()), key="sens_country"
    #     )

    #     # Define parameters that can be tested. You can add more here.
    #     testable_params = [
    #         "disruption_lambda",
    #         "cancellation_probability",
    #         "damage_probability",
    #         "raw_mean",  # Example of a factory cost parameter
    #     ]
    #     param_to_test = st.selectbox("Select Parameter to Test:", testable_params)

    #     sensitivity_range = st.slider(
    #         "Select Variation Range (%):",
    #         min_value=-100,
    #         max_value=100,
    #         value=(-50, 50),
    #         step=10,
    #     )
    #     SENSITIVITY_STEPS = 10  # Number of points to calculate in the range

    # with sens_col2:
    #     with st.spinner(
    #         f"Running sensitivity analysis on '{param_to_test}' for {country_to_analyze}..."
    #     ):
    #         base_params = copy.deepcopy(COUNTRIES[country_to_analyze])

    #         # Find the original value of the parameter
    #         if param_to_test == "raw_mean":  # Handle nested parameters
    #             original_value = base_params["raw"]["mean"]
    #         else:
    #             original_value = base_params[param_to_test]

    #         param_values = np.linspace(
    #             original_value * (1 + sensitivity_range[0] / 100),
    #             original_value * (1 + sensitivity_range[1] / 100),
    #             SENSITIVITY_STEPS,
    #         )

    #         expected_costs = []
    #         for val in param_values:
    #             temp_params = copy.deepcopy(base_params)
    #             # Update the parameter in the temporary dictionary
    #             if param_to_test == "raw_mean":
    #                 temp_params["raw"]["mean"] = val
    #             else:
    #                 temp_params[param_to_test] = val

    #             # Run a smaller simulation for speed
    #             results = run_monte_carlo(country_to_analyze, temp_params, order_size)
    #             avg_total_cost = np.mean(results["total_cost"])
    #             expected_costs.append(avg_total_cost / order_size)

    #         df_sens = pd.DataFrame(
    #             {
    #                 "Parameter Value": param_values,
    #                 "Expected Cost per Lamp": expected_costs,
    #             }
    #         )

    #         fig_sens = px.line(
    #             df_sens,
    #             x="Parameter Value",
    #             y="Expected Cost per Lamp",
    #             title=f"Impact of '{param_to_test}' on Cost in {country_to_analyze}",
    #             markers=True,
    #         )
    #         fig_sens.update_layout(
    #             xaxis_title=f"Value of {param_to_test}",
    #             yaxis_title="Expected Cost ($/Lamp)",
    #         )
    #         st.plotly_chart(fig_sens, use_container_width=True)
    #         st.caption(
    #             "This chart shows how the final expected cost changes as a single input parameter is varied. A steep slope indicates the model is highly sensitive to that parameter."
    #         )

    # # --- END: NEW SECTIONS ---

    # st.markdown("---")


st.markdown("---")

# --- Portfolio Optimization UI ---
st.header("Portfolio Optimization (Mean-Variance)")
st.write(
    "Find the optimal sourcing allocation that minimizes: **E[Cost] + λ × SD[Cost]**"
)
st.write(
    "This balances expected cost against risk (volatility). Higher λ means you care more about reducing risk."
)

opt_col1, opt_col2 = st.columns([1, 2])
with opt_col1:
    st.subheader("Optimization Settings")

    lambda_risk = st.slider(
        "Risk Aversion (λ)",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1,
        help="λ=0: minimize expected cost only. Higher λ: care more about reducing risk.",
    )

    st.markdown("##### Optional Constraints")
    add_constraints = st.checkbox("Add min/max allocation constraints")

    constraints = {}
    if add_constraints:
        for country in COUNTRIES.keys():
            col_a, col_b = st.columns(2)
            with col_a:
                min_val = st.number_input(
                    f"{country} Min %", 0, 100, 0, 5, key=f"min_{country}"
                )
            with col_b:
                max_val = st.number_input(
                    f"{country} Max %", 0, 100, 100, 5, key=f"max_{country}"
                )
            constraints[country] = (min_val / 100, max_val / 100)

    run_optimization = st.button("🎯 Run Optimization", type="primary")

if run_optimization:
    with st.spinner("Running optimization..."):
        # STEP 1: Run simulations to get BOTH costs and lost units
        all_results = {
            country: run_monte_carlo(country, params, order_size)
            for country, params in COUNTRIES.items()
        }

        # A) Separate the results for clarity
        all_costs = {
            country: results["total_cost"] for country, results in all_results.items()
        }
        all_lost_units = {
            country: results["lost_units"] for country, results in all_results.items()
        }
        # Create the per-unit cost dictionary specifically for the optimizer
        all_costs_per_lamp = {
            country: total_costs / order_size
            for country, total_costs in all_costs.items()
        }

        # B) CALCULATE EXPECTED YIELD FOR EACH COUNTRY
        # Yield = (units ordered - expected lost units) / units ordered
        expected_yields = {}
        for country, results in all_results.items():
            expected_lost_units = np.mean(results["lost_units"])
            yield_rate = (order_size - expected_lost_units) / order_size
            expected_yields[country] = yield_rate

        print(expected_yields)

        # STEP 2: Run the financial optimization (this part is unchanged)
        opt_constraints = constraints if add_constraints else None
        # result = optimize_portfolio(
        #     all_costs_per_lamp, expected_yields, lambda_risk, opt_constraints
        # )
        result = optimize_without_yield(
            all_costs_per_lamp, lambda_risk, opt_constraints
        )

        if result:
            with opt_col2:
                st.subheader("Optimal Portfolio Performance")

                # --- NEW: Calculate and Display Expected Units Received ---

                # Calculate the expected number of lost units for each country
                expected_losses = {
                    country: np.mean(units) for country, units in all_lost_units.items()
                }

                # Get the optimal allocation weights from the optimizer
                allocations = result["allocations"]

                # Calculate the weighted average of lost units for the entire portfolio
                portfolio_expected_lost_units = sum(
                    expected_losses[country] * allocations[country]
                    for country in COUNTRIES.keys()
                )

                # Calculate the final number of units you expect to receive
                expected_units_received = order_size - portfolio_expected_lost_units

                # Display the new metrics
                st.metric(label="Initial Order Size", value=f"{order_size:,}")
                st.metric(
                    label="Expected Units Received",
                    value=f"{int(expected_units_received):,}",
                    help=f"Based on the optimal mix, you can expect to lose {int(portfolio_expected_lost_units):,} units to various risks.",
                )

                portfolio_yield = (expected_units_received / order_size) * 100
                st.metric(label="Portfolio Yield", value=f"{portfolio_yield:.2f}%")

                st.markdown("---")  # Visual separator

                # --- The rest of your existing display code ---
                # st.markdown("##### Financial Outcome")
                # st.metric("Expected Cost per Unit", f"${result['expected_cost']:.2f}")
                # st.metric("Standard Deviation per Unit", f"${result['std_cost']:.2f}")

                st.markdown("##### Optimal Allocations")

                alloc_df = pd.DataFrame(
                    [
                        {
                            "Country": country,
                            "Allocation": f"{weight * 100:.1f}%",
                            "Weight": weight,
                        }
                        for country, weight in result["allocations"].items()
                    ]
                ).sort_values("Weight", ascending=False)
                st.dataframe(alloc_df[["Country", "Allocation"]], hide_index=True)

                # Pie chart of allocations
                fig_pie = px.pie(
                    alloc_df,
                    values="Weight",
                    names="Country",
                    title="Optimal Allocation Mix",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.error(
                "Optimization failed. Try relaxing constraints or adjusting parameters."
            )

st.markdown("---")


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
