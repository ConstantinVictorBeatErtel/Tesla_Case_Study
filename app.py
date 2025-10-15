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

# white background
# Apply custom CSS for white background
st.markdown(
    """
    <style>
    .main {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# initialize session state to hold results (useful later)
if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

left_col, right_col = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: CONTROLS ---
with left_col:
    st.markdown("#### Configuration")

    # Renamed from 'target_order_size' to 'target_order_size' for clarity
    target_order_size = st.slider(
        "Anticipated Order Size",
        min_value=1_000,
        max_value=50_000,
        value=8_000,
        step=500,
        help="The target number of usable headlamps you want to receive.",
    )

    risk_tolerance = st.slider(
        "Risk Aversion (Lambda)",
        min_value=0.0,
        max_value=5.0,
        value=5.0,
        step=0.1,
        help="A higher value means you prioritize a more predictable (less risky) cost over the absolute lowest cost.",
    )

    run_button = st.button("Run", type="primary", use_container_width=True)

# --- PROCESSING LOGIC ---
if run_button:
    with st.spinner("Running simulations for all countries..."):
        # 1. run monte carlo simulation
        all_results = {
            country: run_monte_carlo(country, params, target_order_size)
            for country, params in COUNTRIES.items()
        }
        all_costs = {
            country: results["total_cost"] for country, results in all_results.items()
        }
        all_lost_units = {
            country: results["lost_units"] for country, results in all_results.items()
        }
        # create the per-unit cost dictionary specifically for the optimizer
        all_costs_per_lamp = {
            country: total_costs / target_order_size
            for country, total_costs in all_costs.items()
        }

        # 2. run the optimization
        result = optimize_without_yield(all_costs_per_lamp, risk_tolerance, None)

        if result:
            # 3. calculate portfolio-level metrics
            allocations = result["allocations"]

            # calculate weighted average of expected lost units
            portfolio_expected_lost_units = sum(
                np.mean(all_results[country]["lost_units"]) * allocations[country]
                for country in COUNTRIES.keys()
            )

            # calculate portfolio's overall yield rate (units received) / (units ordered)
            portfolio_yield_rate = (
                target_order_size - portfolio_expected_lost_units
            ) / target_order_size

            # calculate the recommended order size to meet the target
            if portfolio_yield_rate > 0:
                recommended_orders = int(target_order_size / portfolio_yield_rate)
            else:  # GUARD: division by 0
                recommended_orders = float("inf")

            # prepare allocation data for the pie chart
            alloc_df = pd.DataFrame(
                [
                    {"Country": country, "Weight": weight}
                    for country, weight in allocations.items()
                ]
            ).sort_values("Weight", ascending=False)

            # 4. store all results in session state for display
            st.session_state.optimization_results = {
                "blended_cost": result["expected_cost"],
                "recommended_orders": recommended_orders,
                "alloc_df": alloc_df,
            }
        else:
            st.error("Optimization failed. Please check parameters and try again.")
            st.session_state.optimization_results = None


# --- RIGHT COLUMN: OUTPUTS ---
with right_col:
    # Display results if they exist in the session state
    if st.session_state.optimization_results:
        results = st.session_state.optimization_results

        # Create columns for the metrics
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                label="Cost / Headlamp",
                value=f"${results['blended_cost']:.2f}",
                help="The expected cost per usable headlamp from the optimized blend of suppliers.",
            )

        with metric_col2:
            st.metric(
                label="Recommended Orders",
                value=f"{results['recommended_orders']:,}",
                help=f"To meet your target of {target_order_size:,} usable units, you should place a total order of this size.",
            )

        # Create and display the pie chart
        fig_pie = px.pie(
            results["alloc_df"],
            values="Weight",
            names="Country",
            hole=0.3,
            width=250,
            height=250,
        )

        # Update chart appearance to match wireframe
        fig_pie.update_traces(
            textinfo="percent+label",
            textposition="inside",
            pull=[0.05, 0, 0],  # Slightly pull out the largest slice
        )
        fig_pie.update_layout(
            title_text="",  # No title on the chart itself
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),  # Reduce whitespace
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        # Show a placeholder message before the first run
        st.info(
            "Adjust the parameters on the left and click **Run** to see the optimal supplier mix."
        )
