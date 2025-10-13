import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import beta
from scipy.optimize import minimize
import copy

# --- Parameters ---
# Parameters based on researched values (means from case, std/dist from sources like USITC, FHWA, etc.)
countries = {
    'US': {
        'raw': {'dist': 'normal', 'mean': 40, 'std': 4},
        'labor': {'dist': 'normal', 'mean': 12, 'std': 0.6},
        'indirect': {'dist': 'normal', 'mean': 10, 'std': 0.5},
        'logistics': {'dist': 'normal', 'mean': 9, 'std': 0},
        'electricity': {'dist': 'normal', 'mean': 4, 'std': 0.4},
        'depreciation': {'dist': 'normal', 'mean': 5, 'std': 0.25},
        'working_capital': {'dist': 'normal', 'mean': 5, 'std': 0.5},
        'yield_params': {'a': 79, 'b': 20},  # Approx for mean 0.8, std 0.04
        'tariff': {'fixed': 0},
        'tariff_escal': {'mean': 0, 'std': 0},
        'currency_std': 0,
        'disruption_prob': 0.05,
        'disruption_impact': 10,
        'border_mean': 0,
        'border_std': 0,
        'border_threshold': 2,
        'border_cost_per_hr': 10,
        'damage_prob': 0.01,
        'damage_impact': 20,
        'skills_mean': 0,
        'skills_std': 0,
        'cancellation_prob': 0,
        'cancellation_impact': 50
    },
    'Mexico': {
        'raw': {'dist': 'normal', 'mean': 35, 'std': 3.5},
        'labor': {'dist': 'normal', 'mean': 8, 'std': 0.4},
        'indirect': {'dist': 'normal', 'mean': 8, 'std': 0.4},
        'logistics': {'dist': 'normal', 'mean': 7, 'std': 0.056},
        'electricity': {'dist': 'normal', 'mean': 3, 'std': 0.3},
        'depreciation': {'dist': 'normal', 'mean': 1, 'std': 0.05},
        'working_capital': {'dist': 'normal', 'mean': 6, 'std': 0.6},
        'yield_params': {'a': 12, 'b': 1},  # Approx for mean 0.9, std 0.08
        'tariff': {'fixed': 15.5},
        'tariff_escal': {'mean': 0, 'std': 2},
        'currency_std': 0.08,
        'disruption_prob': 0.1,
        'disruption_impact': 10,
        'border_mean': 0.83,
        'border_std': 0.67,
        'border_threshold': 2,
        'border_cost_per_hr': 10,
        'damage_prob': 0.015,
        'damage_impact': 20,
        'skills_mean': 0,
        'skills_std': 0.05,
        'cancellation_prob': 0,
        'cancellation_impact': 50
    },
    'China': {
        'raw': {'dist': 'normal', 'mean': 30, 'std': 3},
        'labor': {'dist': 'normal', 'mean': 4, 'std': 0.2},
        'indirect': {'dist': 'normal', 'mean': 4, 'std': 0.2},
        'logistics': {'dist': 'lognormal', 'mean': 12, 'std': 8},  # High volatility from trade disruptions
        'electricity': {'dist': 'normal', 'mean': 4, 'std': 0.4},
        'depreciation': {'dist': 'normal', 'mean': 5, 'std': 0.25},
        'working_capital': {'dist': 'normal', 'mean': 10, 'std': 1},
        'yield_params': {'a': 49, 'b': 3},  # Approx for mean 0.95, std 0.03
        'tariff': {'fixed': 15},
        'tariff_escal': {'mean': 0, 'std': 2},
        'currency_std': 0.03,
        'disruption_prob': 0.2,
        'disruption_impact': 10,
        'border_mean': 0,
        'border_std': 0,
        'border_threshold': 2,
        'border_cost_per_hr': 10,
        'damage_prob': 0.02,
        'damage_impact': 20,
        'skills_mean': 0,
        'skills_std': 0,
        'cancellation_prob': 0.3,  # Updated from recent shipping data (30% cancellations)
        'cancellation_impact': 50
    }
}

# --- Simulation Function ---
def simulate_country(params, n_runs):
    """Runs a Monte Carlo simulation for a single country's sourcing cost."""
    # Sample costs from distributions
    raw = np.random.normal(params['raw']['mean'], params['raw']['std'], n_runs)
    labor = np.random.normal(params['labor']['mean'], params['labor']['std'], n_runs)
    indirect = np.random.normal(params['indirect']['mean'], params['indirect']['std'], n_runs)
    electricity = np.random.normal(params['electricity']['mean'], params['electricity']['std'], n_runs)
    depreciation = np.random.normal(params['depreciation']['mean'], params['depreciation']['std'], n_runs)
    working = np.random.normal(params['working_capital']['mean'], params['working_capital']['std'], n_runs)
    yield_ = beta.rvs(params['yield_params']['a'], params['yield_params']['b'], size=n_runs)

    # Handle lognormal distribution for logistics
    if params['logistics']['dist'] == 'lognormal':
        m, s = params['logistics']['mean'], params['logistics']['std']
        sigma = np.sqrt(np.log(1 + (s**2 / m**2)))
        mu = np.log(m) - (sigma**2 / 2)
        logistics = np.random.lognormal(mu, sigma, n_runs)
    else:
        logistics = np.random.normal(params['logistics']['mean'], params['logistics']['std'], n_runs)

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working

    # Apply currency fluctuation
    base *= (1 + np.random.normal(0, params['currency_std'], n_runs))

    # Apply tariff and potential escalation
    tariff = np.full(n_runs, params['tariff']['fixed']) + np.random.normal(params['tariff_escal']['mean'], params['tariff_escal']['std'], n_runs)

    # Calculate total cost before discrete risks
    total = base / yield_ + tariff

    # --- Add Discrete Risk Events ---
    disruption = np.random.binomial(1, params['disruption_prob'], n_runs)
    total += disruption * params['disruption_impact']
    border_time = np.random.normal(params['border_mean'], params['border_std'], n_runs)
    border_cost = np.maximum(0, border_time - params['border_threshold']) * params['border_cost_per_hr']
    total += border_cost
    damage = np.random.binomial(1, params['damage_prob'], n_runs)
    total += damage * params['damage_impact']
    skills_adj = np.random.normal(params['skills_mean'], params['skills_std'], n_runs)
    total *= (1 + skills_adj)
    cancellation = np.random.binomial(1, params['cancellation_prob'], n_runs)
    total += cancellation * params['cancellation_impact']

    return total

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
    constraint_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
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
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[constraint_sum])
    
    if result.success:
        optimal_weights = result.x
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        for i, country in enumerate(countries_list):
            portfolio += optimal_weights[i] * all_costs[country]
        
        return {
            'allocations': {countries_list[i]: w for i, w in enumerate(optimal_weights)},
            'expected_cost': np.mean(portfolio),
            'std_cost': np.std(portfolio),
            'portfolio_costs': portfolio
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
    base_costs = simulate_country(base_params, n_runs)
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
        mean_low = np.mean(simulate_country(params_low, n_runs))
        mean_high = np.mean(simulate_country(params_high, n_runs))
        
        results.append({
            'Factor': factor_name,
            'Low Cost': mean_low,
            'High Cost': mean_high,
            'Impact': mean_high - mean_low
        })
        
    return pd.DataFrame(results), baseline_mean

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Tesla Sourcing Monte Carlo Simulation Dashboard")
st.write("This dashboard simulates total costs per lamp for sourcing from the US, Mexico, or China, accounting for uncertainties in costs, yields, and risks.")

n_runs = st.number_input("Number of Simulation Runs", min_value=1000, max_value=100000, value=10000, step=1000)

# --- Global Simulation Section ---
if st.button("Run Global Simulation"):
    with st.spinner("Running simulations for all countries..."):
        all_costs = {country: simulate_country(params, n_runs) for country, params in countries.items()}

        st.subheader("Summary Statistics")
        cols = st.columns(len(countries))
        for i, (country, costs) in enumerate(all_costs.items()):
            with cols[i]:
                st.markdown(f"### {country}")
                st.metric(label="Expected Cost", value=f"${np.mean(costs):.2f}")
                st.metric(label="Standard Deviation", value=f"${np.std(costs):.2f}")
                st.metric(label="5th Percentile Cost", value=f"${np.percentile(costs, 5):.2f}")
                st.metric(label="95th Percentile Cost", value=f"${np.percentile(costs, 95):.2f}")

        percentiles = np.arange(1, 101)
        percentile_data = []
        for country, costs in all_costs.items():
            percentile_values = np.percentile(costs, percentiles)
            for p, v in zip(percentiles, percentile_values):
                percentile_data.append({'Country': country, 'Percentile': p, 'Cost ($)': v})
        df_percentiles = pd.DataFrame(percentile_data)

        st.subheader("Cost Distribution by Percentile")
        fig = px.line(df_percentiles, x='Percentile', y='Cost ($)', color='Country', title='Cost Distribution by Percentile', labels={'Percentile': 'Cost Percentile', 'Cost ($)': 'Total Cost ($/lamp)'}, hover_data={'Cost ($)': ':.2f'})
        fig.update_layout(legend_title_text='Country', yaxis_title="Total Cost ($/lamp)", xaxis_title="Cost Percentile (%)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Portfolio Optimization UI ---
st.header("Portfolio Optimization (Mean-Variance)")
st.write("Find the optimal sourcing allocation that minimizes: **E[Cost] + λ × SD[Cost]**")
st.write("This balances expected cost against risk (volatility). Higher λ means you care more about reducing risk.")

opt_col1, opt_col2 = st.columns([1, 2])
with opt_col1:
    st.subheader("Optimization Settings")
    
    lambda_risk = st.slider(
        "Risk Aversion (λ)", 
        min_value=0.0, 
        max_value=5.0, 
        value=1.0, 
        step=0.1,
        help="λ=0: minimize expected cost only. Higher λ: care more about reducing risk."
    )
    
    st.markdown("##### Optional Constraints")
    add_constraints = st.checkbox("Add min/max allocation constraints")
    
    constraints = {}
    if add_constraints:
        for country in countries.keys():
            col_a, col_b = st.columns(2)
            with col_a:
                min_val = st.number_input(f"{country} Min %", 0, 100, 0, 5, key=f"min_{country}")
            with col_b:
                max_val = st.number_input(f"{country} Max %", 0, 100, 100, 5, key=f"max_{country}")
            constraints[country] = (min_val/100, max_val/100)
    
    run_optimization = st.button("🎯 Run Optimization", type="primary")

if run_optimization:
    with st.spinner("Running optimization..."):
        # Run simulations
        all_costs = {country: simulate_country(params, n_runs) for country, params in countries.items()}
        
        # Optimize
        opt_constraints = constraints if add_constraints else None
        result = optimize_portfolio(all_costs, lambda_risk, opt_constraints)
        
        if result:
            with opt_col2:
                st.subheader("Optimal Portfolio")
                st.metric("Expected Cost", f"${result['expected_cost']:.2f}")
                st.metric("Standard Deviation", f"${result['std_cost']:.2f}")
                st.metric("Objective Value", f"${result['expected_cost'] + lambda_risk * result['std_cost']:.2f}")
                
                st.markdown("##### Optimal Allocations")
                alloc_df = pd.DataFrame([
                    {'Country': country, 'Allocation': f"{weight*100:.1f}%", 'Weight': weight}
                    for country, weight in result['allocations'].items()
                ]).sort_values('Weight', ascending=False)
                st.dataframe(alloc_df[['Country', 'Allocation']], hide_index=True)
                
                # Pie chart of allocations
                fig_pie = px.pie(
                    alloc_df, 
                    values='Weight', 
                    names='Country', 
                    title='Optimal Allocation Mix'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.error("Optimization failed. Try relaxing constraints or adjusting parameters.")

st.markdown("---")

# --- Sensitivity Analysis UI ---
st.header("Sensitivity Analysis")
st.write("Analyze which factors have the biggest impact on the total cost for a specific country. This chart shows how a +/- 20% change in each input variable affects the expected total cost.")

sa_col1, sa_col2 = st.columns([1, 3])
with sa_col1:
    sa_country = st.selectbox("Select Country to Analyze", list(countries.keys()), key="sa_country")
    run_sa = st.button("Run Sensitivity Analysis")

if run_sa:
    with st.spinner(f"Running sensitivity analysis for {sa_country}..."):
        base_params = countries[sa_country]
        
        # Define factors to test for each country
        factors = {
            'US': [
                ('Raw Material Mean', ('raw', 'mean')),
                ('Labor Mean', ('labor', 'mean')),
                ('Working Capital Mean', ('working_capital', 'mean')),
                ('Manufacturing Yield', ('yield_params', 'a')),
                ('Disruption Probability', ('disruption_prob',))
            ],
            'Mexico': [
                ('Raw Material Mean', ('raw', 'mean')),
                ('Labor Mean', ('labor', 'mean')),
                ('Tariff (Fixed)', ('tariff', 'fixed')),
                ('Currency Volatility', ('currency_std',)),
                ('Border Crossing Time', ('border_mean',)),
                ('Manufacturing Yield', ('yield_params', 'a')),
            ],
            'China': [
                ('Raw Material Mean', ('raw', 'mean')),
                ('Logistics Mean', ('logistics', 'mean')),
                ('Logistics Volatility', ('logistics', 'std')),
                ('Cancellation Probability', ('cancellation_prob',)),
                ('Manufacturing Yield', ('yield_params', 'a')),
                ('Tariff (Fixed)', ('tariff', 'fixed')),
            ]
        }
        
        factors_to_test = factors[sa_country]
        sa_results, baseline_mean = run_sensitivity_analysis(base_params, factors_to_test, n_runs)
        
        sa_results = sa_results.sort_values(by='Impact', ascending=True)

        # Create Tornado Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sa_results['Factor'],
            x=sa_results['High Cost'] - baseline_mean,
            name='High Estimate (Input +20%)',
            orientation='h',
            marker_color='indianred'
        ))
        fig.add_trace(go.Bar(
            y=sa_results['Factor'],
            x=sa_results['Low Cost'] - baseline_mean,
            name='Low Estimate (Input -20%)',
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Tornado Plot for {sa_country} (Baseline Cost: ${baseline_mean:.2f})',
            xaxis_title='Impact on Total Cost ($/lamp)',
            yaxis_title='Sensitivity Factor',
            barmode='relative',
            yaxis_autorange='reversed',
            legend=dict(x=0.01, y=0.01, traceorder='normal'),
            margin=dict(l=150) # Add left margin for long factor names
        )
        
        with sa_col2:
            st.plotly_chart(fig, use_container_width=True)
