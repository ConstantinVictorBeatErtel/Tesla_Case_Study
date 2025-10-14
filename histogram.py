 # === TAB 1: Histogram/KDE overlays ===
def histogram_tab(tab, all_costs):
    import streamlit as st
    import plotly.graph_objects as go
    import numpy as np
        # Histogram overlay
    with tab:
        fig_h = go.Figure()
        for country, costs in all_costs.items():
            fig_h.add_trace(go.Histogram(x=costs, name=country, opacity=0.55, nbinsx=60))
        fig_h.update_layout(
            barmode='overlay',
            title="Overlaid Histograms of Total Cost",
            xaxis_title="Total Cost ($/lamp)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_h, use_container_width=True)