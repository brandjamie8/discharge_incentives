import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------
# Data Loading
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/test_dataset1.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# -----------------------
# Utility Function to Aggregate Data
# -----------------------
def aggregate_data(df, freq, agg_type, metrics):
    """
    Given a DataFrame, frequency ('daily' or 'weekly'), 
    and aggregation type ('sum' or 'mean'), 
    along with list of metrics, return aggregated data.
    """
    # If daily, just return the original (already filtered/sorted) subset
    if freq == "daily":
        return df

    # If weekly, resample based on W-Mon, then apply sum or mean
    elif freq == "weekly":
        agg_map = {}
        # For each metric in metrics, map it to the chosen agg_type
        for metric in metrics:
            agg_map[metric] = agg_type

        # But we need "date" as a time index to resample
        # Let's set the date as the index temporarily
        weekly_df = df.set_index("date").resample("W-Mon").agg(agg_map)
        weekly_df.reset_index(inplace=True)
        return weekly_df

    # Fallback
    return df

# -----------------------
# Chart Creation Function
# -----------------------
def create_chart(data, x_col, y_cols, color_map=None, labels=None):
    """
    Create a Plotly line chart for the given DataFrame.
    x_col: string, name of the x-axis column (e.g. "date")
    y_cols: list of strings, columns to plot on y-axis
    color_map: optional color mapping for each y column
    labels: dict for axis labels
    """
    # If you want custom colors, you can build a discrete color sequence
    # from color_map. Otherwise, Plotly picks automatically.
    if color_map:
        # Make sure the order in color_discrete_sequence aligns with y_cols
        color_sequence = [color_map.get(col, "#000000") for col in y_cols]
    else:
        color_sequence = None  # Use default

    fig = px.line(
        data, 
        x=x_col, 
        y=y_cols, 
        color_discrete_sequence=color_sequence,
        labels=labels
    )
    fig.update_layout(
        template="plotly_white",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(page_title="LGT Discharge Incentives Dashboard", layout="wide")

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Global Filters")
df = load_data()
df = df.sort_values("date")

sites = df["site"].unique()
selected_site = st.sidebar.multiselect("Select Site(s)", sites, default=sites)

# Date Range: default to last 30 days
date_range = st.sidebar.date_input(
    "Select Date Range", 
    [df["date"].max() - pd.Timedelta(days=30), df["date"].max()]
)

# Filter the data using the global filters
filtered_data = df[
    (df["site"].isin(selected_site)) &
    (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# -----------------------
# Main Title
# -----------------------
st.title("Discharge Incentives Monitoring")

# -----------------------
# Quadrant Layout
# -----------------------
col1, col2 = st.columns(2)  # top row
col3, col4 = st.columns(2)  # bottom row

# -----------------------
# CHART 1 (Admissions & Discharges)
# -----------------------
with col1:
    # Expandable chart settings
    with st.expander("Chart 1 Settings (Admissions & Discharges)"):
        freq_option_1 = st.radio("Frequency", ["daily", "weekly"], key="chart1_freq")
        agg_option_1 = st.radio("Aggregation Type", ["sum", "mean"], key="chart1_agg")
        
        # Available metrics for this chart
        available_metrics_1 = ["admissions", "discharges"]
        selected_metrics_1 = st.multiselect(
            "Select Metrics to Display",
            available_metrics_1, 
            default=available_metrics_1
        )

    # Aggregate data based on user selections
    chart_data_1 = aggregate_data(filtered_data, freq_option_1, agg_option_1, selected_metrics_1)

    # Build the chart
    if freq_option_1 == "weekly":
        subheader_1 = f"Admissions & Discharges per Week (Aggregated by {agg_option_1})"
    else:
        subheader_1 = f"Admissions & Discharges per Day (Aggregated by {agg_option_1})"

    st.subheader(subheader_1)
    
    # For a custom color mapping (optional)
    color_map_1 = {
        "admissions": "#1f77b4",
        "discharges": "#ff7f0e"
    }

    fig1 = create_chart(
        chart_data_1,
        x_col="date",
        y_cols=selected_metrics_1,
        color_map=color_map_1,
        labels={"value": "Patients", "variable": "Metric"}
    )
    st.plotly_chart(fig1, use_container_width=True)

# -----------------------
# CHART 2 (7+ LoS and 14+ LoS)
# -----------------------
with col2:
    with st.expander("Chart 2 Settings (LoS Metrics)"):
        freq_option_2 = st.radio("Frequency", ["daily", "weekly"], key="chart2_freq")
        agg_option_2 = st.radio("Aggregation Type", ["sum", "mean"], key="chart2_agg")
        
        # Available metrics for this chart
        available_metrics_2 = ["patients LoS 7+ days", "patients LoS 14+ days"]
        selected_metrics_2 = st.multiselect(
            "Select Metrics to Display",
            available_metrics_2,
            default=available_metrics_2
        )

    # Aggregate data based on user selections
    chart_data_2 = aggregate_data(filtered_data, freq_option_2, agg_option_2, selected_metrics_2)

    if freq_option_2 == "weekly":
        subheader_2 = f"7+ LoS and 14+ LoS per Week (Aggregated by {agg_option_2})"
    else:
        subheader_2 = f"7+ LoS and 14+ LoS per Day (Aggregated by {agg_option_2})"

    st.subheader(subheader_2)

    color_map_2 = {
        "patients LoS 7+ days": "#e377c2",
        "patients LoS 14+ days": "#17becf",
    }
    fig2 = create_chart(
        chart_data_2,
        x_col="date",
        y_cols=selected_metrics_2,
        color_map=color_map_2,
        labels={"value": "Patients", "variable": "LoS Category"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# CHART 3 (Reserved or Another Metric)
# -----------------------
with col3:
    st.subheader("Reserved for Future Use")
    st.write("This quadrant is currently blank or can be used for another chart.")

# -----------------------
# CHART 4 (Escalation & Boarded Beds)
# -----------------------
with col4:
    with st.expander("Chart 3 Settings (Beds)"):
        freq_option_3 = st.radio("Frequency", ["daily", "weekly"], key="chart3_freq")
        agg_option_3 = st.radio("Aggregation Type", ["sum", "mean"], key="chart3_agg")
        
        # Available metrics for this chart
        available_metrics_3 = ["escalation beds", "boarded beds"]
        selected_metrics_3 = st.multiselect(
            "Select Metrics to Display",
            available_metrics_3,
            default=available_metrics_3
        )

    # Aggregate data based on user selections
    chart_data_3 = aggregate_data(filtered_data, freq_option_3, agg_option_3, selected_metrics_3)

    if freq_option_3 == "weekly":
        subheader_3 = f"Escalation & Boarded Beds per Week (Aggregated by {agg_option_3})"
    else:
        subheader_3 = f"Escalation & Boarded Beds per Day (Aggregated by {agg_option_3})"

    st.subheader(subheader_3)

    color_map_3 = {
        "escalation beds": "#2ca02c",
        "boarded beds": "#d62728",
    }
    fig3 = create_chart(
        chart_data_3,
        x_col="date",
        y_cols=selected_metrics_3,
        color_map=color_map_3,
        labels={"value": "Beds", "variable": "Bed Type"}
    )
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# Data Export
# -----------------------
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)
