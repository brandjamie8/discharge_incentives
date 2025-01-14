import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# -----------------------
# Data Loading
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/test_dataset1.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    return df

# -----------------------
# Utility function to handle default date ranges
# -----------------------
def get_default_date_range(df, freq):
    """
    Returns a tuple (start_date, end_date) for the default date range
    based on the global frequency setting.
    - 'daily': last 30 days
    - 'weekly': last 26 complete weeks
    """
    if df.empty:
        # If no data, just return today's date
        return (datetime.date.today(), datetime.date.today())
    
    # The latest date in the dataset
    max_date = df["date"].max().date()
    
    if freq == "weekly":
        # Identify the last "complete" Monday for the end
        # For instance, if max_date is Wed, we go back to previous Monday
        # so that it's a "complete" week up to that Monday
        last_monday = max_date - datetime.timedelta(days=max_date.weekday())
        # 26 weeks ago from that Monday
        start_date = last_monday - pd.Timedelta(weeks=25)
        end_date = last_monday
    else:
        # daily default: last 30 days
        end_date = max_date
        start_date = end_date - datetime.timedelta(days=29)
    
    # Ensure no earlier than min_date
    min_date = df["date"].min().date()
    if start_date < min_date:
        start_date = min_date
    
    return (start_date, end_date)

# -----------------------
# Utility Function to Aggregate Data
# -----------------------
def aggregate_data(df, freq, agg_type, metrics):
    """
    Given a DataFrame, frequency ('daily' or 'weekly'), 
    and aggregation type ('sum' or 'mean'), 
    along with list of metrics, return aggregated data.
    """
    if df.empty or not metrics:
        return df
    
    if freq == "daily":
        # Just return the original (already filtered) subset
        return df
    
    elif freq == "weekly":
        # For each metric in metrics, map it to the chosen agg_type
        agg_map = {}
        for metric in metrics:
            agg_map[metric] = agg_type
        # Resample with W-Mon
        weekly_df = df.set_index("date").resample("W-Mon").agg(agg_map)
        weekly_df.reset_index(inplace=True)
        return weekly_df
    
    else:
        return df

# -----------------------
# Chart Creation Function
# -----------------------
def create_chart(data, x_col, y_cols, color_map=None, labels=None):
    """
    Create a Plotly line chart for the given DataFrame.
    x_col: string, name of the x-axis column (e.g. "date")
    y_cols: list of strings, columns to plot on y-axis
    color_map: optional dict for custom colors {col_name: color}
    labels: dict for axis labels
    """
    if color_map:
        # Build color sequence in the order of y_cols
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

st.title("Discharge Incentives Monitoring")

# -----------------------
# Sidebar (Global Controls)
# -----------------------
st.sidebar.header("Global Filters & Settings")
df = load_data()

# Global frequency
global_freq = st.sidebar.radio("Global Frequency", ["daily", "weekly"], index=0)

# Get default date range for the chosen global frequency
start_default, end_default = get_default_date_range(df, global_freq)

# Provide a date_input with a default tuple
date_input = st.sidebar.date_input(
    "Select Date Range",
    value=(start_default, end_default)
)

# Handle different ways the user can interact with date_input
if isinstance(date_input, tuple):
    if len(date_input) == 2:
        start_date, end_date = date_input
    else:
        # If user only picks one date for some reason
        start_date = date_input[0]
        end_date = date_input[0]
elif isinstance(date_input, datetime.date):
    start_date, end_date = date_input, date_input
else:
    # Fallback
    start_date, end_date = start_default, end_default

# Now filter data
filtered_data = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# Site filter
sites = filtered_data["site"].unique()
selected_site = st.sidebar.multiselect("Select Site(s)", sites, default=sites)

filtered_data = filtered_data[filtered_data["site"].isin(selected_site)]

# -----------------------
# Quadrant Layout
# -----------------------
col1, col2 = st.columns(2)  # top row
col3, col4 = st.columns(2)  # bottom row

# -----------------------
# CHART 1 (Admissions & Discharges)
# -----------------------
with col1:
    # Subheader first
    st.subheader("Admissions & Discharges")
    
    # Then the expander
    with st.expander("Chart 1 Settings (override global)"):
        # Let user choose to override frequency or use global
        freq_override_1 = st.radio(
            "Frequency (Chart 1)",
            ["Use Global Frequency", "daily", "weekly"],
            index=0,
            key="chart1_freq"
        )
        # Determine the actual frequency to use for Chart 1
        if freq_override_1 == "Use Global Frequency":
            freq_option_1 = global_freq
        else:
            freq_option_1 = freq_override_1
        
        agg_option_1 = st.radio("Aggregation Type", ["sum", "mean"], index=0, key="chart1_agg")
        
        # Available metrics for this chart
        available_metrics_1 = ["admissions", "discharges"]
        selected_metrics_1 = st.multiselect(
            "Select Metrics to Display",
            available_metrics_1, 
            default=available_metrics_1
        )
    
    # Aggregate data for Chart 1
    chart_data_1 = aggregate_data(filtered_data, freq_option_1, agg_option_1, selected_metrics_1)
    
    # Build the subheader string based on actual freq used
    if freq_option_1 == "weekly":
        subheader_1 = f"Admissions & Discharges per Week (Aggregated by {agg_option_1})"
    else:
        subheader_1 = f"Admissions & Discharges per Day (Aggregated by {agg_option_1})"
    
    # Show dynamic subheader under the user-facing subheader text
    st.caption(subheader_1)
    
    # Create chart
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
    # Subheader first
    st.subheader("Length of Stay (7+ & 14+ Days)")
    
    with st.expander("Chart 2 Settings (override global)"):
        freq_override_2 = st.radio(
            "Frequency (Chart 2)",
            ["Use Global Frequency", "daily", "weekly"],
            index=0,
            key="chart2_freq"
        )
        if freq_override_2 == "Use Global Frequency":
            freq_option_2 = global_freq
        else:
            freq_option_2 = freq_override_2
        
        agg_option_2 = st.radio("Aggregation Type", ["sum", "mean"], index=1, key="chart2_agg")
        
        available_metrics_2 = ["patients LoS 7+ days", "patients LoS 14+ days"]
        selected_metrics_2 = st.multiselect(
            "Select Metrics to Display",
            available_metrics_2,
            default=available_metrics_2
        )
    
    chart_data_2 = aggregate_data(filtered_data, freq_option_2, agg_option_2, selected_metrics_2)
    
    if freq_option_2 == "weekly":
        subheader_2 = f"7+ LoS and 14+ LoS per Week (Aggregated by {agg_option_2})"
    else:
        subheader_2 = f"7+ LoS and 14+ LoS per Day (Aggregated by {agg_option_2})"
    
    st.caption(subheader_2)
    
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
    # Subheader
    st.subheader("Escalation & Boarded Beds")
    
    with st.expander("Chart 3 Settings (override global)"):
        freq_override_3 = st.radio(
            "Frequency (Chart 3)",
            ["Use Global Frequency", "daily", "weekly"],
            index=0,
            key="chart3_freq"
        )
        if freq_override_3 == "Use Global Frequency":
            freq_option_3 = global_freq
        else:
            freq_option_3 = freq_override_3
        
        agg_option_3 = st.radio("Aggregation Type", ["sum", "mean"], index=1, key="chart3_agg")
        
        available_metrics_3 = ["escalation beds", "boarded beds"]
        selected_metrics_3 = st.multiselect(
            "Select Metrics to Display",
            available_metrics_3,
            default=available_metrics_3
        )
    
    chart_data_3 = aggregate_data(filtered_data, freq_option_3, agg_option_3, selected_metrics_3)

    if freq_option_3 == "weekly":
        subheader_3 = f"Escalation & Boarded Beds per Week (Aggregated by {agg_option_3})"
    else:
        subheader_3 = f"Escalation & Boarded Beds per Day (Aggregated by {agg_option_3})"

    st.caption(subheader_3)

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
