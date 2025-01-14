import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
def load_data():
    return pd.read_csv("data/test_dataset1.csv")

# Streamlit App
st.set_page_config(page_title="LGT Discharge Incentives Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")
df = load_data()
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime
df = df.sort_values('date')  # Ensure data is sorted by date
sites = df['site'].unique()
selected_site = st.sidebar.multiselect("Select Site(s)", sites, default=sites)

frequency = st.sidebar.radio("Frequency", ["daily", "weekly"], index=0)  # Daily/Weekly toggle

if frequency == "daily":
    date_range = st.sidebar.date_input(
        "Select Date Range", 
        [df['date'].max() - pd.Timedelta(days=30), df['date'].max()]
    )
else:  # Default to last 26 full weeks
    last_full_week = df['date'].max() - pd.Timedelta(days=df['date'].max().weekday())
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [last_full_week - pd.Timedelta(weeks=25), last_full_week]
    )

# Filtered data
filtered_data = df[
    (df['site'].isin(selected_site)) &
    (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

def aggregate_data(data, freq, agg_type):
    if freq == "daily":
        return data
    elif freq == "weekly":
        agg_func = "sum" if agg_type == "sum" else "mean"
        data = data.resample("W-Mon", on="date").agg({
            "admissions": "sum",
            "discharges": "sum",
            "patients LoS 7+ days": "mean",
            "patients LoS 14+ days": "mean",
            "escalation beds": "mean",
            "boarded beds": "mean"
        }).reset_index()
        return data

# Title
st.title("Discharge Incentives Monitoring")

# Quadrants
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Chart Settings
def create_chart(data, x, y, colors, labels):
    fig = px.line(
        data, x=x, y=y, color_discrete_sequence=colors, labels=labels
    )
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

with col1:
    agg_type = "sum"  # Default for admissions and discharges
    chart_data = aggregate_data(filtered_data, frequency, agg_type)
    subheader = f"{'Total ' if frequency == 'weekly' else ''}Total Admissions and Discharges per {'Week' if frequency == 'weekly' else 'Day'}"
    st.subheader(subheader)
    fig1 = create_chart(
        chart_data, "date", ["admissions", "discharges"],
        ["#1f77b4", "#ff7f0e"], {"value": "Patients", "variable": "Metric"}
    )
    st.plotly_chart(fig1, use_container_width=True)

# Top Right: 7+ LoS and 14+ LoS
with col2:
    agg_type = "mean"  # Default for LoS metrics
    chart_data = aggregate_data(filtered_data, frequency, agg_type)
    subheader = f"Average 7+ LoS and 14+ LoS per {'Week' if frequency == 'weekly' else 'Day'}"
    st.subheader(subheader)
    fig2 = create_chart(
        chart_data, "date", ["patients LoS 7+ days", "patients LoS 14+ days"],
        ["#e377c2", "#17becf"], {"value": "Patients", "variable": "LoS Category"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# Bottom Right: Escalation and Boarded Beds
with col4:
    agg_type = "mean"  # Default for bed metrics
    chart_data = aggregate_data(filtered_data, frequency, agg_type)
    subheader = f"Average Escalation and Boarded Beds per {'Week' if frequency == 'weekly' else 'Day'}"
    st.subheader(subheader)
    fig3 = create_chart(
        chart_data, "date", ["escalation beds", "boarded beds"],
        ["#2ca02c", "#d62728"], {"value": "Beds", "variable": "Bed Type"}
    )
    st.plotly_chart(fig3, use_container_width=True)

# Bottom Left: Reserved for Future Use
with col3:
    st.subheader("Reserved for Future Use")
    st.write("This quadrant is currently blank.")

# Data Export
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

