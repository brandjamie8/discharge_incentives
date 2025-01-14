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



date_range = st.sidebar.date_input(
    "Select Date Range", 
    [df['date'].max() - pd.Timedelta(days=30), df['date'].max()]
)

# Filtered data
filtered_data = df[
    (df['site'].isin(selected_site)) &
    (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# Aggregation function
def aggregate_data(data, freq, agg_type):
    if freq == "daily":
        return data
    elif freq == "weekly":
        agg_func = "sum" if agg_type == "sum" else "mean"
        data = data.resample("W-Mon", on="date").agg({
            "admissions": agg_func,
            "discharges": agg_func,
            "patients LoS 7+ days": agg_func,
            "patients LoS 14+ days": agg_func,
            "escalation beds": agg_func,
            "boarded beds": agg_func
        }).reset_index()
        return data

# Title
st.title("Discharge Incentives Monitoring")

# Quadrants
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Chart Settings
def create_chart(data, x, y, title, colors, labels):
    fig = px.line(
        data, x=x, y=y, color_discrete_sequence=colors, labels=labels, title=title
    )
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# Top Left: Admissions and Discharges
with col1:
    st.subheader("Admissions and Discharges Over Time")
    with st.expander("Chart Options"):
        freq = st.radio("Frequency", ["daily", "weekly"], index=0, key="admissions_freq")
        agg_type = st.radio("Aggregation", ["sum", "average"], index=0, key="admissions_agg")
    chart_data = aggregate_data(filtered_data, freq, agg_type)
    fig1 = create_chart(
        chart_data, "date", ["admissions", "discharges"],
        "Admissions and Discharges", ["#1f77b4", "#ff7f0e"], {"value": "Patients", "variable": "Metric"}
    )
    st.plotly_chart(fig1, use_container_width=True)

# Top Right: 7+ LoS and 14+ LoS
with col2:
    st.subheader("7+ LoS and 14+ LoS Over Time")
    with st.expander("Chart Options"):
        freq = st.radio("Frequency", ["daily", "weekly"], index=0, key="los_freq")
        agg_type = st.radio("Aggregation", ["sum", "average"], index=0, key="los_agg")
    chart_data = aggregate_data(filtered_data, freq, agg_type)
    fig2 = create_chart(
        chart_data, "date", ["patients LoS 7+ days", "patients LoS 14+ days"],
        "Length of Stay Metrics", ["#e377c2", "#17becf"], {"value": "Patients", "variable": "LoS Category"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# Bottom Right: Escalation and Boarded Beds
with col4:
    st.subheader("Escalation and Boarded Beds Over Time")
    with st.expander("Chart Options"):
        freq = st.radio("Frequency", ["daily", "weekly"], index=0, key="beds_freq")
        agg_type = st.radio("Aggregation", ["sum", "average"], index=0, key="beds_agg")
    chart_data = aggregate_data(filtered_data, freq, agg_type)
    fig3 = create_chart(
        chart_data, "date", ["escalation beds", "boarded beds"],
        "Escalation and Boarded Beds", ["#2ca02c", "#d62728"], {"value": "Beds", "variable": "Bed Type"}
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

