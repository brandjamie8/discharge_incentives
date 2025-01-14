import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
def load_data():
    return pd.read_csv("data/test_dataset1.csv")

# Streamlit App
st.set_page_config(page_title="Hospital Performance Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")
df = load_data()
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime
df = df.sort_values('date')  # Ensure data is sorted by date
sites = df['site'].unique()
selected_site = st.sidebar.multiselect("Select Site(s)", sites, default=sites)
date_range = st.sidebar.date_input(
    "Select Date Range", 
    [df['date'].min().to_pydatetime().date(), df['date'].max().to_pydatetime().date()]
)

# Filtered Data
filtered_data = df[
    (df['site'].isin(selected_site)) & 
    (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# Main Title
st.title("Hospital Performance Dashboard")

# Quadrant Layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Top Left: Admissions and Discharges Over Time
with col1:
    st.subheader("Admissions and Discharges Over Time")
    fig1 = px.line(
        filtered_data, x='date', y=['admissions', 'discharges'], 
        labels={"value": "Patients", "variable": "Metric"},
    )
    fig1.update_layout(template="plotly_white", legend_title="Metric")
    fig1.update_layout(
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Align vertically at the bottom
            y=1.02,            # Position above the chart
            xanchor="center",  # Center alignment
            x=0.5              # Center horizontally
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

# Top Right: 7+ LoS and 14+ LoS Over Time
with col2:
    st.subheader("7+ LoS and 14+ LoS Over Time")
    fig2 = px.line(
        filtered_data, x='date', y=['patients LoS 7+ days', 'patients LoS 14+ days'],
        labels={"value": "Patients", "variable": "Length of Stay"},
    )
    fig2.update_layout(template="plotly_white", legend_title="Length of Stay")
    fig2.update_layout(
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Align vertically at the bottom
            y=1.02,            # Position above the chart
            xanchor="center",  # Center alignment
            x=0.5              # Center horizontally
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

# Bottom Right: Escalation and Boarded Beds Over Time
with col4:
    st.subheader("Escalation and Boarded Beds Over Time")
    fig3 = px.line(
        filtered_data, x='date', y=['escalation beds', 'boarded beds'],
        labels={"value": "Beds", "variable": "Bed Type"},
    )
    fig3.update_layout(template="plotly_white", legend_title="Bed Type")
    fig3.update_layout(
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Align vertically at the bottom
            y=1.02,            # Position above the chart
            xanchor="center",  # Center alignment
            x=0.5              # Center horizontally
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

# Bottom Left: Reserved for Future Use
with col3:
    st.subheader("Reserved for Future Use")
    st.write("This quadrant is currently blank.")

# Data Export Button
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)
