import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processing import load_data, filter_data, calculate_metrics, compare_weeks
from utils.charts import create_summary_charts, create_comparison_chart

# Set Streamlit page configuration
st.set_page_config(page_title="Hospital Metrics Dashboard", layout="wide")

# Load datasets
dataset1, dataset2 = load_data()

# Sidebar filters
st.sidebar.header("Filters")
period = st.sidebar.date_input("Select Period", [])
sites = st.sidebar.multiselect("Select Site", dataset1['site'].unique())
pathways = st.sidebar.multiselect("Select Pathway", dataset2['pathway'].unique())
boroughs = st.sidebar.multiselect("Select Borough", dataset2['borough'].unique())

# Chart customization
st.sidebar.header("Chart Customization")
line_split_by = st.sidebar.selectbox(
    "Split Lines By", options=["site", "pathway", "borough"], index=0
)

# Apply filters
filtered_data1, filtered_data2 = filter_data(dataset1, dataset2, period, sites, pathways, boroughs)

# Navigation
pages = ["Metrics Summary", "Compare Weeks"]
selected_page = st.sidebar.radio("Navigate to", pages)

if selected_page == "Metrics Summary":
    st.title("Metrics Summary")
    metrics = calculate_metrics(filtered_data1, filtered_data2)
    st.metric("Average LoS 7+ Days", metrics['avg_los_7'])
    st.metric("Average Discharge Completion %", metrics['avg_discharge_passport'])
    st.plotly_chart(create_summary_charts(filtered_data1, filtered_data2, line_split_by), use_container_width=True)
    st.download_button("Export Data", data=filtered_data1.to_csv(index=False), file_name="filtered_data1.csv")

elif selected_page == "Compare Weeks":
    st.title("Compare Two Weeks")
    week1 = st.selectbox("Select Week 1", pd.to_datetime(filtered_data1['date']).dt.to_period('W').unique())
    week2 = st.selectbox("Select Week 2", pd.to_datetime(filtered_data1['date']).dt.to_period('W').unique())

    week_comparison = compare_weeks(filtered_data1, filtered_data2, week1, week2)
    st.dataframe(week_comparison)
    st.plotly_chart(create_comparison_chart(week_comparison), use_container_width=True)
    st.download_button("Export Comparison", data=week_comparison.to_csv(index=False), file_name="week_comparison.csv")
