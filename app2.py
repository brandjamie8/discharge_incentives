import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    # Replace with your data loading logic
    return pd.read_csv(r"data/test_dataset1.csv")

# Streamlit App
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")
df = load_data()
sites = df['site'].unique()
selected_site = st.sidebar.multiselect("Select Site(s)", sites, default=sites)
date_range = st.sidebar.date_input("Select Date Range", 
                                   [df['date'].min(), df['date'].max()])

# Filtered Data
filtered_data = df[(df['site'].isin(selected_site)) & 
                   (df['date'].between(*date_range))]

# Main Title
st.title("Hospital Performance Dashboard")

# Metrics Section
st.subheader("Key Performance Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Admissions", filtered_data['admissions'].sum())
with col2:
    st.metric("Total Discharges", filtered_data['discharges'].sum())
with col3:
    st.metric("Average Boarded Beds", filtered_data['boarded beds'].mean())

# Charts Section
st.subheader("Trend Analysis")

# Admissions and Discharges Over Time
fig1 = px.line(filtered_data, x='date', y=['admissions', 'discharges'], 
               color_discrete_map={'admissions': 'blue', 'discharges': 'green'},
               labels={"value": "Patients", "variable": "Metric"})
fig1.update_layout(title="Admissions and Discharges Over Time", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# Escalation vs. Boarded Beds
fig2 = px.bar(filtered_data, x='date', y=['escalation beds', 'boarded beds'], 
              barmode='group', 
              labels={"value": "Beds", "variable": "Bed Type"})
fig2.update_layout(title="Escalation and Boarded Beds", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# LoS Analysis
fig3 = px.bar(filtered_data, x='site', y=['patients LoS 7+ days', 'patients LoS 14+ days'], 
              barmode='group', 
              labels={"value": "Patients", "variable": "LoS Category"})
fig3.update_layout(title="Length of Stay Analysis", template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# Analysis Section
st.subheader("Analysis")
st.write("The data indicates trends in hospital capacity, patient flow, and bed usage. Use this section to write interpretations of the metrics.")

# Data Export
st.download_button("Download Filtered Data", 
                   data=filtered_data.to_csv(index=False), 
                   file_name="filtered_data.csv", 
                   mime="text/csv")
