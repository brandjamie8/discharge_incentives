import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# -----------------------
# Data Loading
# -----------------------
@st.cache_data
def load_data():
    """Load and prepare the CSV data."""
    df = pd.read_csv("data/test_dataset1.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    return df

# -----------------------
# Determine Default Date Range
# -----------------------
def get_default_date_range(df, freq):
    """
    Returns (start_date, end_date) for the default date range:
      - If 'daily': last 30 days
      - If 'weekly': last 26 complete weeks (Mon-Sun)
    """
    if df.empty:
        today = datetime.date.today()
        return today, today
    
    # The latest date in the dataset
    max_date = df["date"].max().date()
    
    if freq == "weekly":
        # Identify the last "complete" Monday as end date
        # (i.e., if max_date is Wed, we move back to Monday)
        last_monday = max_date - datetime.timedelta(days=max_date.weekday())
        # 26 weeks ago from that Monday
        start_date = last_monday - datetime.timedelta(weeks=25)
        end_date = last_monday - datetime.timedelta(days=1)
    else:
        # 'daily' => last 30 days
        end_date = max_date
        start_date = end_date - datetime.timedelta(days=29)
    
    # Make sure we don't go before the earliest data date
    min_date = df["date"].min().date()
    if start_date < min_date:
        start_date = min_date
    
    return start_date, end_date

# -----------------------
# Utility: Daily or Weekly Aggregation
# -----------------------
def daily_or_weekly(df, freq, agg_map):
    """
    1) First, aggregate all rows for each date into a single row,
       combining data for all selected sites on that date using `agg_map`.
    2) If freq == 'daily', return the daily-aggregated dataframe.
    3) If freq == 'weekly', resample that daily dataframe by W-MON (Monday start),
       and apply the same aggregation map.

    Parameters
    ----------
    df : pd.DataFrame
        Already filtered DataFrame (by site, date range, etc.).
    freq : str
        Either 'daily' or 'weekly'.
    agg_map : dict
        Dict of column -> aggregator, e.g.:
          {
            "admissions": "sum",
            "discharges": "sum",
            "patients LoS 7+ days": "mean",
            "patients LoS 14+ days": "mean",
            ...
          }

    Returns
    -------
    pd.DataFrame
        Aggregated to daily or weekly timescale, with one row per date/week.
    """

    if df.empty:
        return df

    # --- 1) Aggregate to one row per date (summing across sites, etc.) ---
    #     This collapses all rows for each date into a single daily row.
    daily_df = (
        df.groupby("date", as_index=False)
          .agg(agg_map)
    )

    # --- 2) If daily frequency, just return daily_df ---
    if freq == "daily":
        return daily_df

    # --- 3) If weekly frequency, resample the daily_df to Monday-based weeks ---
    elif freq == "weekly":
        # Note: Using 'label="left", closed="left"' means each week's label is the Monday
        # at the start of that week, and it collects data from Monday 00:00 up to
        # (but not including) the following Monday.
        weekly_df = (
            daily_df.set_index("date")
                    .resample("W-MON", label="left", closed="left")
                    .agg(agg_map)
                    .reset_index()
        )
        return weekly_df

    # Fallback if unexpected freq
    return daily_df


# -----------------------
# Chart Creation Function
# -----------------------
def create_chart(data, x_col, y_cols, color_map=None, labels=None):
    """
    Create a Plotly line chart for the given DataFrame.
    x_col:   string, name of the x-axis column (e.g. "date")
    y_cols:  list of strings, columns to plot on y-axis
    color_map: optional dict {col_name: color_hex} for custom colors
    labels:  dict for axis labels
    """
    if color_map:
        # Align color sequence with y_cols
        color_sequence = [color_map.get(col, "#000000") for col in y_cols]
    else:
        color_sequence = None
    
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
            xanchor="left", 
            x=0.05
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
# Sidebar Controls
# -----------------------
st.sidebar.header("Filters & Settings")
df = load_data()

# Global frequency
frequency = st.sidebar.radio("Frequency", ["daily", "weekly"], index=0)

# Default date range
start_default, end_default = get_default_date_range(df, frequency)

# Let the user pick a date range
date_input = st.sidebar.date_input(
    "Select Date Range",
    value=(start_default, end_default)
)

# Safely parse the date input (in case user selects only one date)
if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date, end_date = date_input
elif isinstance(date_input, datetime.date):
    start_date, end_date = date_input, date_input
else:
    start_date, end_date = start_default, end_default

# Filter by the selected date range
filtered_data = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# Site filter
all_sites = filtered_data["site"].unique()
selected_sites = st.sidebar.multiselect("Select Site(s)", all_sites, default=all_sites)

filtered_data = filtered_data[ filtered_data["site"].isin(selected_sites) ]

st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

tab1, tab2 = st.tabs(["Daily Sitrep Metrics", "DPTL Metrics"])

with tab1:

    # -----------------------
    # Layout: Top Row
    # -----------------------
    _, col1, col2 = st.columns([1,7,1])
    
    # Top Left: Admissions & Discharges
    with col1:
        suffix = " (weekly total)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Admissions and Discharges{suffix}")
        # Aggregate
        agg_map_adm = {
            "admissions": "sum",
            "discharges": "sum"
        }
        chart_data_1 = daily_or_weekly(filtered_data, frequency, agg_map_adm)
    
        # Create & display chart
        color_map_1 = {
            "admissions": "#1f77b4",
            "discharges": "#ff7f0e"
        }
        fig1 = create_chart(
            chart_data_1,
            x_col="date",
            y_cols=["admissions", "discharges"],
            color_map=color_map_1,
            labels={"value": "Patients", "variable": "Metric"}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"7+ LoS and 14+ LoS{suffix}")
        
        agg_map_los = {
            "patients LoS 7+ days": "mean",
            "patients LoS 14+ days": "mean"
        }
        chart_data_2 = daily_or_weekly(filtered_data, frequency, agg_map_los)
    
        color_map_2 = {
            "patients LoS 7+ days": "#e377c2",
            "patients LoS 14+ days": "#17becf"
        }
        fig2 = create_chart(
            chart_data_2,
            x_col="date",
            y_cols=["patients LoS 7+ days", "patients LoS 14+ days"],
            color_map=color_map_2,
            labels={"value": "Patients", "variable": "LoS Category"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Escalation & Boarded Beds{suffix}")
        agg_map_beds = {
            "escalation beds": "mean",
            "boarded beds": "mean"
        }
        chart_data_3 = daily_or_weekly(filtered_data, frequency, agg_map_beds)
    
        color_map_3 = {
            "escalation beds": "#2ca02c",
            "boarded beds": "#d62728"
        }
        fig3 = create_chart(
            chart_data_3,
            x_col="date",
            y_cols=["escalation beds", "boarded beds"],
            color_map=color_map_3,
            labels={"value": "Beds", "variable": "Bed Type"}
        )
        st.plotly_chart(fig3, use_container_width=True)
    

