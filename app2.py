import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

# -----------------------
# Data Loading
# -----------------------
@st.cache_data
def load_data():
    """Load and prepare the CSV data."""
    df = pd.read_csv("data/SitrepData.csv")
    # Convert date column
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
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
    1) Aggregate all rows for each date into a single row.
    2) If freq == 'daily', return the daily-aggregated dataframe.
    3) If freq == 'weekly', resample the daily dataframe by W-MON, etc.
    """
    if df.empty:
        return df

    # 1) Aggregate to one row per date
    daily_df = (
        df.groupby("date", as_index=False)
          .agg(agg_map)
    )

    # 2) Daily
    if freq == "daily":
        return daily_df

    # 3) Weekly
    elif freq == "weekly":
        weekly_df = (
            daily_df.set_index("date")
                    .resample("W-MON", label="left", closed="left")
                    .agg(agg_map)
                    .reset_index()
        )
        return weekly_df
    
    return daily_df  # fallback


# -----------------------
# Chart Creation Functions
# -----------------------
def create_line_chart(data, x_col, y_cols, color_map=None, labels=None, show_trendline=False):
    """
    Create a multi-line chart using px.line(), optionally adding
    a dotted OLS best-fit line for each metric in the same color.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with one row per date (aggregated).
    x_col : str
        The name of the x-axis column (e.g., "date").
    y_cols : list
        List of columns to be plotted on the y-axis.
    color_map : dict
        Optional dict mapping each y_col to a color, e.g. {"admissions": "#1f77b4"}.
    labels : dict
        Custom axis labels, e.g. {"value": "Patients", "variable": "Metric"}.
    show_trendline : bool
        If True, compute & add a dotted best-fit line for each y_col.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    # Melt data for px.line usage
    df_melted = data.melt(
        id_vars=[x_col],
        value_vars=y_cols,
        var_name="variable",
        value_name="value"
    )

    # Build a color mapping for each variable
    if color_map:
        color_discrete_map = {var: color_map.get(var, "#000000") for var in y_cols}
    else:
        # fallback: Let px pick
        color_discrete_map = {}

    # 1) Base line chart
    fig = px.line(
        df_melted,
        x=x_col,
        y="value",
        color="variable",
        color_discrete_map=color_discrete_map,
        labels=labels
    )
    
    # Thicker lines
    fig.update_traces(line=dict(width=3))

    # 2) If show_trendline, add a separate dotted line for each variable
    if show_trendline:
        for var in y_cols:
            df_var = df_melted[df_melted["variable"] == var].dropna(subset=["value"])
            if df_var.empty:
                continue
            
            # Convert date to ordinal for polyfit
            # (because np.polyfit needs numeric x)
            x_ord = df_var[x_col].apply(lambda d: d.toordinal()).values
            y_vals = df_var["value"].values
            
            if len(x_ord) < 2:
                # Not enough points for regression
                continue

            # Fit a linear model y = m*x + c
            slope, intercept = np.polyfit(x_ord, y_vals, 1)

            # Generate a "smooth" line of best fit
            x_min, x_max = x_ord.min(), x_ord.max()
            x_fit = np.linspace(x_min, x_max, 50)
            y_fit = slope * x_fit + intercept

            # Convert back to datetime
            dt_fit = [datetime.date.fromordinal(int(xi)) for xi in x_fit]

            # Add as a new trace
            fig.add_trace(
                go.Scatter(
                    x=dt_fit,
                    y=y_fit,
                    mode="lines",
                    name=f"{var} Trend",
                    line=dict(
                        dash="dot",
                        width=2,
                        color=color_discrete_map.get(var, "#000000")
                    ),
                    showlegend=False  # or True if you want a separate legend entry
                )
            )

    # Larger text & consistent layout
    fig.update_layout(
        template="plotly_white",
        font=dict(size=14),
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


def create_stacked_bar_chart(data, x_col, y_cols, color_map=None, labels=None):
    """
    Creates a stacked bar chart with larger text.
    """
    df_melted = data.melt(
        id_vars=[x_col],
        value_vars=y_cols,
        var_name="variable",
        value_name="value"
    )

    if color_map:
        color_discrete_map = {var: color_map.get(var, "#000000") for var in y_cols}
    else:
        color_discrete_map = {}

    fig = px.bar(
        df_melted,
        x=x_col,
        y="value",
        color="variable",
        barmode="stack",
        color_discrete_map=color_discrete_map,
        labels=labels
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(size=14),
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
# Helper: Return latest & previous values
# -----------------------
def get_latest_and_previous_values(series):
    """
    Returns (latest_value, previous_value).
    If there's only one value, previous_value = 0 for delta calc.
    If empty, returns (0, 0).
    """
    if len(series) == 0:
        return 0, 0
    elif len(series) == 1:
        return series.iloc[-1], 0
    else:
        return series.iloc[-1], series.iloc[-2]


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

# Safely parse the date input
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

# Download button
st.sidebar.download_button(
    label="Download Data",
    data=filtered_data.to_csv(index=False),
    file_name="discharge_data.csv",
    mime="text/csv"
)

# Two main tabs
tab1, tab2 = st.tabs(["Daily Sitrep Metrics", "DPTL Metrics"])

with tab1:
    _, col1, _ = st.columns([1,10,1])
    
    with col1:
        # ---------------------------------------
        # 1) Admissions & Discharges
        # ---------------------------------------
        suffix = " (weekly total)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Admissions and Discharges{suffix}")
        
        agg_map_adm = {"admissions": "sum", "discharges": "sum"}
        chart_data_1 = daily_or_weekly(filtered_data, frequency, agg_map_adm)

        # Metric cards
        latest_adm, prev_adm = get_latest_and_previous_values(chart_data_1["admissions"])
        latest_dis, prev_dis = get_latest_and_previous_values(chart_data_1["discharges"])

        card_col1, card_col2 = st.columns(2)
        with card_col1:
            st.metric(
                label="Admissions (Latest vs Previous)",
                value=int(latest_adm),
                delta=int(latest_adm - prev_adm),
            )
        with card_col2:
            st.metric(
                label="Discharges (Latest vs Previous)",
                value=int(latest_dis),
                delta=int(latest_dis - prev_dis),
            )

        # Toggle trendline
        show_trend_1 = st.radio(
            "Show Trendline (Admissions & Discharges)?",
            ["No", "Yes"],
            index=0
        )
        color_map_1 = {
            "admissions": "#1f77b4",
            "discharges": "#ff7f0e"
        }
        fig1 = create_line_chart(
            data=chart_data_1,
            x_col="date",
            y_cols=["admissions", "discharges"],
            color_map=color_map_1,
            labels={"value": "Patients", "variable": "Metric"},
            show_trendline=(show_trend_1 == "Yes")
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # ---------------------------------------
        # 2) 21+ LoS and 14+ LoS
        # ---------------------------------------
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"21+ LoS and 14+ LoS{suffix}")
        
        agg_map_los = {
            "patients LoS 21+ days": "mean",
            "patients LoS 14+ days": "mean"
        }
        chart_data_2 = daily_or_weekly(filtered_data, frequency, agg_map_los)

        # Metrics
        latest_21plus, prev_21plus = get_latest_and_previous_values(chart_data_2["patients LoS 21+ days"])
        latest_14plus, prev_14plus = get_latest_and_previous_values(chart_data_2["patients LoS 14+ days"])

        card_col3, card_col4 = st.columns(2)
        with card_col3:
            st.metric(
                label="21+ LoS (Latest vs Previous)",
                value=round(latest_21plus, 1),
                delta=round(latest_21plus - prev_21plus, 1),
            )
        with card_col4:
            st.metric(
                label="14+ LoS (Latest vs Previous)",
                value=round(latest_14plus, 1),
                delta=round(latest_14plus - prev_14plus, 1),
            )

        show_trend_2 = st.radio(
            "Show Trendline (21+ LoS & 14+ LoS)?",
            ["No", "Yes"],
            index=0
        )
        color_map_2 = {
            "patients LoS 21+ days": "#e377c2",
            "patients LoS 14+ days": "#17becf"
        }
        fig2 = create_line_chart(
            data=chart_data_2,
            x_col="date",
            y_cols=["patients LoS 21+ days", "patients LoS 14+ days"],
            color_map=color_map_2,
            labels={"value": "Patients", "variable": "LoS Category"},
            show_trendline=(show_trend_2 == "Yes")
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # ---------------------------------------
        # 3) Escalation & Boarded Beds (STACKED BAR)
        # ---------------------------------------
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Escalation & Boarded Beds{suffix}")

        agg_map_beds = {
            "escalation beds": "mean",
            "boarded beds": "mean"
        }
        chart_data_3 = daily_or_weekly(filtered_data, frequency, agg_map_beds)

        # Metric cards
        latest_esc, prev_esc = get_latest_and_previous_values(chart_data_3["escalation beds"])
        latest_boarded, prev_boarded = get_latest_and_previous_values(chart_data_3["boarded beds"])

        card_col5, card_col6 = st.columns(2)
        with card_col5:
            st.metric(
                label="Escalation Beds (Latest vs Previous)",
                value=round(latest_esc, 1),
                delta=round(latest_esc - prev_esc, 1),
            )
        with card_col6:
            st.metric(
                label="Boarded Beds (Latest vs Previous)",
                value=round(latest_boarded, 1),
                delta=round(latest_boarded - prev_boarded, 1),
            )
        
        color_map_3 = {
            "escalation beds": "#2ca02c",
            "boarded beds": "#d62728"
        }
        fig3 = create_stacked_bar_chart(
            data=chart_data_3,
            x_col="date",
            y_cols=["escalation beds", "boarded beds"],
            color_map=color_map_3,
            labels={"value": "Beds", "variable": "Bed Type"}
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.write("DPTL Metrics content goes here.")
