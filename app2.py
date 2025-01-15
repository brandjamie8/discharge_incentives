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
    
    max_date = df["date"].max().date()
    
    if freq == "weekly":
        last_monday = max_date - datetime.timedelta(days=max_date.weekday())
        start_date = last_monday - datetime.timedelta(weeks=25)
        end_date = last_monday - datetime.timedelta(days=1)
    else:
        end_date = max_date
        start_date = end_date - datetime.timedelta(days=29)
    
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
    2) If freq == 'daily', return daily_df.
    3) If freq == 'weekly', resample the daily_df by W-MON (Monday-based).
    """
    if df.empty:
        return df

    # 1) Aggregate to one row per date
    daily_df = (
        df.groupby("date", as_index=False)
          .agg(agg_map)
    )

    if freq == "daily":
        return daily_df

    elif freq == "weekly":
        weekly_df = (
            daily_df.set_index("date")
                    .resample("W-MON", label="left", closed="left")
                    .agg(agg_map)
                    .reset_index()
        )
        return weekly_df
    
    return daily_df

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
# Chart Creation Functions
# -----------------------
def create_line_chart(
    data,
    x_col,
    y_cols,
    color_map=None,
    labels=None,
    show_trendline=False,
    show_baseline=False,
    baseline_means=None,
    trendline_label="Trendline"
):
    """
    Creates a multi-line chart using px.line(), optionally adding:
      - A dotted OLS best-fit line for each y_col (if show_trendline=True).
      - Horizontal dashed lines for each y_col if show_baseline=True, plus
        a single annotation box at the top-right listing all baseline values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with one row per date (already aggregated).
    x_col : str
        The name of the x-axis column (e.g. "date").
    y_cols : list
        List of columns to be plotted on the y-axis.
    color_map : dict
        Maps each y_col to a color string (e.g. {"admissions": "#1f77b4"}).
    labels : dict
        e.g. {"value": "Patients", "variable": "Metric"}.
    show_trendline : bool
        If True, compute & add dotted best-fit lines for each y_col.
    show_baseline : bool
        If True, add horizontal lines at baseline means (if baseline_means not None).
    baseline_means : dict or None
        For each y_col, the baseline average. e.g. {"admissions": 100.5}.
    trendline_label : str
        Legend text for the trend lines.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Melt for px.line usage
    df_melted = data.melt(
        id_vars=[x_col],
        value_vars=y_cols,
        var_name="variable",
        value_name="value"
    )

    # Color mapping
    if color_map:
        color_discrete_map = {var: color_map.get(var, "#000000") for var in y_cols}
    else:
        color_discrete_map = {}

    # Base line chart
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

    # (1) Trend Lines
    if show_trendline:
        for var in y_cols:
            df_var = df_melted[df_melted["variable"] == var].dropna(subset=["value"])
            if df_var.empty:
                continue
            x_ord = df_var[x_col].apply(lambda d: d.toordinal()).values
            y_vals = df_var["value"].values

            if len(x_ord) < 2:
                continue

            slope, intercept = np.polyfit(x_ord, y_vals, 1)
            x_min, x_max = x_ord.min(), x_ord.max()
            x_fit = np.linspace(x_min, x_max, 50)
            y_fit = slope * x_fit + intercept
            dt_fit = [datetime.date.fromordinal(int(xi)) for xi in x_fit]

            fig.add_trace(
                go.Scatter(
                    x=dt_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(
                        dash="dot",
                        width=2,
                        color=color_discrete_map.get(var, "#000000")
                    ),
                    name=f"{var} {trendline_label}"
                )
            )

    # (2) Baseline
    baseline_texts = []
    if show_baseline and baseline_means is not None:
        # Add a dashed horizontal line for each y_col
        for var in y_cols:
            base_val = baseline_means.get(var, None) if baseline_means else None
            if base_val is None:
                continue

            # Add hline
            fig.add_hline(
                y=base_val,
                line=dict(
                    color=color_discrete_map.get(var, "#000000"),
                    dash="dash",
                    width=2
                ),
                # We'll put the annotation in a single box at top-right instead
                annotation=None
            )
            # We'll collect a label for this baseline
            baseline_texts.append(f"{var}: {round(base_val, 2)}")

    # If we have baseline lines, put a single annotation in top-right
    if baseline_texts:
        annotation_text = "Baseline Means:<br>" + "<br>".join(baseline_texts)
        fig.add_annotation(
            x=1, y=1,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            text=annotation_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=5,
            bgcolor="white"
        )

    # Layout
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
    Creates a stacked bar chart (no trendline or baseline lines).
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

# ---------------
# Streamlit Layout
# ---------------
st.set_page_config(page_title="LGT Discharge Incentives Dashboard", layout="wide")

st.title("Discharge Incentives Monitoring")

# Sidebar
st.sidebar.header("Filters & Settings")
df = load_data()

# Frequency
frequency = st.sidebar.radio("Frequency", ["daily", "weekly"], index=0)

# Default date range
start_default, end_default = get_default_date_range(df, frequency)
date_input = st.sidebar.date_input("Select Date Range", value=(start_default, end_default))

if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date, end_date = date_input
elif isinstance(date_input, datetime.date):
    start_date, end_date = date_input, date_input
else:
    start_date, end_date = start_default, end_default

filtered_data = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# Site filter
all_sites = filtered_data["site"].unique()
selected_sites = st.sidebar.multiselect("Select Site(s)", all_sites, default=all_sites)
filtered_data = filtered_data[filtered_data["site"].isin(selected_sites)]

# Baseline option
st.sidebar.subheader("Baseline Period")
use_baseline = st.sidebar.checkbox("Define a baseline period?", value=False)

# We'll store baseline means for each chart
baseline_means_1 = None
baseline_means_2 = None
baseline_means_3 = None

if use_baseline:
    baseline_range = st.sidebar.date_input(
        "Select baseline date range (inclusive):",
        value=(start_default, end_default)
    )
    if isinstance(baseline_range, tuple) and len(baseline_range) == 2:
        baseline_start, baseline_end = baseline_range
    elif isinstance(baseline_range, datetime.date):
        baseline_start, baseline_end = baseline_range, baseline_range
    else:
        baseline_start, baseline_end = start_default, end_default

    baseline_data = df[
        (df["date"] >= pd.to_datetime(baseline_start)) &
        (df["date"] <= pd.to_datetime(baseline_end)) &
        (df["site"].isin(selected_sites))
    ]

    # 1) For the first chart
    agg_map_adm = {"admissions": "sum", "discharges": "sum"}
    baseline_1_df = daily_or_weekly(baseline_data, frequency, agg_map_adm)
    baseline_means_1 = {
        "admissions": baseline_1_df["admissions"].mean() if not baseline_1_df.empty else None,
        "discharges": baseline_1_df["discharges"].mean() if not baseline_1_df.empty else None
    }

    # 2) For the second chart
    agg_map_los = {
        "patients LoS 21+ days": "mean",
        "patients LoS 14+ days": "mean"
    }
    baseline_2_df = daily_or_weekly(baseline_data, frequency, agg_map_los)
    baseline_means_2 = {
        "patients LoS 21+ days": baseline_2_df["patients LoS 21+ days"].mean() if not baseline_2_df.empty else None,
        "patients LoS 14+ days": baseline_2_df["patients LoS 14+ days"].mean() if not baseline_2_df.empty else None
    }

    # 3) For the third chart
    agg_map_beds = {"escalation beds": "mean", "boarded beds": "mean"}
    baseline_3_df = daily_or_weekly(baseline_data, frequency, agg_map_beds)
    baseline_means_3 = {
        "escalation beds": baseline_3_df["escalation beds"].mean() if not baseline_3_df.empty else None,
        "boarded beds": baseline_3_df["boarded beds"].mean() if not baseline_3_df.empty else None
    }

# Download
st.sidebar.download_button(
    label="Download Data",
    data=filtered_data.to_csv(index=False),
    file_name="discharge_data.csv",
    mime="text/csv"
)

tab1, tab2 = st.tabs(["Daily Sitrep Metrics", "DPTL Metrics"])

with tab1:
    _, col1, _ = st.columns([1,10,1])
    
    with col1:
        # -------------------------------
        # 1) Admissions & Discharges
        # -------------------------------
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

        # Chart Settings
        with st.expander("Chart Settings"):
            show_trend_1 = st.checkbox("Show Trendline", value=False, key="adm_dis_trend")
            show_baseline_1 = st.checkbox("Show Baseline", value=False, key="adm_dis_base")

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
            show_trendline=show_trend_1,
            show_baseline=show_baseline_1,
            baseline_means=baseline_means_1
        )
        st.plotly_chart(fig1, use_container_width=True)

        # -------------------------------
        # 2) 21+ LoS and 14+ LoS
        # -------------------------------
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"21+ LoS and 14+ LoS{suffix}")

        agg_map_los = {
            "patients LoS 21+ days": "mean",
            "patients LoS 14+ days": "mean"
        }
        chart_data_2 = daily_or_weekly(filtered_data, frequency, agg_map_los)

        # Metric cards
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

        with st.expander("Chart Settings"):
            show_trend_2 = st.checkbox("Show Trendline", value=False, key="los_trend")
            show_baseline_2 = st.checkbox("Show Baseline", value=False, key="los_base")

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
            show_trendline=show_trend_2,
            show_baseline=show_baseline_2,
            baseline_means=baseline_means_2
        )
        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------
        # 3) Escalation & Boarded Beds
        # -------------------------------
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Escalation & Boarded Beds{suffix}")

        agg_map_beds = {"escalation beds": "mean", "boarded beds": "mean"}
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

        # Chart Settings
        with st.expander("Chart Settings"):
            # Optionally enable trendlines or baseline if desired:
            # (But you only mentioned toggling between line & stacked bar,
            #  so we'll keep the checkboxes for completeness.)
            show_trend_3 = st.checkbox("Show Trendline", value=False, key="beds_trend")
            show_baseline_3 = st.checkbox("Show Baseline", value=False, key="beds_base")
            chart_type_3 = st.radio(
                "Chart Type:",
                ["Stacked Bar", "Line"],
                index=0,
                key="beds_chart_type"
            )

        color_map_3 = {
            "escalation beds": "#2ca02c",
            "boarded beds": "#d62728"
        }

        # If user selects "Line," call create_line_chart. Otherwise stacked bar.
        if chart_type_3 == "Line":
            # We'll produce a multi-line chart of these 2 metrics
            fig3 = create_line_chart(
                data=chart_data_3,
                x_col="date",
                y_cols=["escalation beds", "boarded beds"],
                color_map=color_map_3,
                labels={"value": "Beds", "variable": "Bed Type"},
                show_trendline=show_trend_3,
                show_baseline=show_baseline_3,
                baseline_means=baseline_means_3
            )
        else:
            # Stacked bar
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
