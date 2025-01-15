import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

# --------------------------------------
# Data Loading
# --------------------------------------
@st.cache_data
def load_data():
    """Load and prepare the primary CSV data (SitrepData)."""
    df = pd.read_csv("data/SitrepData.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df.sort_values("date", inplace=True)
    return df

@st.cache_data
def load_external_delay_data():
    """Load and prepare the second CSV data for external delays."""
    df2 = pd.read_csv("data/ExternalDelays.csv")
    df2["date"] = pd.to_datetime(df2["date"], format="%d/%m/%Y", errors="coerce")
    df2.sort_values("date", inplace=True)
    return df2

# --------------------------------------
# Determine Default Date Range
# --------------------------------------
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

# --------------------------------------
# Aggregation Logic
# --------------------------------------
def aggregate_chart_data(df, frequency, chart_id):
    """
    Returns a daily or weekly aggregated DataFrame based on chart_id logic:
      - chart_id=1 => Admissions/Discharges
           * daily => sum
           * weekly => sum (weekly total)
      - chart_id=2 => 21+ LoS & 14+ LoS
           * daily => sum
           * weekly => mean (average per day)
      - chart_id=3 => Escalation & Boarded Beds
           * daily => sum
           * weekly => mean (average per day)
      - chart_id=4 => External Delays (NMCTR external delay, Total external delay days)
           * daily => sum
           * weekly => mean (average per day)
    """

    if df.empty:
        return df

    if chart_id == 1:
        needed_cols = ["admissions", "discharges"]
    elif chart_id == 2:
        needed_cols = ["patients LoS 21+ days", "patients LoS 14+ days"]
    elif chart_id == 3:
        needed_cols = ["escalation beds", "boarded beds"]
    elif chart_id == 4:
        needed_cols = ["NMCTR external delay", "Total external delay days"]
    else:
        needed_cols = df.columns.drop(["date", "site"], errors="ignore")  # fallback

    # 1) Daily-level sum
    daily_agg_map = {col: "sum" for col in needed_cols}
    daily_df = df.groupby("date", as_index=False).agg(daily_agg_map)

    # 2) If daily frequency, return daily_df
    if frequency == "daily":
        return daily_df

    # 3) Otherwise, weekly
    #    chart_id=1 => weekly sum
    #    chart_id=2,3,4 => weekly mean
    if chart_id == 1:
        weekly_agg_map = {col: "sum" for col in needed_cols}
    else:
        weekly_agg_map = {col: "mean" for col in needed_cols}

    weekly_df = (
        daily_df.set_index("date")
                .resample("W-MON", label="left", closed="left")
                .agg(weekly_agg_map)
                .reset_index()
    )
    return weekly_df


# --------------------------------------
# Helper: Return latest & previous values
# --------------------------------------
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


# --------------------------------------
# Chart Creation Functions
# --------------------------------------
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

    fig = px.line(
        df_melted,
        x=x_col,
        y="value",
        color="variable",
        color_discrete_map=color_discrete_map,
        labels=labels
    )
    fig.update_traces(line=dict(width=3))

    # -- Trendline
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

    # -- Baseline
    baseline_texts = []
    if show_baseline and baseline_means is not None:
        for var in y_cols:
            base_val = baseline_means.get(var, None)
            if base_val is None:
                continue
            fig.add_hline(
                y=base_val,
                line=dict(
                    color=color_discrete_map.get(var, "#000000"),
                    dash="dash",
                    width=2
                ),
                annotation=None
            )
            baseline_texts.append(f"{var}: {round(base_val, 2)}")

    if baseline_texts:
        annotation_text = "Baseline Means:<br>" + "<br>".join(baseline_texts)
        fig.add_annotation(
            x=0, y=1,
            xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            text=annotation_text,
            showarrow=False,
            bgcolor="white"
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


def create_stacked_bar_chart(data, x_col, y_cols, color_map=None, labels=None):
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


# --------------------------------------
# Streamlit Layout
# --------------------------------------
st.set_page_config(page_title="LGT Discharge Incentives Dashboard", layout="wide")
st.title("Discharge Incentives Monitoring")

# -- Sidebar --
st.sidebar.header("Filters & Settings")
df = load_data()
df2 = load_external_delay_data()  # <-- Load the second dataset

frequency = st.sidebar.radio("Frequency", ["daily", "weekly"], index=0)

# Determine default date range from primary data
start_default, end_default = get_default_date_range(df, frequency)

date_input = st.sidebar.date_input("Select Date Range", value=(start_default, end_default))
if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date, end_date = date_input
elif isinstance(date_input, datetime.date):
    start_date, end_date = date_input, date_input
else:
    start_date, end_date = start_default, end_default

# Filter both datasets by date
filtered_data = df[
    (df["date"] >= pd.to_datetime(start_date)) & 
    (df["date"] <= pd.to_datetime(end_date))
]

filtered_data2 = df2[
    (df2["date"] >= pd.to_datetime(start_date)) & 
    (df2["date"] <= pd.to_datetime(end_date))
]

# Site filter
all_sites = filtered_data["site"].unique()
selected_sites = st.sidebar.multiselect("Select Site(s)", all_sites, default=all_sites)

filtered_data = filtered_data[filtered_data["site"].isin(selected_sites)]
filtered_data2 = filtered_data2[filtered_data2["site"].isin(selected_sites)]

# Baseline option
st.sidebar.subheader("Baseline Period")
use_baseline = st.sidebar.checkbox("Define a baseline period?", value=False)

baseline_means_1 = None
baseline_means_2 = None
baseline_means_3 = None
baseline_means_4 = None  # For external delays

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

    # Filter the data for the baseline range (for both datasets)
    baseline_data = df[
        (df["date"] >= pd.to_datetime(baseline_start)) &
        (df["date"] <= pd.to_datetime(baseline_end)) &
        (df["site"].isin(selected_sites))
    ]
    baseline_data2 = df2[
        (df2["date"] >= pd.to_datetime(baseline_start)) &
        (df2["date"] <= pd.to_datetime(baseline_end)) &
        (df2["site"].isin(selected_sites))
    ]

    # Chart 1 baseline
    baseline_1 = aggregate_chart_data(baseline_data, frequency, chart_id=1)
    baseline_means_1 = {
        "admissions": baseline_1["admissions"].mean() if not baseline_1.empty else None,
        "discharges": baseline_1["discharges"].mean() if not baseline_1.empty else None
    }

    # Chart 2 baseline
    baseline_2 = aggregate_chart_data(baseline_data, frequency, chart_id=2)
    baseline_means_2 = {
        "patients LoS 21+ days": baseline_2["patients LoS 21+ days"].mean() if not baseline_2.empty else None,
        "patients LoS 14+ days": baseline_2["patients LoS 14+ days"].mean() if not baseline_2.empty else None
    }

    # Chart 3 baseline
    baseline_3 = aggregate_chart_data(baseline_data, frequency, chart_id=3)
    baseline_means_3 = {
        "escalation beds": baseline_3["escalation beds"].mean() if not baseline_3.empty else None,
        "boarded beds": baseline_3["boarded beds"].mean() if not baseline_3.empty else None
    }

    # Chart 4 baseline (External Delays)
    baseline_4 = aggregate_chart_data(baseline_data2, frequency, chart_id=4)
    baseline_means_4 = {
        "NMCTR external delay": baseline_4["NMCTR external delay"].mean() if not baseline_4.empty else None,
        "Total external delay days": baseline_4["Total external delay days"].mean() if not baseline_4.empty else None
    }

# Download button (primary data only or you can combine)
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
        # ------------------------------------------------
        # Chart 1: Admissions & Discharges
        # ------------------------------------------------
        suffix = " (weekly total)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Admissions and Discharges{suffix}")

        chart_data_1 = aggregate_chart_data(filtered_data, frequency, chart_id=1)

        latest_adm, prev_adm = get_latest_and_previous_values(chart_data_1["admissions"])
        latest_dis, prev_dis = get_latest_and_previous_values(chart_data_1["discharges"])

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                label="Admissions (Latest vs Previous)",
                value=int(latest_adm),
                delta=int(latest_adm - prev_adm),
            )
        with c2:
            st.metric(
                label="Discharges (Latest vs Previous)",
                value=int(latest_dis),
                delta=int(latest_dis - prev_dis),
            )

        # Chart Settings
        with st.expander("Chart Settings"):
            show_trend_1 = st.checkbox("Show Trendline", value=False, key="adm_dis_trend")
            show_baseline_1 = st.checkbox("Show Baseline", value=False, key="adm_dis_base")

        color_map_1 = {"admissions": "#1f77b4", "discharges": "#ff7f0e"}
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

        # ------------------------------------------------
        # Chart 2: 21+ LoS & 14+ LoS
        # ------------------------------------------------
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"21+ LoS and 14+ LoS{suffix}")

        chart_data_2 = aggregate_chart_data(filtered_data, frequency, chart_id=2)

        latest_21plus, prev_21plus = get_latest_and_previous_values(chart_data_2["patients LoS 21+ days"])
        latest_14plus, prev_14plus = get_latest_and_previous_values(chart_data_2["patients LoS 14+ days"])

        c3, c4 = st.columns(2)
        with c3:
            st.metric(
                label="21+ LoS (Latest vs Previous)",
                value=float(round(latest_21plus, 1)),
                delta=float(round(latest_21plus - prev_21plus, 1)),
            )
        with c4:
            st.metric(
                label="14+ LoS (Latest vs Previous)",
                value=float(round(latest_14plus, 1)),
                delta=float(round(latest_14plus - prev_14plus, 1)),
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

        # ------------------------------------------------
        # Chart 3: Escalation & Boarded Beds
        # ------------------------------------------------
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Escalation & Boarded Beds{suffix}")

        chart_data_3 = aggregate_chart_data(filtered_data, frequency, chart_id=3)

        latest_esc, prev_esc = get_latest_and_previous_values(chart_data_3["escalation beds"])
        latest_boarded, prev_boarded = get_latest_and_previous_values(chart_data_3["boarded beds"])

        c5, c6 = st.columns(2)
        with c5:
            st.metric(
                label="Escalation Beds (Latest vs Previous)",
                value=float(round(latest_esc, 1)),
                delta=float(round(latest_esc - prev_esc, 1)),
            )
        with c6:
            st.metric(
                label="Boarded Beds (Latest vs Previous)",
                value=float(round(latest_boarded, 1)),
                delta=float(round(latest_boarded - prev_boarded, 1)),
            )

        with st.expander("Chart Settings"):
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

        if chart_type_3 == "Line":
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
            fig3 = create_stacked_bar_chart(
                data=chart_data_3,
                x_col="date",
                y_cols=["escalation beds", "boarded beds"],
                color_map=color_map_3,
                labels={"value": "Beds", "variable": "Bed Type"}
            )
        st.plotly_chart(fig3, use_container_width=True)

        # ------------------------------------------------
        # CHART 4: External Delay (From Second Dataset)
        # ------------------------------------------------
        #   - daily => sum
        #   - weekly => average
        #   - columns => "NMCTR external delay" & "Total external delay days"
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"External Delays{suffix}")

        chart_data_4 = aggregate_chart_data(filtered_data2, frequency, chart_id=4)

        # Extract the latest & previous values
        latest_nmctr, prev_nmctr = get_latest_and_previous_values(chart_data_4["NMCTR external delay"])
        latest_total, prev_total = get_latest_and_previous_values(chart_data_4["Total external delay days"])

        c7, c8 = st.columns(2)
        with c7:
            st.metric(
                label="NMCTR External Delay (Latest vs Previous)",
                value=float(round(latest_nmctr, 1)),
                delta=float(round(latest_nmctr - prev_nmctr, 1)),
            )
        with c8:
            st.metric(
                label="Total External Delay Days (Latest vs Previous)",
                value=float(round(latest_total, 1)),
                delta=float(round(latest_total - prev_total, 1)),
            )

        with st.expander("Chart Settings"):
            show_trend_4 = st.checkbox("Show Trendline", value=False, key="ext_trend")
            show_baseline_4 = st.checkbox("Show Baseline", value=False, key="ext_base")

        color_map_4 = {
            "NMCTR external delay": "#9467bd",
            "Total external delay days": "#8c564b"
        }

        fig4 = create_line_chart(
            data=chart_data_4,
            x_col="date",
            y_cols=["NMCTR external delay", "Total external delay days"],
            color_map=color_map_4,
            labels={"value": "Delays", "variable": "Delay Type"},
            show_trendline=show_trend_4,
            show_baseline=show_baseline_4,
            baseline_means=baseline_means_4
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.write("DPTL Metrics content goes here.")
