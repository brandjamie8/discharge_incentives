import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

# ------------------------------------------------------------------------
# 1) Load First Dataset (existing)
# ------------------------------------------------------------------------
@st.cache_data
def load_first_dataset():
    # Example: admissions/discharges data
    df1 = pd.read_csv("data/SitrepData.csv")
    df1["date"] = pd.to_datetime(df1["date"], format="%d/%m/%Y", errors="coerce")
    df1.sort_values("date", inplace=True)
    return df1

# ------------------------------------------------------------------------
# 2) Load Second Dataset (NMCTR external delay)
# ------------------------------------------------------------------------
@st.cache_data
def load_nmctr_dataset():
    """
    Columns:
        date, site, Borough, Pathway, 
        NMCTR external delay, Total external delay days
    """
    df2 = pd.read_csv("data/NMCTR_Delays.csv")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2.sort_values("date", inplace=True)
    return df2

# ------------------------------------------------------------------------
# Helper to get default date range
# ------------------------------------------------------------------------
def get_default_date_range(df, freq):
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

# ------------------------------------------------------------------------
# Aggregation Logic for NMCTR data
#  - daily = sum
#  - weekly = average per day
#  - plus ability to "split" by site, Borough, or Pathway
# ------------------------------------------------------------------------
def aggregate_nmctr_data(df, freq, group_by):
    """
    group_by: one of ["site", "Borough", "Pathway"] 
              (or None if you want a single line).
    This function returns a DataFrame with columns:
      ["date", group_by, "NMCTR external delay", "Total external delay days"]
    where daily = sum, weekly = average (across the 7 daily sums).
    """

    # 1) If group_by is None or "none", we won't group on that column
    group_cols = ["date"]
    if group_by is not None and group_by.lower() not in ("none", ""):
        group_cols.append(group_by)

    # 2) Daily sum
    daily_agg = df.groupby(group_cols, as_index=False).agg({
        "NMCTR external delay": "sum",
        "Total external delay days": "sum",
    })

    if freq == "daily":
        return daily_agg  # sum across that group for each date

    # 3) Weekly => average of the daily sums
    weekly_agg = (
        daily_agg.set_index("date")
                 .resample("W-MON", label="left", closed="left")
                 .agg({
                     "NMCTR external delay": "mean",
                     "Total external delay days": "mean"
                 })
                 .reset_index()
    )
    # if group_by was included, we must re-group after resample
    if len(group_cols) > 1:
        # We'll resample for each group_by category separately.
        # One approach: pivot on group_by -> resample each column -> melt back.
        # But a simpler approach is:
        #   1) separate daily_agg by each group, 2) resample => average, 3) concat
        # For clarity, let's do a quick approach:

        results = []
        # group_cols[0] = "date", group_cols[1] = group_by
        # We'll group daily_agg by the "group_by" dimension, then resample
        if len(group_cols) > 1:
            unique_categories = daily_agg[group_by].unique()
            for cat_val in unique_categories:
                sub = daily_agg[daily_agg[group_by] == cat_val].copy()
                sub.set_index("date", inplace=True)
                sub_weekly = (
                    sub.resample("W-MON", label="left", closed="left")
                       .mean(numeric_only=True)
                       .reset_index()
                )
                sub_weekly[group_by] = cat_val
                results.append(sub_weekly)
            weekly_agg = pd.concat(results, ignore_index=True)

    return weekly_agg


# ------------------------------------------------------------------------
# Compute a second column "avg external delay days" 
#   = Total external delay days / NMCTR external delay
#   for each row (daily or weekly).
# ------------------------------------------------------------------------
def add_avg_external_delay_col(df):
    df = df.copy()  # don’t modify in place
    df["avg external delay days"] = np.where(
        df["NMCTR external delay"] != 0,
        df["Total external delay days"] / df["NMCTR external delay"],
        0  # or np.nan, up to you
    )
    return df

# ------------------------------------------------------------------------
# Convert a numeric (possibly numpy type) to a safe float for st.metric
# ------------------------------------------------------------------------
def safe_float(x, decimals=1):
    return float(round(x, decimals))


# ------------------------------------------------------------------------
# Create line chart for NMCTR data
#   y_cols could be ["NMCTR external delay"] or ["avg external delay days"], etc.
#   We incorporate grouping dimension in color, plus optional trendline & baseline
#   (just like your existing create_line_chart).
# ------------------------------------------------------------------------
def create_nmctr_line_chart(
    data, 
    x_col, 
    y_col, 
    color_col=None,    # "site", "Borough", or "Pathway"
    color_map=None, 
    labels=None,
    show_trendline=False,
    show_baseline=False,
    baseline_mean=None,
):
    """
    Similar to your create_line_chart, but we only have one Y column 
    that might be multiple categories if color_col is used. 
    We'll pivot or rely on px.line's color argument.

    data columns e.g.: ["date", "site", "NMCTR external delay", "avg external delay days", ...]
    y_col is a string, e.g. "NMCTR external delay" or "avg external delay days"
    color_col can be "site", "Borough", "Pathway", or None.

    If show_baseline=True and baseline_mean is not None, we draw a single horizontal line.
      (If you want multiple lines for each site/borough, you'd do more logic.)
    """

    # If we want multiple lines by color_col, we just pass it to px.line
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color=color_col,     # None => single line
        labels=labels
    )
    fig.update_traces(line=dict(width=3))

    # (A) Trendline if requested
    if show_trendline:
        # We do one line fit per color category
        # px.scatter trendline approach won't easily show lines,
        # so let's do a manual approach.
        # We'll do it for each color_col group if color_col is not None:
        if color_col is not None and color_col in data.columns:
            categories = data[color_col].dropna().unique()
            for cat_val in categories:
                sub = data[data[color_col] == cat_val].dropna(subset=[y_col])
                if len(sub) < 2:
                    continue
                # Fit line
                x_ord = sub[x_col].apply(lambda d: d.toordinal()).values
                y_vals = sub[y_col].values
                if len(x_ord) < 2:
                    continue
                slope, intercept = np.polyfit(x_ord, y_vals, 1)
                x_min, x_max = x_ord.min(), x_ord.max()
                x_fit = np.linspace(x_min, x_max, 50)
                y_fit = slope * x_fit + intercept
                dt_fit = [datetime.date.fromordinal(int(xx)) for xx in x_fit]

                c = "blue"
                # If you want custom color, you'd build a color map from category
                # or just let plotly pick. For brevity let's keep a single color.
                fig.add_trace(
                    go.Scatter(
                        x=dt_fit,
                        y=y_fit,
                        mode="lines",
                        line=dict(dash="dot", width=2, color=c),
                        name=f"{cat_val} Trendline"
                    )
                )
        else:
            # No color_col => single line
            sub = data.dropna(subset=[y_col])
            if len(sub) >= 2:
                x_ord = sub[x_col].apply(lambda d: d.toordinal()).values
                y_vals = sub[y_col].values
                slope, intercept = np.polyfit(x_ord, y_vals, 1)
                x_min, x_max = x_ord.min(), x_ord.max()
                x_fit = np.linspace(x_min, x_max, 50)
                y_fit = slope * x_fit + intercept
                dt_fit = [datetime.date.fromordinal(int(xx)) for xx in x_fit]
                fig.add_trace(
                    go.Scatter(
                        x=dt_fit,
                        y=y_fit,
                        mode="lines",
                        line=dict(dash="dot", width=2, color="blue"),
                        name="Trendline"
                    )
                )

    # (B) Baseline if requested
    # For simplicity, we do a single baseline line for the entire chart
    # If you want separate lines for each site/borough, you'd do more logic.
    if show_baseline and baseline_mean is not None:
        fig.add_hline(
            y=baseline_mean,
            line=dict(color="black", dash="dash", width=2),
            annotation=None
        )
        fig.add_annotation(
            x=0, y=1,
            xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            text=f"Baseline Mean: {round(baseline_mean, 2)}",
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=5,
            bgcolor="white"
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ------------------------------------------------------------------------
# Streamlit layout
# ------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

# Load both datasets
df1 = load_first_dataset()       # admissions/discharges, etc.
df2 = load_nmctr_dataset()       # NMCTR external delay

st.title("Combined Dashboard")

# 1) Shared Frequency
frequency = st.sidebar.radio("Frequency", ["daily", "weekly"], index=0)

# 2) Date range filters (we'll unify for both, or do them separately if you prefer)
start_default_1, end_default_1 = get_default_date_range(df1, frequency)
start_default_2, end_default_2 = get_default_date_range(df2, frequency)
# For simplicity, pick the earliest of the start_defaults, etc.
unified_start = min(start_default_1, start_default_2)
unified_end = max(end_default_1, end_default_2)

date_input = st.sidebar.date_input(
    "Select Date Range",
    value=(unified_start, unified_end)
)
if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date, end_date = date_input
else:
    start_date, end_date = unified_start, unified_end

# 3) Filter df1 by date
df1_filtered = df1[
    (df1["date"] >= pd.to_datetime(start_date)) &
    (df1["date"] <= pd.to_datetime(end_date))
]
# 4) Filter df2 by date
df2_filtered = df2[
    (df2["date"] >= pd.to_datetime(start_date)) &
    (df2["date"] <= pd.to_datetime(end_date))
]

# 5) Shared site filter
all_sites_1 = df1_filtered["site"].dropna().unique()
all_sites_2 = df2_filtered["site"].dropna().unique()
all_sites = sorted(set(all_sites_1).union(all_sites_2))

selected_sites = st.sidebar.multiselect("Select Site(s)", all_sites, default=all_sites)

df1_filtered = df1_filtered[df1_filtered["site"].isin(selected_sites)]
df2_filtered = df2_filtered[df2_filtered["site"].isin(selected_sites)]


# -------------------------------
# Baseline approach
# -------------------------------
st.sidebar.subheader("Baseline Period")
use_baseline = st.sidebar.checkbox("Define a baseline period?", value=False)

# We'll store baseline means for the new NMCTR charts
baseline_nmctr_total = None  # chart A
baseline_nmctr_avg   = None  # chart B

if use_baseline:
    baseline_range = st.sidebar.date_input(
        "Select baseline date range (inclusive):",
        value=(unified_start, unified_end)
    )
    if isinstance(baseline_range, tuple) and len(baseline_range) == 2:
        base_start, base_end = baseline_range
    else:
        base_start, base_end = unified_start, unified_end

    df2_baseline = df2[
        (df2["date"] >= pd.to_datetime(base_start)) &
        (df2["date"] <= pd.to_datetime(base_end)) &
        (df2["site"].isin(selected_sites))
    ]
    # For baseline, we'll just do a single line (no grouping). 
    # If you want separate baselines for each site, borough, etc., you'd do more logic.
    # We'll do daily= sum, weekly= average
    base_agg = aggregate_nmctr_data(df2_baseline, frequency, group_by=None)
    # a) baseline for NMCTR external delay
    if not base_agg.empty:
        baseline_nmctr_total = base_agg["NMCTR external delay"].mean()
        # b) for average external delay days => need to add column
        base_agg = add_avg_external_delay_col(base_agg)
        baseline_nmctr_avg = base_agg["avg external delay days"].mean()


# Now we do separate tabs for the old metrics vs new NMCTR metrics
tab1, tab2, tab3 = st.tabs(["Original Data", "DPTL Metrics", "NMCTR Delays"])

with tab1:
    st.write("Your existing charts (admissions, discharges, etc.) go here...")

with tab2:
    st.write("DPTL Metrics content goes here...")

with tab3:
    st.header("NMCTR External Delay Charts")

    # Extra filters for Borough & Pathway
    all_boroughs = df2_filtered["Borough"].dropna().unique()
    selected_boroughs = st.multiselect("Select Borough(s)", all_boroughs, default=all_boroughs)

    all_pathways = df2_filtered["Pathway"].dropna().unique()
    selected_pathways = st.multiselect("Select Pathway(s)", all_pathways, default=all_pathways)

    # Filter by borough, pathway
    df2_filtered = df2_filtered[df2_filtered["Borough"].isin(selected_boroughs)]
    df2_filtered = df2_filtered[df2_filtered["Pathway"].isin(selected_pathways)]

    # One last setting: how to "split lines by" site, borough, or pathway?
    st.subheader("NMCTR Chart Settings")
    split_lines_by = st.radio(
        "Split lines by:",
        ["None", "site", "Borough", "Pathway"],
        index=0
    )

    # ~~~~~~~~~ Chart A: NMCTR External Delay ~~~~~~~~
    st.markdown("### Chart A: NMCTR External Delay")
    # aggregator
    chartA_data = aggregate_nmctr_data(df2_filtered, frequency, group_by=None if split_lines_by=="None" else split_lines_by)
    # We now have columns: ["date", (split_lines_by?), "NMCTR external delay", "Total external delay days"]
    # We'll do daily= sum, weekly= average
    show_trend_A = st.checkbox("Show Trendline (Chart A)")
    show_base_A  = st.checkbox("Show Baseline (Chart A)")

    figA = create_nmctr_line_chart(
        data=chartA_data,
        x_col="date",
        y_col="NMCTR external delay",
        color_col=None if split_lines_by=="None" else split_lines_by,
        labels={"NMCTR external delay": "Delay (patients)", "date": "Date"},
        show_trendline=show_trend_A,
        show_baseline=show_base_A,
        baseline_mean=baseline_nmctr_total
    )
    st.plotly_chart(figA, use_container_width=True)


    # ~~~~~~~~~ Chart B: Average External Delay Days ~~~~~~~~
    st.markdown("### Chart B: Average External Delay Days")
    # We'll define "avg external delay days" = Total external delay days / NMCTR external delay
    # step 1) aggregator
    chartB_data = aggregate_nmctr_data(df2_filtered, frequency, group_by=None if split_lines_by=="None" else split_lines_by)
    # step 2) add the "avg external delay days" column
    chartB_data = add_avg_external_delay_col(chartB_data)
    show_trend_B = st.checkbox("Show Trendline (Chart B)")
    show_base_B  = st.checkbox("Show Baseline (Chart B)")

    figB = create_nmctr_line_chart(
        data=chartB_data,
        x_col="date",
        y_col="avg external delay days",
        color_col=None if split_lines_by=="None" else split_lines_by,
        labels={"avg external delay days": "Avg Delay (days)", "date": "Date"},
        show_trendline=show_trend_B,
        show_baseline=show_base_B,
        baseline_mean=baseline_nmctr_avg
    )
    st.plotly_chart(figB, use_container_width=True)


    # Example metric display for the latest vs previous
    # (just for demonstration)
    if not chartB_data.empty:
        # sort by date ascending
        chartB_data.sort_values("date", inplace=True)
        latest_val = chartB_data.iloc[-1]["avg external delay days"]
        if len(chartB_data) > 1:
            prev_val = chartB_data.iloc[-2]["avg external delay days"]
        else:
            prev_val = 0
        st.metric(
            label="Avg Delay Days (Latest vs Previous)",
            value=round(float(latest_val),2),
            delta=round(float(latest_val - prev_val),2)
        )
