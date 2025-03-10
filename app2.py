import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import io

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
      - If 'weekly': last 26 complete weeks (Mon-Sun).
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
# Aggregation Logic (Supports chart_id=1..6 and split_by)
# --------------------------------------
def aggregate_chart_data(df, frequency, chart_id, split_by=None):
    """
    Returns a daily or weekly aggregated DataFrame based on chart_id logic.
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
    elif chart_id == 5:
        needed_cols = ["NMCTR external delay", "Total external delay days"]
    elif chart_id == 6:
        needed_cols = ["Total external delay days"]
    else:
        needed_cols = df.columns.drop(["date", "site", "Borough", "Pathway"], errors="ignore")

    group_cols = ["date"]
    if split_by and split_by in df.columns:
        group_cols.append(split_by)

    daily_df = df.groupby(group_cols, as_index=False)[needed_cols].sum()

    if chart_id == 5:
        daily_df["average_delay_days"] = daily_df.apply(
            lambda row: (row["Total external delay days"] / row["NMCTR external delay"])
            if row["NMCTR external delay"] != 0 else 0,
            axis=1
        )

    if frequency == "daily":
        if chart_id == 5:
            keep_cols = ["date"]
            if split_by and split_by in df.columns:
                keep_cols.append(split_by)
            keep_cols.append("average_delay_days")
            return daily_df[keep_cols]
        else:
            return daily_df

    daily_df.set_index("date", inplace=True)
    if chart_id == 1:
        if split_by and split_by in df.columns:
            weekly_df = (
                daily_df.groupby(split_by)
                        .resample("W-MON", label="left", closed="left")[needed_cols]
                        .sum()
                        .reset_index()
            )
        else:
            weekly_agg_map = {col: "sum" for col in needed_cols}
            weekly_df = (
                daily_df.resample("W-MON", label="left", closed="left")
                        .agg(weekly_agg_map)
                        .reset_index()
            )
    elif chart_id in [2, 3, 4]:
        if split_by and split_by in df.columns:
            weekly_df = (
                daily_df.groupby(split_by)
                        .resample("W-MON", label="left", closed="left")[needed_cols]
                        .mean()
                        .reset_index()
            )
        else:
            weekly_agg_map = {col: "mean" for col in needed_cols}
            weekly_df = (
                daily_df.resample("W-MON", label="left", closed="left")
                        .agg(weekly_agg_map)
                        .reset_index()
            )
    elif chart_id == 5:
        if split_by and split_by in df.columns:
            sum_df = (
                daily_df.groupby(split_by)
                        .resample("W-MON", label="left", closed="left")[["NMCTR external delay", "Total external delay days"]]
                        .sum()
                        .reset_index()
            )
        else:
            sum_df = (
                daily_df.resample("W-MON", label="left", closed="left")[["NMCTR external delay", "Total external delay days"]]
                .sum()
                .reset_index()
            )
        sum_df["average_delay_days"] = sum_df.apply(
            lambda row: (row["Total external delay days"] / row["NMCTR external delay"])
            if row["NMCTR external delay"] != 0 else 0,
            axis=1
        )
        weekly_df = sum_df
    elif chart_id == 6:
        if split_by and split_by in df.columns:
            weekly_df = (
                daily_df.groupby(split_by)
                        .resample("W-MON", label="left", closed="left")[needed_cols]
                        .sum()
                        .reset_index()
            )
        else:
            weekly_agg_map = {col: "sum" for col in needed_cols}
            weekly_df = (
                daily_df.resample("W-MON", label="left", closed="left")
                        .agg(weekly_agg_map)
                        .reset_index()
            )
    daily_df.reset_index(drop=True, inplace=True)
    return weekly_df.reset_index(drop=True)

# --------------------------------------
# Helper: Return latest & previous values
# --------------------------------------
def get_latest_and_previous_values(series):
    if len(series) == 0:
        return 0, 0
    elif len(series) == 1:
        return series.iloc[-1], 0
    else:
        return series.iloc[-1], series.iloc[-2]

# --------------------------------------
# Chart Creation Functions
# --------------------------------------
def create_line_chart(data, x_col, y_cols, color_map=None, labels=None,
                      show_trendline=False, show_baseline=False, baseline_means=None,
                      trendline_label="Trendline"):
    df_melted = data.melt(id_vars=[x_col], value_vars=y_cols,
                          var_name="variable", value_name="value")
    if color_map:
        color_discrete_map = {var: color_map.get(var, "#000000") for var in y_cols}
    else:
        color_discrete_map = {}
    fig = px.line(df_melted, x=x_col, y="value", color="variable",
                  color_discrete_map=color_discrete_map, labels=labels)
    fig.update_traces(line=dict(width=3))
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
                    line=dict(dash="dot", width=2,
                              color=color_discrete_map.get(var, "#000000")),
                    name=f"{var} {trendline_label}"
                )
            )
    baseline_texts = []
    if show_baseline and baseline_means is not None:
        for var in y_cols:
            base_val = baseline_means.get(var, None)
            if base_val is None:
                continue
            fig.add_hline(
                y=base_val,
                line=dict(color=color_discrete_map.get(var, "#000000"), dash="dash", width=2),
                annotation=None
            )
            baseline_texts.append(f"{var}: {round(base_val, 2)}")
    if baseline_texts:
        annotation_text = "Baseline Means:<br>" + "<br>".join(baseline_texts)
        fig.add_annotation(x=0, y=1, xref="paper", yref="paper",
                           xanchor="left", yanchor="top", text=annotation_text,
                           showarrow=False, bgcolor="white")
    fig.update_layout(template="plotly_white",
                      font=dict(size=14),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0.05),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig

def create_stacked_bar_chart(data, x_col, y_cols, color_map=None, labels=None):
    df_melted = data.melt(id_vars=[x_col], value_vars=y_cols,
                          var_name="variable", value_name="value")
    if color_map:
        color_discrete_map = {var: color_map.get(var, "#000000") for var in y_cols}
    else:
        color_discrete_map = {}
    fig = px.bar(df_melted, x=x_col, y="value", color="variable",
                 barmode="stack", color_discrete_map=color_discrete_map, labels=labels)
    fig.update_layout(template="plotly_white",
                      font=dict(size=14),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0.05),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig

# --------------------------------------
# Helper function for adding change date markers
# --------------------------------------
def add_change_date_marker(fig, date, color="#00CC96"):
    x_val = pd.to_datetime(date)
    formatted_date = date.strftime("%d-%b-%y")
    fig.add_vline(x=x_val, line_width=2, line_dash="dot", line_color=color)
    fig.add_annotation(x=x_val, y=1.05, xref="x", yref="paper",
                       text=formatted_date, showarrow=False,
                       font=dict(color=color), yanchor="bottom")

# --------------------------------------
# Streamlit Layout
# --------------------------------------
st.set_page_config(page_title="LGT Discharge Incentives Dashboard", layout="wide")
st.title("Discharge Incentives Monitoring")

# -- Sidebar --
st.sidebar.header("Filters & Settings")
df = load_data()
df2 = load_external_delay_data()

frequency = st.sidebar.radio("Frequency", ["daily", "weekly"], index=0)
start_default, end_default = get_default_date_range(df, frequency)
date_input = st.sidebar.date_input("Select Date Range", value=(start_default, end_default))
if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date, end_date = date_input
elif isinstance(date_input, datetime.date):
    start_date, end_date = date_input, date_input
else:
    start_date, end_date = start_default, end_default

filtered_data = df[(df["date"] >= pd.to_datetime(start_date)) &
                   (df["date"] <= pd.to_datetime(end_date))]
filtered_data2 = df2[(df2["date"] >= pd.to_datetime(start_date)) &
                     (df2["date"] <= pd.to_datetime(end_date))]

all_sites = filtered_data["site"].unique()
selected_sites = st.sidebar.multiselect("Select Site(s)", all_sites, default=all_sites)
filtered_data = filtered_data[filtered_data["site"].isin(selected_sites)]
filtered_data2 = filtered_data2[filtered_data2["site"].isin(selected_sites)]

st.sidebar.subheader("Baseline Period")
use_baseline = st.sidebar.checkbox("Define a baseline period?", value=False)
if use_baseline:
    baseline_range = st.sidebar.date_input("Select baseline date range (inclusive):",
                                           value=(start_default, end_default))
    if isinstance(baseline_range, tuple) and len(baseline_range) == 2:
        baseline_start, baseline_end = baseline_range
    elif isinstance(baseline_range, datetime.date):
        baseline_start, baseline_end = baseline_range, baseline_range
    else:
        baseline_start, baseline_end = start_default, end_default
    baseline_data = df[(df["date"] >= pd.to_datetime(baseline_start)) &
                       (df["date"] <= pd.to_datetime(baseline_end)) &
                       (df["site"].isin(selected_sites))]
    baseline_data2 = df2[(df2["date"] >= pd.to_datetime(baseline_start)) &
                         (df2["date"] <= pd.to_datetime(baseline_end)) &
                         (df2["site"].isin(selected_sites))]
else:
    baseline_data = pd.DataFrame()
    baseline_data2 = pd.DataFrame()

# --------------------------------------
# Change Dates: Multiple change dates via a button
# --------------------------------------
st.sidebar.subheader("Change Dates")
# Initialize session state if not already done
if "change_dates" not in st.session_state:
    st.session_state.change_dates = []

# When the "Add Change Date" button is clicked, add a new date (default to today)
if st.sidebar.button("Add Change Date"):
    st.session_state.change_dates.append(datetime.date.today())

# Display one date picker for each change date stored
for i, cd in enumerate(st.session_state.change_dates):
    new_date = st.sidebar.date_input(f"Change Date {i+1}", value=cd, key=f"change_date_{i}")
    st.session_state.change_dates[i] = new_date

# --------------------------------------
# Download Data Button
# --------------------------------------
output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    filtered_data.to_excel(writer, sheet_name="SitrepData", index=False)
    filtered_data2.to_excel(writer, sheet_name="ExternalDelays", index=False)
excel_bytes = output.getvalue()

st.sidebar.download_button(label="Download Data",
                           data=excel_bytes,
                           file_name="data_extract.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

tab1, tab2 = st.tabs(["Daily Sitrep Metrics", "DPTL Metrics"])

# ================================================
# TAB 1: DAILY SITREP METRICS
# ================================================
with tab1:
    _, col1, _ = st.columns([1,10,1])
    with col1:
        # ---- CHART 1: Admissions & Discharges ----
        suffix = " (weekly total)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Admissions and Discharges{suffix}")
        chart_data_1 = aggregate_chart_data(filtered_data, frequency, chart_id=1)
        latest_adm, prev_adm = get_latest_and_previous_values(chart_data_1["admissions"]) if not chart_data_1.empty else (0, 0)
        latest_dis, prev_dis = get_latest_and_previous_values(chart_data_1["discharges"]) if not chart_data_1.empty else (0, 0)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Admissions (Latest vs Previous)",
                      value=int(latest_adm),
                      delta=int(latest_adm - prev_adm))
        with c2:
            st.metric(label="Discharges (Latest vs Previous)",
                      value=int(latest_dis),
                      delta=int(latest_dis - prev_dis))
        with st.expander("Chart Settings"):
            show_trend_1 = st.checkbox("Show Trendline", value=False, key="adm_dis_trend")
            show_baseline_1 = st.checkbox("Show Baseline", value=False, key="adm_dis_base")
        baseline_means_1 = None
        if use_baseline and show_baseline_1:
            baseline_1 = aggregate_chart_data(baseline_data, frequency, chart_id=1)
            if not baseline_1.empty:
                baseline_means_1 = {
                    "admissions": baseline_1["admissions"].mean(),
                    "discharges": baseline_1["discharges"].mean()
                }
        color_map_1 = {"admissions": "#1f77b4", "discharges": "#ff7f0e"}
        fig1 = create_line_chart(data=chart_data_1, x_col="date",
                                 y_cols=["admissions", "discharges"],
                                 color_map=color_map_1,
                                 labels={"value": "Patients", "variable": "Metric"},
                                 show_trendline=show_trend_1,
                                 show_baseline=show_baseline_1,
                                 baseline_means=baseline_means_1)
        # Add all change date markers (all in the same color)
        for cd in st.session_state.change_dates:
            add_change_date_marker(fig1, cd)
        st.plotly_chart(fig1, use_container_width=True)

        # ---- CHART 2: 21+ LoS & 14+ LoS ----
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"21+ LoS and 14+ LoS{suffix}")
        chart_data_2 = aggregate_chart_data(filtered_data, frequency, chart_id=2)
        latest_21plus, prev_21plus = get_latest_and_previous_values(chart_data_2["patients LoS 21+ days"]) if not chart_data_2.empty else (0, 0)
        latest_14plus, prev_14plus = get_latest_and_previous_values(chart_data_2["patients LoS 14+ days"]) if not chart_data_2.empty else (0, 0)
        c3, c4 = st.columns(2)
        with c3:
            st.metric(label="21+ LoS (Latest vs Previous)",
                      value=float(round(latest_21plus, 1)),
                      delta=float(round(latest_21plus - prev_21plus, 1)))
        with c4:
            st.metric(label="14+ LoS (Latest vs Previous)",
                      value=float(round(latest_14plus, 1)),
                      delta=float(round(latest_14plus - prev_14plus, 1)))
        with st.expander("Chart Settings"):
            show_trend_2 = st.checkbox("Show Trendline", value=False, key="los_trend")
            show_baseline_2 = st.checkbox("Show Baseline", value=False, key="los_base")
        baseline_means_2 = None
        if use_baseline and show_baseline_2:
            baseline_2 = aggregate_chart_data(baseline_data, frequency, chart_id=2)
            if not baseline_2.empty:
                baseline_means_2 = {
                    "patients LoS 21+ days": baseline_2["patients LoS 21+ days"].mean(),
                    "patients LoS 14+ days": baseline_2["patients LoS 14+ days"].mean()
                }
        color_map_2 = {"patients LoS 21+ days": "#e377c2", "patients LoS 14+ days": "#17becf"}
        fig2 = create_line_chart(data=chart_data_2, x_col="date",
                                 y_cols=["patients LoS 21+ days", "patients LoS 14+ days"],
                                 color_map=color_map_2,
                                 labels={"value": "Patients", "variable": "LoS Category"},
                                 show_trendline=show_trend_2,
                                 show_baseline=show_baseline_2,
                                 baseline_means=baseline_means_2)
        for cd in st.session_state.change_dates:
            add_change_date_marker(fig2, cd)
        st.plotly_chart(fig2, use_container_width=True)

        # ---- CHART 3: Escalation & Boarded Beds ----
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Escalation & Boarded Beds{suffix}")
        chart_data_3 = aggregate_chart_data(filtered_data, frequency, chart_id=3)
        latest_esc, prev_esc = get_latest_and_previous_values(chart_data_3["escalation beds"]) if not chart_data_3.empty else (0, 0)
        latest_boarded, prev_boarded = get_latest_and_previous_values(chart_data_3["boarded beds"]) if not chart_data_3.empty else (0, 0)
        c5, c6 = st.columns(2)
        with c5:
            st.metric(label="Escalation Beds (Latest vs Previous)",
                      value=float(round(latest_esc, 1)),
                      delta=float(round(latest_esc - prev_esc, 1)))
        with c6:
            st.metric(label="Boarded Beds (Latest vs Previous)",
                      value=float(round(latest_boarded, 1)),
                      delta=float(round(latest_boarded - prev_boarded, 1)))
        with st.expander("Chart Settings"):
            show_trend_3 = st.checkbox("Show Trendline", value=False, key="beds_trend")
            show_baseline_3 = st.checkbox("Show Baseline", value=False, key="beds_base")
            chart_type_3 = st.radio("Chart Type:", ["Stacked Bar", "Line"], index=0, key="beds_chart_type")
        baseline_means_3 = None
        if use_baseline and show_baseline_3:
            baseline_3 = aggregate_chart_data(baseline_data, frequency, chart_id=3)
            if not baseline_3.empty:
                baseline_means_3 = {
                    "escalation beds": baseline_3["escalation beds"].mean(),
                    "boarded beds": baseline_3["boarded beds"].mean()
                }
        color_map_3 = {"escalation beds": "#2ca02c", "boarded beds": "#d62728"}
        if chart_type_3 == "Line":
            fig3 = create_line_chart(data=chart_data_3, x_col="date",
                                     y_cols=["escalation beds", "boarded beds"],
                                     color_map=color_map_3,
                                     labels={"value": "Beds", "variable": "Bed Type"},
                                     show_trendline=show_trend_3,
                                     show_baseline=show_baseline_3,
                                     baseline_means=baseline_means_3)
        else:
            fig3 = create_stacked_bar_chart(data=chart_data_3, x_col="date",
                                            y_cols=["escalation beds", "boarded beds"],
                                            color_map=color_map_3,
                                            labels={"value": "Beds", "variable": "Bed Type"})
        for cd in st.session_state.change_dates:
            add_change_date_marker(fig3, cd)
        st.plotly_chart(fig3, use_container_width=True)

        # ---- CHART 4: External Delays ----
        suffix = " (avg per day)" if frequency == "weekly" else " (daily)"
        st.subheader(f"External Delays{suffix}")
        with st.expander("Chart Settings"):
            st.write("**Filter by Borough and Pathway**")
            available_boroughs = sorted(filtered_data2["Borough"].dropna().unique())
            selected_boroughs = st.multiselect("Select Borough(s)", options=available_boroughs,
                                               default=available_boroughs, key="ext_borough_4")
            available_pathways = sorted(filtered_data2["Pathway"].dropna().unique())
            selected_pathways = st.multiselect("Select Pathway(s)", options=available_pathways,
                                               default=available_pathways, key="ext_pathway_4")
            split_option_4 = st.radio("Split By:", ["None", "Site", "Borough", "Pathway"],
                                      index=0, key="ext_split_4")
        if split_option_4 == "None":
            split_by_4 = None
        elif split_option_4 == "Site":
            split_by_4 = "site"
        else:
            split_by_4 = split_option_4
        chart_data_4 = aggregate_chart_data(filtered_data2, frequency, chart_id=4, split_by=split_by_4)
        if split_by_4 is None and not chart_data_4.empty:
            latest_nmctr, prev_nmctr = get_latest_and_previous_values(chart_data_4["NMCTR external delay"])
            c7, c8 = st.columns(2)
            with c7:
                st.metric(label="NMCTR External Delay (Latest vs Previous)",
                          value=float(round(latest_nmctr, 1)),
                          delta=float(round(latest_nmctr - prev_nmctr, 1)))
        filters_msg_4 = (f"Filters applied: **Site** = {', '.join(selected_sites)} | "
                         f"**Borough** = {', '.join(selected_boroughs)} | "
                         f"**Pathway** = {', '.join(selected_pathways)}")
        st.markdown(filters_msg_4)
        if split_by_4 is not None and split_by_4 in chart_data_4.columns:
            df_melted_4 = chart_data_4.melt(id_vars=["date", split_by_4],
                                            value_vars=["NMCTR external delay"],
                                            var_name="variable", value_name="value")
            fig4 = px.line(df_melted_4, x="date", y="value", color=split_by_4, line_group=split_by_4)
        else:
            df_melted_4 = chart_data_4.melt(id_vars=["date"],
                                            value_vars=["NMCTR external delay"],
                                            var_name="variable", value_name="value")
            fig4 = px.line(df_melted_4, x="date", y="value", color="variable")
        fig4.update_layout(template="plotly_white",
                           font=dict(size=14),
                           margin=dict(l=0, r=0, t=40, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.05))
        fig4.update_traces(line=dict(width=3))
        for cd in st.session_state.change_dates:
            add_change_date_marker(fig4, cd)
        st.plotly_chart(fig4, use_container_width=True)

        # ---- CHART 5: Average External Delay Days per Patient ----
        suffix = " (weekly)" if frequency == "weekly" else " (daily ratio)"
        st.subheader(f"Average External Delay Days per Patient{suffix}")
        with st.expander("Chart Settings"):
            st.write("**Filter by Borough and Pathway**")
            available_boroughs_5 = sorted(filtered_data2["Borough"].dropna().unique())
            selected_boroughs_5 = st.multiselect("Select Borough(s)", options=available_boroughs_5,
                                                 default=available_boroughs_5, key="ext_borough_5")
            available_pathways_5 = sorted(filtered_data2["Pathway"].dropna().unique())
            selected_pathways_5 = st.multiselect("Select Pathway(s)", options=available_pathways_5,
                                                 default=available_pathways_5, key="ext_pathway_5")
            split_option_5 = st.radio("Split By:", ["None", "Site", "Borough", "Pathway"],
                                      index=0, key="ext_split_5")
        if split_option_5 == "None":
            split_by_5 = None
        elif split_option_5 == "Site":
            split_by_5 = "site"
        else:
            split_by_5 = split_option_5
        chart_data_5 = aggregate_chart_data(filtered_data2, frequency, chart_id=5, split_by=split_by_5)
        if split_by_5 is None and not chart_data_5.empty:
            latest_avg, prev_avg = get_latest_and_previous_values(chart_data_5["average_delay_days"])
            st.metric(label="Average Delay (Latest vs Previous)",
                      value=float(round(latest_avg, 2)),
                      delta=float(round(latest_avg - prev_avg, 2)))
        filters_msg_5 = (f"Filters applied: **Site** = {', '.join(selected_sites)} | "
                         f"**Borough** = {', '.join(selected_boroughs_5)} | "
                         f"**Pathway** = {', '.join(selected_pathways_5)}")
        st.markdown(filters_msg_5)
        if split_by_5 is not None and split_by_5 in chart_data_5.columns:
            fig5 = px.line(chart_data_5, x="date", y="average_delay_days", color=split_by_5, line_group=split_by_5)
        else:
            fig5 = px.line(chart_data_5, x="date", y="average_delay_days")
        fig5.update_layout(template="plotly_white",
                           font=dict(size=14),
                           margin=dict(l=0, r=0, t=40, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.05))
        fig5.update_traces(line=dict(width=3))
        for cd in st.session_state.change_dates:
            add_change_date_marker(fig5, cd)
        st.plotly_chart(fig5, use_container_width=True)

        # ---- CHART 6: Total External Delay Days ----
        suffix = " (weekly total)" if frequency == "weekly" else " (daily)"
        st.subheader(f"Total External Delay Days{suffix}")
        with st.expander("Chart Settings"):
            st.write("**Filter by Borough and Pathway**")
            available_boroughs_6 = sorted(filtered_data2["Borough"].dropna().unique())
            selected_boroughs_6 = st.multiselect("Select Borough(s)", options=available_boroughs_6,
                                                 default=available_boroughs_6, key="ext_borough_6")
            available_pathways_6 = sorted(filtered_data2["Pathway"].dropna().unique())
            selected_pathways_6 = st.multiselect("Select Pathway(s)", options=available_pathways_6,
                                                 default=available_pathways_6, key="ext_pathway_6")
            split_option_6 = st.radio("Split By:", ["None", "Site", "Borough", "Pathway"],
                                      index=0, key="ext_split_6")
        if split_option_6 == "None":
            split_by_6 = None
        elif split_option_6 == "Site":
            split_by_6 = "site"
        else:
            split_by_6 = split_option_6
        chart_data_6 = aggregate_chart_data(filtered_data2, frequency, chart_id=6, split_by=split_by_6)
        if split_by_6 is None and not chart_data_6.empty:
            latest_total, prev_total = get_latest_and_previous_values(chart_data_6["Total external delay days"])
            st.metric(label="Total Delay Days (Latest vs Previous)",
                      value=int(latest_total),
                      delta=int(latest_total - prev_total))
        filters_msg_6 = (f"Filters applied: **Site** = {', '.join(selected_sites)} | "
                         f"**Borough** = {', '.join(selected_boroughs_6)} | "
                         f"**Pathway** = {', '.join(selected_pathways_6)}")
        st.markdown(filters_msg_6)
        if split_by_6 is not None and split_by_6 in chart_data_6.columns:
            fig6 = px.line(chart_data_6, x="date", y="Total external delay days",
                           color=split_by_6, line_group=split_by_6)
        else:
            fig6 = px.line(chart_data_6, x="date", y="Total external delay days")
        fig6.update_layout(template="plotly_white",
                           font=dict(size=14),
                           margin=dict(l=0, r=0, t=40, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.05))
        fig6.update_traces(line=dict(width=3))
        for cd in st.session_state.change_dates:
            add_change_date_marker(fig6, cd)
        st.plotly_chart(fig6, use_container_width=True)

# ================================================
# TAB 2: DPTL METRICS
# ================================================
with tab2:
    st.write("DPTL Metrics content goes here...")
