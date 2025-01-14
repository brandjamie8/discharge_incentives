import plotly.express as px
import plotly.graph_objects as go

# Define a custom color palette
CUSTOM_COLORS = {
    "site1": "#1f77b4",
    "site2": "#ff7f0e",
    "borough1": "#2ca02c",
    "borough2": "#d62728",
    "pathway1": "#9467bd",
    "pathway2": "#8c564b",
    # Add more colors as needed
}

def create_summary_charts(data1, data2, split_by):
    fig = go.Figure()

    # Group data by the selected column for splitting lines
    grouped_data = data1.groupby(split_by)

    for group, df in grouped_data:
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['admissions'], 
            mode='lines+markers',
            name=f"{group} Admissions",
            line=dict(color=CUSTOM_COLORS.get(group, "#000000"))  # Use default color if group not in palette
        ))

        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['discharges'], 
            mode='lines+markers',
            name=f"{group} Discharges",
            line=dict(color=CUSTOM_COLORS.get(group, "#000000"))
        ))

    fig.update_layout(
        title=f"Admissions and Discharges Split by {split_by.capitalize()}",
        xaxis_title="Date",
        yaxis_title="Count",
        legend_title=split_by.capitalize()
    )
    return fig

def create_comparison_chart(comparison_data):
    fig = px.bar(
        comparison_data,
        x="Metric",
        y=["Week 1", "Week 2"],
        barmode="group",
        color_discrete_sequence=list(CUSTOM_COLORS.values()),
        title="Comparison of Metrics Between Weeks",
    )
    return fig
