"""
datalens/visualizer.py
Interactive Plotly visualizations for DataLens.
Adheres to design tokens from config.THEME (Data-Dense Dashboard style).
Implements performance safeguards (sampling large datasets, top-N truncation).
Always returns plotly.graph_objects.Figure instances without calling .show().
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datalens.config import (
    THEME,
    MAX_SCATTER_SAMPLE_SIZE,
    DEFAULT_TOP_N_CATEGORIES,
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling, typography, and palette to a Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"family": THEME["font_family"], "size": 16, "color": THEME["foreground"]},
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor=THEME["surface"],
        plot_bgcolor="#F1F5F9",
        font={"family": THEME["font_family"], "color": THEME["foreground"], "size": 12},
        margin={"l": 40, "r": 30, "t": 50, "b": 40},
        hoverlabel={"bgcolor": THEME["foreground"], "font_color": "#FFFFFF", "font_family": THEME["font_family"]},
        xaxis={
            "gridcolor": "#E2E8F0",
            "zerolinecolor": "#CBD5E1",
            "tickfont": {"size": 11, "color": THEME["muted"]},
            "title_font": {"size": 12, "color": THEME["foreground"]},
        },
        yaxis={
            "gridcolor": "#E2E8F0",
            "zerolinecolor": "#CBD5E1",
            "tickfont": {"size": 11, "color": THEME["muted"]},
            "title_font": {"size": 12, "color": THEME["foreground"]},
        },
    )
    return fig


def plot_missing_overview(quality_report: Dict[str, Any]) -> go.Figure:
    """Render horizontal bar chart of missing values across all columns."""
    missing_by_col = quality_report.get("missing_by_column", {})
    if not missing_by_col:
        fig = go.Figure()
        fig.add_annotation(text="No missing value data available", showarrow=False)
        return _apply_theme(fig, "Missing Data Overview")

    data = [
        {"column": col, "count": info["count"], "pct": info["pct"]}
        for col, info in missing_by_col.items()
    ]
    df_missing = pd.DataFrame(data).sort_values(by="pct", ascending=True)

    # Color code bars: green if 0%, amber if <= 5%, red if > 5%
    colors = [
        THEME["success"] if p == 0 else (THEME["warning"] if p <= 5 else THEME["danger"])
        for p in df_missing["pct"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_missing["pct"],
            y=df_missing["column"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}% (%{customdata} rows)<extra></extra>",
            customdata=df_missing["count"],
        )
    )
    fig.update_xaxes(title="Missing Percentage (%)", range=[0, max(10, df_missing["pct"].max() * 1.15)])
    fig.update_yaxes(title="")
    return _apply_theme(fig, "Missing Values by Column")


def plot_distribution(df: pd.DataFrame, column: str, column_type: str) -> go.Figure:
    """Render distribution plot based on column type."""
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' not found", showarrow=False)
        return _apply_theme(fig, "Distribution")

    series = df[column].dropna()
    if len(series) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Column contains only null values", showarrow=False)
        return _apply_theme(fig, f"Distribution: {column}")

    if column_type == "numeric":
        s_num = pd.to_numeric(series, errors="coerce").dropna()
        fig = px.histogram(
            s_num,
            x=s_num.values,
            marginal="box",
            nbins=30,
            color_discrete_sequence=[THEME["primary"]],
        )
        fig.update_xaxes(title=column)
        fig.update_yaxes(title="Count")
        return _apply_theme(fig, f"Numeric Distribution: {column}")

    else:
        # Categorical or text distribution (top 15)
        counts = series.value_counts().head(15).reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(
            counts,
            x="count",
            y=column,
            orientation="h",
            color_discrete_sequence=[THEME["secondary"]],
        )
        fig.update_yaxes(categoryorder="total ascending", title="")
        fig.update_xaxes(title="Count")
        return _apply_theme(fig, f"Category Frequencies: {column}")


def plot_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    sample_size: int = MAX_SCATTER_SAMPLE_SIZE,
) -> go.Figure:
    """Render scatter plot with trendline between two numeric features, sampled for speed."""
    sub = df[[col1, col2]].dropna()
    if len(sub) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No complete rows for scatter plot", showarrow=False)
        return _apply_theme(fig, f"{col1} vs {col2}")

    # Sample if large
    is_sampled = len(sub) > sample_size
    if is_sampled:
        sub = sub.sample(n=sample_size, random_state=42)

    try:
        fig = px.scatter(
            sub,
            x=col1,
            y=col2,
            trendline="ols",
            color_discrete_sequence=[THEME["primary"]],
            opacity=0.65,
        )
        # Style trendline
        fig.data[1].line.color = THEME["accent"]
        fig.data[1].line.width = 2.5
    except Exception:
        # If statsmodels fails, fallback to standard scatter
        fig = px.scatter(
            sub,
            x=col1,
            y=col2,
            color_discrete_sequence=[THEME["primary"]],
            opacity=0.65,
        )

    title = f"Correlation: {col1} vs {col2}"
    if is_sampled:
        title += f" (Sampled {sample_size:,} points)"

    return _apply_theme(fig, title)


def plot_time_trend(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    period_type: str = "daily",
) -> go.Figure:
    """Render aggregated time trend line chart."""
    sub = df[[date_col, metric_col]].dropna().copy()
    sub["_dt"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=["_dt"])

    if len(sub) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient time series data", showarrow=False)
        return _apply_theme(fig, f"{metric_col} Trend")

    freq = "D" if period_type == "daily" else ("W" if period_type == "weekly" else "M")
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    resampled = sub.set_index("_dt")[metric_col].resample(freq).sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=resampled["_dt"],
            y=resampled[metric_col],
            mode="lines+markers",
            line={"color": THEME["primary"], "width": 2.5},
            marker={"size": 5, "color": THEME["secondary"]},
            name="Observed Total",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Sum: %{y:,.2f}<extra></extra>",
        )
    )

    # Add linear trendline
    if len(resampled) >= 3:
        x_numeric = np.arange(len(resampled))
        y = resampled[metric_col].values
        slope, intercept = np.polyfit(x_numeric, y, 1)
        trend_y = slope * x_numeric + intercept
        fig.add_trace(
            go.Scatter(
                x=resampled["_dt"],
                y=trend_y,
                mode="lines",
                line={"color": THEME["accent"], "dash": "dot", "width": 2},
                name="Trendline (Linear)",
            )
        )

    return _apply_theme(fig, f"{period_type.capitalize()} Trend: {metric_col}")


def plot_category_breakdown(
    df: pd.DataFrame,
    cat_col: str,
    metric_col: str,
    top_n: int = DEFAULT_TOP_N_CATEGORIES,
) -> go.Figure:
    """Render horizontal bar chart for categorical aggregations with Top-N limit."""
    sub = df[[cat_col, metric_col]].dropna().copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    grouped = sub.groupby(cat_col)[metric_col].sum().reset_index()
    grouped = grouped.sort_values(by=metric_col, ascending=False).head(top_n)

    fig = px.bar(
        grouped,
        x=metric_col,
        y=cat_col,
        orientation="h",
        color_discrete_sequence=[THEME["secondary"]],
    )
    fig.update_yaxes(categoryorder="total ascending", title="")
    fig.update_xaxes(title=f"Total {metric_col}")
    return _apply_theme(fig, f"Top {top_n} {cat_col} by {metric_col}")
