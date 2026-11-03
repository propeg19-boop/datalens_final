"""
pages/3_Data_Health.py
Step 3: Data Detective health check, scoring, and defect diagnostics.
"""
import streamlit as st
import pandas as pd

from datalens.pipeline import DataLensPipeline
from datalens.visualizer import plot_missing_overview, plot_distribution
from ui.components import inject_custom_css, render_hero, render_quality_gauge, render_metric_card

st.set_page_config(page_title="3. Data Health | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state or st.session_state.pipeline.raw_df is None:
    st.warning("No dataset loaded yet. Please upload a file in **1_Upload** first.")
    st.stop()

pipeline: DataLensPipeline = st.session_state.pipeline
df = pipeline.cleaned_df if pipeline.cleaned_df is not None else pipeline.raw_df
report = pipeline.quality_report

st.markdown(
    render_hero(
        "Step 3: Data Detective (Health Audit)",
        "Audit dataset completeness, uniqueness, and outlier anomalies. The explainable quality score ranges from 0 to 100.",
    ),
    unsafe_allow_html=True,
)

# Render Large Quality Gauge
st.markdown(
    render_quality_gauge(report.get("quality_score", 100.0), report.get("score_breakdown")),
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        render_metric_card("Missing Cells", f"{report.get('total_missing_cells', 0):,}", f"Avg {report.get('missing_pct_avg', 0)}% per column", "alert"),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        render_metric_card("Duplicate Rows", f"{report.get('duplicate_rows', 0):,}", f"{report.get('duplicate_pct', 0)}% of dataset", "rows"),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        render_metric_card("IQR Outliers", f"{report.get('total_outlier_count', 0):,}", "Across numeric features", "sparkles"),
        unsafe_allow_html=True,
    )

st.markdown("---")

# Missing values section
st.markdown("### Missingness Overview")
col_chart, col_table = st.columns([3, 2])
with col_chart:
    st.plotly_chart(plot_missing_overview(report), use_container_width=True)
with col_table:
    missing_data = [
        {"Column": col, "Missing Rows": info["count"], "Missing %": f"{info['pct']}%", "Type": info["column_type"]}
        for col, info in report.get("missing_by_column", {}).items()
        if info["count"] > 0
    ]
    if missing_data:
        st.dataframe(pd.DataFrame(missing_data), use_container_width=True)
    else:
        st.success("🎉 Complete dataset: Zero missing values detected across all columns!")

st.markdown("---")

# Outliers Section
st.markdown("### Outlier & Skewness Analysis (Numeric Columns)")
outliers = report.get("outliers_by_column", {})

if outliers:
    outlier_rows = []
    for col, o in outliers.items():
        outlier_rows.append({
            "Column": col,
            "Outlier Count": o["count"],
            "Outlier %": f"{o['pct']}%",
            "IQR Range": f"[{o['lower_bound']}, {o['upper_bound']}]",
            "Skewness": o["skewness"],
            "Skewed?": "Yes (Likely Valid Entity)" if o["is_skewed"] else "No (Check Entry)",
            "Assessment Note": o["note"],
        })
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

    # Interactive Distribution Inspector
    with st.expander("📊 Inspect Column Distribution & Outliers"):
        inspect_col = st.selectbox("Pick Numeric Column:", list(outliers.keys()))
        if inspect_col:
            st.plotly_chart(plot_distribution(df, inspect_col, "numeric"), use_container_width=True)
else:
    st.info("No numeric columns available for outlier analysis.")

st.info("Next: Head over to **4_Clean** to review and approve explainable cleaning fixes.")
