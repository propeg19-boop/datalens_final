"""
pages/2_Overview.py
Step 2: Dataset overview, structure, and user-overrideable column classification.
"""
import streamlit as st
import pandas as pd

from datalens.pipeline import DataLensPipeline
from ui.components import inject_custom_css, render_hero, render_type_badge, render_metric_card

st.set_page_config(page_title="2. Overview | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state or st.session_state.pipeline.raw_df is None:
    st.warning("No dataset loaded yet. Please upload a file in **1_Upload** first.")
    st.stop()

pipeline: DataLensPipeline = st.session_state.pipeline
df = pipeline.cleaned_df if pipeline.cleaned_df is not None else pipeline.raw_df

st.markdown(
    render_hero(
        "Step 2: Dataset Overview & Schema Verification",
        "Inspect row volume, memory footprint, and behavior-driven column classifications. You can override any detected type.",
    ),
    unsafe_allow_html=True,
)

# High Level Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(render_metric_card("Row Count", f"{len(df):,}", "Active records", "rows"), unsafe_allow_html=True)
with col2:
    st.markdown(render_metric_card("Column Count", len(df.columns), "Features & IDs", "columns"), unsafe_allow_html=True)
with col3:
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.markdown(render_metric_card("Memory Usage", f"{mem_mb:.2f} MB", "In-memory RAM", "database"), unsafe_allow_html=True)
with col4:
    total_cells = len(df) * len(df.columns)
    st.markdown(render_metric_card("Total Cells", f"{total_cells:,}", "Data points", "check"), unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Detected Column Classifications (Editable)")
st.caption("The behavioral engine classifies columns based on value patterns, not column names. If any type is wrong, change it below.")

type_options = ["numeric", "categorical", "datetime", "id", "text", "time"]

col_summary = []
for col in df.columns:
    c_type = pipeline.column_types.get(col, "text")
    nunique = df[col].nunique()
    nulls = int(df[col].isna().sum())
    col_summary.append({
        "Column": col,
        "Classified Type": c_type,
        "Unique Values": nunique,
        "Null Count": nulls,
        "Null %": f"{(nulls / len(df)) * 100:.1f}%",
        "Sample Value": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A",
    })

st.dataframe(pd.DataFrame(col_summary), use_container_width=True)

# User Type Override Form
with st.expander("✏️ Override Column Classifications"):
    col_to_edit = st.selectbox("Select Column to Reclassify:", list(df.columns))
    current_type = pipeline.column_types.get(col_to_edit, "text")
    default_idx = type_options.index(current_type) if current_type in type_options else 0
    new_type = st.selectbox("Select New Column Type:", type_options, index=default_idx)

    if st.button("Apply Reclassification"):
        pipeline.override_column_type(col_to_edit, new_type)
        st.success(f"Column '{col_to_edit}' reclassified to '{new_type}'. Health audit refreshed!")
        st.rerun()

st.markdown("---")
st.markdown("### Data Sample (First 25 Rows)")
st.dataframe(df.head(25), use_container_width=True)
