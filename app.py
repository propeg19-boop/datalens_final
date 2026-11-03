"""
app.py
DataLens - Dataset-Agnostic Analytics, Quality Auditing & Pattern Discovery Web App.
Streamlit Main Entry Point.
"""
from pathlib import Path
import streamlit as st

from datalens.pipeline import DataLensPipeline
from ui.components import inject_custom_css, render_hero, render_metric_card

st.set_page_config(
    page_title="DataLens | Data Analytics & Quality Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern Data-Dense Dashboard styles
inject_custom_css()

# Initialize session state pipeline
if "pipeline" not in st.session_state:
    st.session_state.pipeline = DataLensPipeline()

pipeline: DataLensPipeline = st.session_state.pipeline

# Hero Header
st.markdown(
    render_hero(
        "DataLens: Explainable Data Analytics Platform",
        "Upload any CSV or Excel file to assess data health, approve explainable cleaning fixes, "
        "explore statistical patterns with plain-English captions, and train baseline prediction models.",
    ),
    unsafe_allow_html=True,
)

# Status Callout
has_data = pipeline.raw_df is not None

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        render_metric_card(
            "Dataset Status",
            pipeline.filename if has_data else "No File Loaded",
            subtext=f"{len(pipeline.raw_df):,} rows loaded" if has_data else "Upload a file in Step 1",
            icon="database",
        ),
        unsafe_allow_html=True,
    )
with col2:
    score = pipeline.quality_report.get("quality_score", "N/A")
    st.markdown(
        render_metric_card(
            "Data Health Score",
            f"{score}/100" if has_data else "Pending",
            subtext="Run Data Detective" if has_data else "Awaiting upload",
            icon="shield",
        ),
        unsafe_allow_html=True,
    )
with col3:
    n_features = len(pipeline.column_types) if has_data else 0
    st.markdown(
        render_metric_card(
            "Classified Columns",
            n_features if has_data else 0,
            subtext="Behavior-driven types",
            icon="columns",
        ),
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown("### How DataLens Works (The Journey)")

step_cols = st.columns(4)
with step_cols[0]:
    st.markdown(
        """
        <div class="dl-metric-card" style="min-height: 160px;">
            <div class="dl-metric-label">Phase 1</div>
            <div style="font-size: 16px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">1. Ingest & Classify</div>
            <p style="font-size: 13px; color: var(--dl-text-muted); margin: 0;">
                Load CSV/Excel files. Behavioral type classifier labels columns without name assumptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_cols[1]:
    st.markdown(
        """
        <div class="dl-metric-card" style="min-height: 160px;">
            <div class="dl-metric-label">Phase 2</div>
            <div style="font-size: 16px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">2. Audit & Clean</div>
            <p style="font-size: 13px; color: var(--dl-text-muted); margin: 0;">
                Data Detective scores health. Cleaning Assistant recommends KNN, median, or mode with human-in-the-loop review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_cols[2]:
    st.markdown(
        """
        <div class="dl-metric-card" style="min-height: 160px;">
            <div class="dl-metric-label">Phase 3</div>
            <div style="font-size: 16px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">3. Visual Insights</div>
            <p style="font-size: 13px; color: var(--dl-text-muted); margin: 0;">
                Discover correlations, temporal trends, and category breakdowns with plain-English captions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_cols[3]:
    st.markdown(
        """
        <div class="dl-metric-card" style="min-height: 160px;">
            <div class="dl-metric-label">Phase 4</div>
            <div style="font-size: 16px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">4. Baseline Predict</div>
            <p style="font-size: 13px; color: var(--dl-text-muted); margin: 0;">
                Automated ML benchmark. Compare Random Forest vs naive Dummy Baselines with leakage warnings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
if not has_data:
    st.info("👋 Get started by navigating to **1_Upload** in the sidebar or loading the bundled Coffee Shop reference dataset.")
    sample_file = Path(__file__).resolve().parent / "data" / "sample" / "coffeeshopsales_sample.csv"
    if sample_file.exists():
        if st.button("🚀 Load Bundled Coffee Shop Sales Dataset", type="primary"):
            success, err = pipeline.load_dataset(sample_file, filename="coffeeshopsales_sample.csv")
            if success:
                st.success("Successfully loaded reference dataset! Navigate to 2_Overview or 3_Data_Health.")
                st.rerun()
            else:
                st.error(f"Failed to load sample: {err}")
else:
    st.success(f"Active dataset: **{pipeline.filename}** ({len(pipeline.raw_df):,} rows). Use the sidebar to explore each page.")
