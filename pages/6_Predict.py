"""
pages/6_Predict.py
Step 6: Baseline Machine Learning - Model training vs naive dummy baselines.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from datalens.pipeline import DataLensPipeline
from ui.components import inject_custom_css, render_hero, render_metric_card
from datalens.config import THEME

st.set_page_config(page_title="6. Predict | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state or st.session_state.pipeline.raw_df is None:
    st.warning("No dataset loaded yet. Please upload a file in **1_Upload** first.")
    st.stop()

pipeline: DataLensPipeline = st.session_state.pipeline
df = pipeline.cleaned_df if pipeline.cleaned_df is not None else pipeline.raw_df

st.markdown(
    render_hero(
        "Step 6: Baseline Machine Learning Benchmark",
        "Select any numeric or categorical column to predict. DataLens trains baseline ML models and compares them honestly against naive guessing baselines.",
    ),
    unsafe_allow_html=True,
)

# Eligible Target Columns
eligible_targets = [
    col for col, ctype in pipeline.column_types.items()
    if ctype in {"numeric", "categorical"} and col in df.columns
]

if not eligible_targets:
    st.error("No eligible numeric or categorical target columns found.")
    st.stop()

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    target_col = st.selectbox("Select Target Column to Predict:", eligible_targets)
    target_type = pipeline.column_types.get(target_col, "numeric")
    inferred_task = "Regression (Continuous Target)" if target_type == "numeric" else "Classification (Discrete Classes)"
    st.caption(f"Inferred Task: **{inferred_task}**")

with col_btn:
    st.write("")
    st.write("")
    train_clicked = st.button("🚀 Train Baseline Models", type="primary")

if train_clicked:
    with st.spinner("Preprocessing, splitting 80/20, and training models vs naive baselines..."):
        res = pipeline.train_model(target_col)
        if "error" in res:
            st.error(res["error"])
        else:
            st.success("Training complete!")

if pipeline.model_results and "metrics_table" in pipeline.model_results:
    res = pipeline.model_results
    verdict = res.get("verdict", {})

    st.markdown("---")
    st.markdown("### Model Benchmark Verdict")

    # Verdict Card
    badge = verdict.get("badge", "Benchmark Complete")
    text = verdict.get("text", "")
    badge_color = "var(--dl-success)" if verdict.get("beats_baseline") else "var(--dl-warning)"

    st.markdown(
        f"""
        <div class="dl-rec-card" style="border-left: 4px solid {badge_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span class="dl-metric-label">Assessment</span>
                <span class="dl-badge" style="background: var(--dl-primary-subtle); color: var(--dl-primary); font-size: 13px;">
                    {badge}
                </span>
            </div>
            <div style="font-size: 15px; color: var(--dl-text); line-height: 1.5;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if res.get("leakage_warnings"):
        for lw in res["leakage_warnings"]:
            st.warning(f"⚠️ **Potential Data Leakage**: {lw}")

    st.markdown("### Performance Benchmark vs Naive Baseline")
    metrics_df = pd.DataFrame(res["metrics_table"])
    st.dataframe(metrics_df, use_container_width=True)

    # Feature Importance
    importances = res.get("feature_importances", [])
    if importances:
        st.markdown("### Top Predictive Features (Random Forest)")
        imp_df = pd.DataFrame(importances)
        fig = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            color_discrete_sequence=[THEME["primary"]],
        )
        fig.update_yaxes(categoryorder="total ascending", title="")
        fig.update_xaxes(title="Feature Importance")
        fig.update_layout(
            paper_bgcolor=THEME["surface"],
            plot_bgcolor="#F1F5F9",
            margin={"l": 40, "r": 30, "t": 20, "b": 30},
            font={"family": THEME["font_family"]},
        )
        st.plotly_chart(fig, use_container_width=True)

st.info("Next: Head over to **7_Export** to download the cleaned dataset and full analysis report.")
