"""
pages/5_Insights.py
Step 5: Statistical insights, Plotly charts, and plain-English captions.
"""
import streamlit as st

from datalens.pipeline import DataLensPipeline
from datalens.visualizer import plot_correlation, plot_time_trend, plot_category_breakdown
from ui.components import inject_custom_css, render_hero

st.set_page_config(page_title="5. Insights | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state or st.session_state.pipeline.raw_df is None:
    st.warning("No dataset loaded yet. Please upload a file in **1_Upload** first.")
    st.stop()

pipeline: DataLensPipeline = st.session_state.pipeline
df = pipeline.cleaned_df if pipeline.cleaned_df is not None else pipeline.raw_df

st.markdown(
    render_hero(
        "Step 5: Pattern Hunter & AI Narrator",
        "Explore statistical correlations, temporal movements, and category segmentations accompanied by plain-English captions.",
    ),
    unsafe_allow_html=True,
)

# AI Narrator Options
col_opt1, col_opt2 = st.columns([3, 1])
with col_opt2:
    use_ai = st.toggle("Enable Cloud LLM Captions", value=False, help="Uses Anthropic / Gemini API if key is present; otherwise uses deterministic factual template.")

if not pipeline.patterns:
    with st.spinner("Analyzing patterns across dimensions..."):
        pipeline.discover_patterns(use_ai=use_ai)

patterns = pipeline.patterns
corrs = patterns.get("correlations", [])
trends = patterns.get("trends", [])
breakdowns = patterns.get("breakdowns", [])

tab_corr, tab_trends, tab_breakdown = st.tabs(["🔗 Feature Correlations", "📈 Temporal Trends", "📊 Category Breakdowns"])

with tab_corr:
    if corrs:
        st.markdown("### Significant Pairwise Correlations (|r| ≥ 0.5)")
        for c in corrs:
            col_chart, col_cap = st.columns([3, 2])
            with col_chart:
                fig = plot_correlation(df, c["col1"], c["col2"])
                st.plotly_chart(fig, use_container_width=True)
            with col_cap:
                st.markdown(
                    f"""
                    <div class="dl-rec-card" style="margin-top: 50px;">
                        <div class="dl-metric-label">Statistical Correlation</div>
                        <div style="font-size: 20px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">
                            r = {c['r']} ({c['strength'].capitalize()})
                        </div>
                        <div class="dl-rec-reason">
                            {c.get('caption', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No pairwise correlations exceeding |r| ≥ 0.5 were discovered in numeric columns.")

with tab_trends:
    if trends:
        st.markdown("### Time Series Trajectories")
        for t in trends:
            col_chart, col_cap = st.columns([3, 2])
            with col_chart:
                fig = plot_time_trend(df, t["date_col"], t["metric_col"], t["period_type"])
                st.plotly_chart(fig, use_container_width=True)
            with col_cap:
                st.markdown(
                    f"""
                    <div class="dl-rec-card" style="margin-top: 50px;">
                        <div class="dl-metric-label">{t['period_type'].capitalize()} Trend Direction</div>
                        <div style="font-size: 20px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">
                            {t['direction'].capitalize()} ({t['pct_change_per_step']}% / period)
                        </div>
                        <div class="dl-rec-reason">
                            {t.get('caption', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No temporal trends available (requires at least one valid datetime column and numeric metrics).")

with tab_breakdown:
    if breakdowns:
        st.markdown("### Category Distributions & Aggregations")
        for b in breakdowns[:5]:
            col_chart, col_cap = st.columns([3, 2])
            with col_chart:
                fig = plot_category_breakdown(df, b["category_col"], b["metric_col"])
                st.plotly_chart(fig, use_container_width=True)
            with col_cap:
                st.markdown(
                    f"""
                    <div class="dl-rec-card" style="margin-top: 50px;">
                        <div class="dl-metric-label">Leading Segment</div>
                        <div style="font-size: 20px; font-weight: 700; color: var(--dl-primary); margin: 6px 0;">
                            {b['top_category']} ({b['top_category_share_pct']}%)
                        </div>
                        <div class="dl-rec-reason">
                            {b.get('caption', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No category breakdowns available (requires categorical columns with ≤ 30 classes).")

st.info("Next: Head over to **6_Predict** to train baseline machine learning models.")
