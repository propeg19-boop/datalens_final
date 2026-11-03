"""
pages/7_Export.py
Step 7: Export cleaned dataset and download comprehensive analysis report.
"""
import streamlit as st

from datalens.pipeline import DataLensPipeline
from ui.components import inject_custom_css, render_hero, render_metric_card

st.set_page_config(page_title="7. Export | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state or st.session_state.pipeline.raw_df is None:
    st.warning("No dataset loaded yet. Please upload a file in **1_Upload** first.")
    st.stop()

pipeline: DataLensPipeline = st.session_state.pipeline
df_clean = pipeline.cleaned_df if pipeline.cleaned_df is not None else pipeline.raw_df

st.markdown(
    render_hero(
        "Step 7: Export Cleaned Data & Executive Report",
        "Download your verified, cleaned dataset and download comprehensive executive reports in Markdown or HTML.",
    ),
    unsafe_allow_html=True,
)

# Export Summary Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        render_metric_card("Export Rows", f"{len(df_clean):,}", "Active verified rows", "rows"),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        render_metric_card("Final Quality Score", f"{pipeline.quality_report.get('quality_score')}/100", "Post-cleaning rating", "shield"),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        render_metric_card("Applied Transformations", len(pipeline.applied_log), "Recorded in audit log", "check"),
        unsafe_allow_html=True,
    )

st.markdown("---")

col_csv, col_md, col_html = st.columns(3)

with col_csv:
    st.markdown("### 📥 Cleaned Dataset")
    st.write("Export the cleaned, imputed dataset as a standard CSV file for use in downstream pipelines or spreadsheets.")
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cleaned CSV",
        data=csv_bytes,
        file_name=f"datalens_cleaned_{pipeline.filename.replace('.csv', '')}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

with col_md:
    st.markdown("### 📄 Markdown Report")
    st.write("Download an explainable documentation report summarizing data health, applied fixes, and key insights.")
    md_rep = pipeline.get_summary_report(format="markdown")
    st.download_button(
        label="Download Report (.md)",
        data=md_rep.encode("utf-8"),
        file_name=f"datalens_report_{pipeline.filename.replace('.csv', '')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

with col_html:
    st.markdown("### 🌐 Standalone HTML Report")
    st.write("Download a publication-ready HTML executive report formatted with modern styles for presentation.")
    html_rep = pipeline.get_summary_report(format="html")
    st.download_button(
        label="Download Report (.html)",
        data=html_rep.encode("utf-8"),
        file_name=f"datalens_report_{pipeline.filename.replace('.csv', '')}.html",
        mime="text/html",
        use_container_width=True,
    )

st.markdown("---")
st.markdown("### Live Preview of Summary Report")
with st.expander("👁️ View Markdown Summary Content", expanded=True):
    st.markdown(md_rep)
