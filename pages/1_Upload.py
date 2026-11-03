"""
pages/1_Upload.py
Step 1: Upload CSV / Excel or load sample dataset.
"""
from pathlib import Path
import streamlit as st

from datalens.pipeline import DataLensPipeline
from datalens.data_loader import get_excel_sheet_names
from ui.components import inject_custom_css, render_hero

st.set_page_config(page_title="1. Upload | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DataLensPipeline()
pipeline: DataLensPipeline = st.session_state.pipeline

st.markdown(
    render_hero(
        "Step 1: Ingest Dataset",
        "Upload a CSV or Excel file (up to 100MB). Formats and sheet selections are handled automatically.",
    ),
    unsafe_allow_html=True,
)

tab_upload, tab_sample = st.tabs(["📁 Upload Your File", "☕ Sample Dataset"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Maximum upload size: 100MB. Supported formats: .csv, .xlsx, .xls",
    )

    if uploaded_file is not None:
        filename = uploaded_file.name
        is_excel = filename.lower().endswith((".xlsx", ".xls"))
        sheet_choice = 0

        if is_excel:
            sheets, err = get_excel_sheet_names(uploaded_file, filename=filename)
            if not err and len(sheets) > 1:
                sheet_choice = st.selectbox("Select Excel Sheet to Load:", sheets)

        if st.button("Load and Inspect File", type="primary"):
            with st.spinner("Parsing and classifying columns..."):
                success, err = pipeline.load_dataset(uploaded_file, filename=filename)
                if success:
                    st.success(f"Loaded '{filename}' successfully ({len(pipeline.raw_df):,} rows, {len(pipeline.raw_df.columns)} columns)!")
                    st.rerun()
                else:
                    st.error(f"Error loading file: {err}")

with tab_sample:
    st.markdown("Try DataLens with the official Coffee Shop Sales reference dataset.")
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "coffeeshopsales_sample.csv"

    if sample_file.exists():
        if st.button("Load Reference Coffee Shop Dataset", key="btn_sample"):
            with st.spinner("Loading coffee shop sales dataset..."):
                success, err = pipeline.load_dataset(sample_file, filename="coffeeshopsales_sample.csv")
                if success:
                    st.success("Loaded reference coffee shop sales dataset! Proceed to Step 2.")
                    st.rerun()
                else:
                    st.error(f"Error: {err}")
    else:
        st.warning("Sample dataset not found on disk.")

# If dataset is loaded, show preview
if pipeline.raw_df is not None:
    st.markdown("---")
    st.markdown(f"#### Active Dataset: `{pipeline.filename}`")
    st.dataframe(pipeline.raw_df.head(10), use_container_width=True)
    st.info("Dataset loaded. Head over to **2_Overview** to inspect and verify column classifications.")
