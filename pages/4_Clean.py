"""
pages/4_Clean.py
Step 4: Cleaning Assistant - Explainable recommendations, user approvals, and before/after verification.
"""
import streamlit as st

from datalens.pipeline import DataLensPipeline
from ui.components import (
    inject_custom_css,
    render_hero,
    render_recommendation_card,
    render_before_after_diff,
)

st.set_page_config(page_title="4. Clean | DataLens", layout="wide")
inject_custom_css()

if "pipeline" not in st.session_state or st.session_state.pipeline.raw_df is None:
    st.warning("No dataset loaded yet. Please upload a file in **1_Upload** first.")
    st.stop()

pipeline: DataLensPipeline = st.session_state.pipeline
recs = pipeline.recommendations

st.markdown(
    render_hero(
        "Step 4: Cleaning Assistant (Human-in-the-Loop)",
        "Every fix carries a plain-English reason. Nothing is modified silently. Approve recommendations or provide manual inputs below.",
    ),
    unsafe_allow_html=True,
)

# Reset Button in Top Toolbar
col_tb1, col_tb2 = st.columns([4, 1])
with col_tb2:
    if st.button("↺ Reset to Original", help="Discard all cleaning changes and revert to raw uploaded data"):
        pipeline.reset_to_raw()
        st.success("Dataset reset to original raw state!")
        st.rerun()

missing_recs = recs.get("missing_recommendations", [])
dup_rec = recs.get("duplicate_recommendation")
outlier_recs = recs.get("outlier_recommendations", [])

has_any_recs = bool(missing_recs or dup_rec or outlier_recs)

if not has_any_recs:
    st.success("🎉 Outstanding data quality! No missingness or duplicate defects were detected.")
    st.info("Head to **5_Insights** to discover statistical patterns.")
    st.stop()

# Approval Form
with st.form("cleaning_approval_form"):
    st.markdown("### 1. Missing Value Recommendations")
    approved_actions = []
    manual_inputs = {}

    if missing_recs:
        for idx, rec in enumerate(missing_recs):
            st.markdown(render_recommendation_card(rec), unsafe_allow_html=True)
            col = rec["column"]
            action = rec["action"]
            is_safe = rec["is_safe_auto"]

            col_chk, col_in = st.columns([2, 3])
            with col_chk:
                approve = st.checkbox(
                    f"Approve fix for '{col}'",
                    value=is_safe,
                    key=f"chk_miss_{idx}",
                )
                if approve:
                    approved_actions.append(rec)

            with col_in:
                if action == "manual":
                    val = st.text_input(
                        f"Manual value for '{col}':",
                        value="Unknown" if rec["column_type"] == "categorical" else "",
                        key=f"input_manual_{idx}",
                        help="Enter replacement value or leave blank to skip filling.",
                    )
                    if val.strip():
                        manual_inputs[col] = val
    else:
        st.write("No missing values detected.")

    st.markdown("---")
    st.markdown("### 2. Duplicate Rows Recommendation")
    if dup_rec:
        st.markdown(render_recommendation_card(dup_rec), unsafe_allow_html=True)
        approve_dup = st.checkbox(
            "Approve: Drop exact duplicate rows",
            value=dup_rec["is_safe_auto"],
            key="chk_dup_approve",
        )
        if approve_dup:
            approved_actions.append(dup_rec)
    else:
        st.write("No exact duplicate rows found.")

    st.markdown("---")
    st.markdown("### 3. Outlier Recommendations (Review Only)")
    if outlier_recs:
        st.caption("Outliers are never auto-deleted. Review below to inform downstream modeling.")
        for o_rec in outlier_recs:
            st.markdown(render_recommendation_card(o_rec), unsafe_allow_html=True)
    else:
        st.write("No outliers detected.")

    st.write("")
    submitted = st.form_submit_button("⚡ Apply Approved Transformations", type="primary")

if submitted:
    with st.spinner("Applying approved transformations and running before/after verification..."):
        diff = pipeline.apply_cleaning_actions(approved_actions, manual_inputs)
        st.success("Transformations applied successfully!")
        st.rerun()

# If verification exists, render the before/after comparison
if pipeline.verification:
    st.markdown("---")
    st.markdown("### Before / After Verification Proof")
    st.markdown(render_before_after_diff(pipeline.verification), unsafe_allow_html=True)

    if pipeline.applied_log:
        st.markdown("#### Applied Transformation Log")
        log_records = [
            {
                "Target": item.get("target"),
                "Action": item.get("action"),
                "Value Used": item.get("value_used"),
                "Rows Affected": item.get("rows_affected"),
                "Status": item.get("status"),
                "Note": item.get("note"),
            }
            for item in pipeline.applied_log
        ]
        st.dataframe(log_records, use_container_width=True)

st.info("Next: Head over to **5_Insights** to explore correlations, trends, and category breakdowns.")
