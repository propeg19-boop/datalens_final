# DataLens 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()

> **Dataset-Agnostic Data Health, Explainable Cleaning & Pattern Discovery Web App**  
> Upload any CSV or Excel file to assess data quality, clean with explicit approval, uncover visual patterns with plain-English captions, and benchmark baseline ML models.

---

## 🌟 Key Features

1. **Dataset-Agnostic Column Classification**
   - Pure data-driven classification (`id`, `datetime`, `time`, `numeric`, `categorical`, `text`).
   - Prioritizes uniqueness and behavioral patterns over column names (e.g., distinguishing repeating foreign keys from unique IDs).
2. **Data Detective (Health Audit & Scoring)**
   - Single explainable 0–100 Data Quality Score with transparent penalty deductions.
   - Audits missingness, duplicate rows, and IQR outliers with **skewness detection** to avoid over-penalizing legitimate premium data.
3. **Cleaning Assistant (Human-in-the-Loop)**
   - Actionable recommendations with plain-English reasoning for every fix.
   - Authentic KNN imputation (`sklearn.impute.KNNImputer`), median/mode imputation, and duplicate removal.
   - Side-by-side before/after verification proving defect resolution without mutating original raw data.
4. **Pattern Hunter & AI Narrator**
   - Computes pairwise Pearson correlations ($|r| \ge 0.5$), resampled linear time trends with polynomial slopes, and category breakdowns.
   - Generates plain-English captions with a 100% deterministic template fallback (no API key required).
5. **Baseline Machine Learning**
   - Automated regression and classification pipeline with data leakage detection.
   - Evaluates Linear/Logistic and Random Forest models against naive **Dummy Baselines** (mean / majority class).
6. **Executive Export**
   - One-click export of cleaned CSV datasets, along with comprehensive Markdown and publication-ready HTML reports.

---

## 🎨 UI & UX Design System

Engineered with modern tokens adhering to `ui-styling` and `ui-ux-pro-max`:
- **Theme:** Data-Dense Dashboard (Enterprise Analytics).
- **Palette:** Slate Blue (`#1E40AF`), Electric Blue (`#3B82F6`), Amber (`#D97706`), Light Slate Surface (`#F8FAFC`).
- **Typography:** Inter (Headings/Body) + Fira Code (Metrics/Data tokens).
- **Accessibility:** High-contrast text ($\ge 4.5:1$), focus rings, inline SVGs (no emojis as structural icons), and reduced-motion support.

---

## 📁 Repository Structure

```
datalens/
├── .streamlit/
│   ├── config.toml               # Streamlit theme & upload configuration
│   └── secrets.toml.example      # Template for optional LLM API keys
├── datalens/                     # Pure Python core package (no Streamlit imports)
│   ├── __init__.py
│   ├── config.py                 # Central thresholds, weights, and tokens
│   ├── data_loader.py            # M1: Multi-format file & buffer ingestion
│   ├── column_types.py           # M2: Behavioral column classification
│   ├── data_quality.py           # M3: Data Detective health & quality scoring
│   ├── cleaning_assistant.py     # M4: Explainable recommendation engine
│   ├── verify_cleaning.py        # M5: KNN & rule imputation + before/after diff
│   ├── pattern_hunter.py         # M6: Correlations, trends & breakdowns
│   ├── visualizer.py             # M7: Interactive Plotly chart engine
│   ├── ai_narrator.py            # M8: Plain-English captions + template fallback
│   ├── model_trainer.py          # M9: Baseline ML models vs dummy baselines
│   ├── report.py                 # M10: Markdown & HTML report generators
│   └── pipeline.py               # M10: Central orchestration pipeline
├── ui/                           # UI component library & styling
│   ├── styles.css                # CSS custom properties & component styling
│   └── components.py             # Reusable HTML & SVG card components
├── pages/                        # Multi-page Streamlit application
│   ├── 1_Upload.py               # Step 1: Ingestion & sheet selection
│   ├── 2_Overview.py             # Step 2: Schema overview & editable types
│   ├── 3_Data_Health.py          # Step 3: Health score & defect diagnostics
│   ├── 4_Clean.py                # Step 4: Explainable cleaning approvals
│   ├── 5_Insights.py             # Step 5: Visual patterns & captions
│   ├── 6_Predict.py              # Step 6: ML benchmarks vs naive guessing
│   └── 7_Export.py               # Step 7: Cleaned CSV & summary reports
├── data/
│   └── sample/
│       ├── coffeeshopsales_sample.csv
│       ├── coffeeshopsales_sample.xlsx
│       └── generate_reference_data.py
├── tests/                        # Comprehensive pytest test suite (37 passing tests)
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_column_types.py
│   ├── test_data_quality.py
│   ├── test_cleaning_assistant.py
│   ├── test_verify_cleaning.py
│   ├── test_pattern_hunter.py
│   ├── test_visualizer.py
│   ├── test_ai_narrator.py
│   ├── test_model_trainer.py
│   └── test_pipeline.py
├── app.py                        # Streamlit main entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/propeg19-boop/datalens_final.git
cd datalens_final

# Install dependencies
pip install -r requirements.txt

# Run the full test suite (37 tests)
pytest tests/ -v

# Launch the Streamlit web application
streamlit run app.py
```
