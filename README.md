# DataLens

> **Dataset-Agnostic Data Health, Explainable Cleaning & Pattern Discovery Web App**

DataLens empowers non-technical users and analysts to upload any CSV or Excel file, understand its quality score, review explainable cleaning recommendations with human-in-the-loop approval, and discover underlying patterns.

---

## Architecture & Principles

1. **Column Type Drives Everything:** Behavior-driven classification (`id`, `datetime`, `time`, `numeric`, `categorical`, `text`).
2. **Deterministic Data Changes; LLMs Only Explain:** Data transformations are 100% rule-based and reproducible.
3. **No Silent Fixes:** Every recommendation presents plain-English reasoning with transparent user approval.
4. **Clean UI & UX:** Engineered with design tokens from `ui-styling` and `ui-ux-pro-max` (Data-Dense Dashboard theme, Inter & Fira Code typography, accessible SVG icons).

---

## Project Structure

```
datalens/
├── .streamlit/
│   ├── config.toml               # Streamlit theme & upload configuration
│   └── secrets.toml.example      # API key template
├── datalens/                     # Pure Python core package (no Streamlit imports)
│   ├── __init__.py
│   ├── config.py                 # Central thresholds, weights, and tokens
│   ├── data_loader.py            # M1: Multi-format file & buffer ingestion
│   ├── column_types.py           # M2: Behavioral column classification
│   ├── data_quality.py           # M3: Data Detective health & quality scoring
│   ├── cleaning_assistant.py     # M4: Explainable recommendation engine
│   ├── verify_cleaning.py        # M5: Real KNN & rule imputation + before/after diff
│   └── pattern_hunter.py         # M6: Statistical correlations, trends & breakdowns
├── ui/                           # UI component library & styling
│   ├── styles.css                # CSS custom properties & component styling
│   └── components.py             # Reusable HTML & SVG card components
├── data/
│   └── sample/
│       ├── coffeeshopsales_sample.csv
│       ├── coffeeshopsales_sample.xlsx
│       └── generate_reference_data.py
├── tests/                        # Comprehensive pytest test suite (26 passing tests)
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_column_types.py
│   ├── test_data_quality.py
│   ├── test_cleaning_assistant.py
│   ├── test_verify_cleaning.py
│   └── test_pattern_hunter.py
├── requirements.txt
└── README.md
```

---

## Running the Test Suite

```bash
pip install -r requirements.txt
pytest tests/ -v
```
