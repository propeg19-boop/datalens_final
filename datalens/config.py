"""
datalens/config.py
Central configuration for DataLens: thresholds, scoring weights, themes, and limits.
All business logic reads from this module to ensure explainability and reproducibility.
"""
from dataclasses import dataclass
from typing import Dict, Any

# File loading configuration
MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
DEFAULT_CSV_ENCODINGS = ["utf-8", "latin-1", "cp1252"]

# Column classification thresholds
ID_UNIQUENESS_RATIO = 0.95
ID_MIN_ROWS = 20
NUMERIC_CATEGORICAL_MAX_UNIQUE = 5
DATE_SAMPLE_SIZE = 50
DATE_PARSE_SUCCESS_RATE = 0.90
CATEGORICAL_RATIO = 0.05
CATEGORICAL_MIN_THRESHOLD = 100  # Allows up to 100 distinct categories (states, product codes, details)
MAX_CATEGORIES_FOR_BREAKDOWN = 30

# Data Detective / Quality Scoring configuration
SCORE_WEIGHT_MISSING = 1.5
SCORE_WEIGHT_DUPLICATE = 2.0
SCORE_WEIGHT_OUTLIER = 0.5
SKEWNESS_THRESHOLD = 1.5  # Skewness beyond this indicates skewed distribution rather than rogue errors
IQR_STANDARD_MULTIPLIER = 1.5
IQR_SKEWED_MULTIPLIER = 3.0

# Cleaning Assistant Decision Rules
MISSING_THRESHOLD_LOW = 5.0     # <= 5%: median / mode
MISSING_THRESHOLD_HIGH = 15.0   # 5-15%: KNN (numeric), manual (others); >15%: manual
KNN_NEIGHBORS = 5

# Pattern Hunter configuration
CORRELATION_MODERATE_THRESHOLD = 0.5
CORRELATION_STRONG_THRESHOLD = 0.7
MAX_SCATTER_SAMPLE_SIZE = 5000
DEFAULT_TOP_N_CATEGORIES = 10

# UI & Visualization Theme Tokens (conforming to ui-styling & ui-ux-pro-max)
THEME = {
    "primary": "#1E40AF",        # Deep Blue
    "secondary": "#3B82F6",      # Electric Blue
    "accent": "#D97706",         # Amber
    "background": "#F8FAFC",     # Light Slate
    "surface": "#FFFFFF",        # White Card
    "foreground": "#0F172A",     # Deep Slate Text
    "muted": "#64748B",          # Muted Slate
    "border": "#E2E8F0",         # Soft Border
    "success": "#10B981",        # Emerald Green
    "warning": "#F59E0B",        # Amber Warning
    "danger": "#EF4444",         # Rose Destructive
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "mono_family": "'Fira Code', 'JetBrains Mono', Consolas, monospace",
}
