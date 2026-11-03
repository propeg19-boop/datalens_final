"""
ui/components.py
Reusable, accessible UI components for DataLens.
Designed using ui-styling and ui-ux-pro-max guidelines.
Provides HTML rendering for metrics, health gauges, recommendation cards, and before/after diffs.
Uses clean SVG vector icons (no emojis as structural icons).
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import html
import streamlit as st


def inject_custom_css() -> None:
    """Inject DataLens styles.css into the active Streamlit app."""
    css_path = Path(__file__).resolve().parent / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_svg_icon(icon_name: str) -> str:
    """Return inline accessible SVG icon markup."""
    icons = {
        "database": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>'
            '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>'
            '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>'
        ),
        "check": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<polyline points="20 6 9 17 4 12"></polyline></svg>'
        ),
        "alert": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>'
            '<line x1="12" y1="9" x2="12" y2="13"></line>'
            '<line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        ),
        "shield": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        ),
        "columns": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<rect width="18" height="18" x="3" y="3" rx="2"></rect>'
            '<path d="M12 3v18"></path></svg>'
        ),
        "rows": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<rect width="18" height="18" x="3" y="3" rx="2"></rect>'
            '<path d="M3 12h18"></path></svg>'
        ),
        "sparkles": (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"></path></svg>'
        ),
    }
    return icons.get(icon_name, icons["database"])


def render_hero(title: str, subtitle: str) -> str:
    """Render hero header banner."""
    esc_title = html.escape(title)
    esc_subtitle = html.escape(subtitle)
    icon_svg = get_svg_icon("sparkles")
    return f"""
    <div class="dl-hero-card">
        <div class="dl-hero-title">
            {icon_svg}
            <span>{esc_title}</span>
        </div>
        <p class="dl-hero-subtitle">{esc_subtitle}</p>
    </div>
    """


def render_type_badge(col_type: str) -> str:
    """Render accessible pill badge for column types."""
    c_type = col_type.lower()
    badge_class = f"dl-badge dl-badge-{c_type}" if c_type in {"id", "numeric", "categorical", "datetime", "text"} else "dl-badge dl-badge-text"
    return f'<span class="{badge_class}">{html.escape(c_type.upper())}</span>'


def render_metric_card(
    label: str,
    value: Any,
    subtext: Optional[str] = None,
    icon: str = "database",
) -> str:
    """Render a standalone KPI card."""
    esc_label = html.escape(str(label))
    esc_value = html.escape(str(value))
    esc_subtext = f'<div class="dl-metric-subtext">{html.escape(str(subtext))}</div>' if subtext else ""
    icon_svg = get_svg_icon(icon)

    return f"""
    <div class="dl-metric-card">
        <div class="dl-metric-header">
            <span class="dl-metric-label">{esc_label}</span>
            <span style="color: var(--dl-secondary);">{icon_svg}</span>
        </div>
        <div class="dl-metric-value">{esc_value}</div>
        {esc_subtext}
    </div>
    """


def render_quality_gauge(score: float, breakdown: Optional[Dict[str, Any]] = None) -> str:
    """Render large data quality score card with breakdown."""
    score_val = round(float(score), 1)
    if score_val >= 90.0:
        badge_class = "dl-score-good"
        status_text = "High Quality Dataset"
    elif score_val >= 75.0:
        badge_class = "dl-score-moderate"
        status_text = "Acceptable — Cleaning Recommended"
    else:
        badge_class = "dl-score-critical"
        status_text = "Requires Urgent Attention"

    breakdown_html = ""
    if breakdown:
        init_s = breakdown.get("initial", 100.0)
        m_pen = breakdown.get("missing_penalty", 0.0)
        d_pen = breakdown.get("duplicate_penalty", 0.0)
        o_pen = breakdown.get("outlier_penalty", 0.0)

        breakdown_html = f"""
        <div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--dl-border); font-size: 13px; color: var(--dl-text-muted);">
            <div style="font-weight: 600; margin-bottom: 6px; color: var(--dl-text);">Score Deduction Breakdown:</div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <span>Starting Base: <strong style="color: var(--dl-text);">{init_s} pts</strong></span>
                <span>Missingness: <strong style="color: var(--dl-danger); font-family: var(--dl-font-mono);">-{m_pen} pts</strong></span>
                <span>Duplicates: <strong style="color: var(--dl-danger); font-family: var(--dl-font-mono);">-{d_pen} pts</strong></span>
                <span>Outliers: <strong style="color: var(--dl-warning); font-family: var(--dl-font-mono);">-{o_pen} pts</strong></span>
            </div>
        </div>
        """

    return f"""
    <div class="dl-health-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; color: var(--dl-text-muted); letter-spacing: 0.04em;">
                    Overall Data Health Score
                </div>
                <div style="font-size: 18px; font-weight: 600; color: var(--dl-text); margin-top: 2px;">
                    {status_text}
                </div>
            </div>
            <div class="dl-score-badge-large {badge_class}">
                <span style="font-size: 32px;">{score_val}</span>
                <span style="font-size: 16px; opacity: 0.8;">/ 100</span>
            </div>
        </div>
        {breakdown_html}
    </div>
    """


def render_recommendation_card(rec: Dict[str, Any]) -> str:
    """Render a single cleaning recommendation card."""
    col = rec.get("column", "Dataset")
    col_type = rec.get("column_type", "dataset")
    issue = rec.get("issue", "")
    action = rec.get("action", "")
    reasoning = rec.get("reasoning", "")
    suggested = rec.get("suggested_value")
    is_safe = rec.get("is_safe_auto", False)

    card_mod = "dl-rec-card-safe" if is_safe else "dl-rec-card-manual"
    type_badge = render_type_badge(col_type)

    suggested_box = ""
    if suggested is not None:
        suggested_box = f"""
        <div style="margin-top: 8px; font-size: 12px; color: var(--dl-text);">
            <strong>Proposed Value:</strong> <code style="font-family: var(--dl-font-mono); background: #E2E8F0; padding: 2px 6px; border-radius: 4px;">{html.escape(str(suggested))}</code>
        </div>
        """

    return f"""
    <div class="dl-rec-card {card_mod}">
        <div class="dl-rec-header">
            <div>
                <div class="dl-rec-title">
                    <span>{html.escape(col)}</span>
                    {type_badge}
                </div>
                <div style="font-size: 13px; color: var(--dl-danger); font-weight: 500; margin-top: 4px;">
                    {html.escape(issue)}
                </div>
            </div>
            <span class="dl-badge" style="background: var(--dl-primary-subtle); color: var(--dl-primary); font-size: 12px; font-family: var(--dl-font-mono);">
                {html.escape(action)}
            </span>
        </div>
        <div class="dl-rec-reason">
            <strong>Why:</strong> {html.escape(reasoning)}
        </div>
        {suggested_box}
    </div>
    """


def render_before_after_diff(comp: Dict[str, Any]) -> str:
    """Render side-by-side comparison card proving cleaning efficacy."""
    score_b = comp.get("score_before", 0.0)
    score_a = comp.get("score_after", 0.0)
    score_imp = comp.get("score_improvement", 0.0)
    miss_b = comp.get("missing_before", 0)
    miss_a = comp.get("missing_after", 0)
    dup_b = comp.get("duplicates_before", 0)
    dup_a = comp.get("duplicates_after", 0)

    score_imp_badge = (
        f'<span style="color: var(--dl-success); font-weight: 700;">+{score_imp} pts</span>'
        if score_imp > 0
        else f'<span>{score_imp} pts</span>'
    )

    return f"""
    <div class="dl-diff-container">
        <div class="dl-diff-column dl-diff-before">
            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #9F1239; margin-bottom: 8px;">
                Original (Before Cleaning)
            </div>
            <div style="margin-bottom: 6px;">Quality Score: <strong style="font-family: var(--dl-font-mono); font-size: 18px;">{score_b}</strong></div>
            <div style="margin-bottom: 4px; font-size: 13px;">Missing Cells: <strong style="font-family: var(--dl-font-mono);">{miss_b}</strong></div>
            <div style="font-size: 13px;">Duplicate Rows: <strong style="font-family: var(--dl-font-mono);">{dup_b}</strong></div>
        </div>
        <div class="dl-diff-column dl-diff-after">
            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #166534; margin-bottom: 8px;">
                Cleaned (After Transformations)
            </div>
            <div style="margin-bottom: 6px;">Quality Score: <strong style="font-family: var(--dl-font-mono); font-size: 18px;">{score_a}</strong> ({score_imp_badge})</div>
            <div style="margin-bottom: 4px; font-size: 13px;">Missing Cells: <strong style="font-family: var(--dl-font-mono); color: var(--dl-success);">{miss_a}</strong> (Resolved: {comp.get('missing_resolved', 0)})</div>
            <div style="font-size: 13px;">Duplicate Rows: <strong style="font-family: var(--dl-font-mono); color: var(--dl-success);">{dup_a}</strong> (Resolved: {comp.get('duplicates_resolved', 0)})</div>
        </div>
    </div>
    """

