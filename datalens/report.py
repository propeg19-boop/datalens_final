"""
datalens/report.py
Exportable summary report generator for DataLens.
Generates comprehensive Markdown and styled HTML reports summarizing data health,
applied cleaning actions, top statistical patterns, and predictive model benchmarks.
"""
from typing import Dict, Any, Optional
import html


def generate_markdown_report(
    summary: Dict[str, Any],
) -> str:
    """Produce clean Markdown report for documentation, college submissions, or GitHub."""
    dataset_name = summary.get("dataset_name", "Uploaded Dataset")
    n_rows = summary.get("rows", 0)
    n_cols = summary.get("cols", 0)
    score_before = summary.get("score_before", 0.0)
    score_after = summary.get("score_after", score_before)
    col_types = summary.get("column_types", {})
    applied_log = summary.get("applied_log", [])
    findings = summary.get("findings", {})
    model_res = summary.get("model_results", {})

    lines = [
        f"# DataLens Analysis & Verification Report: {dataset_name}",
        f"\n**Date:** {summary.get('generated_at', '2026-11-03')}",
        f"**Shape:** {n_rows:,} rows × {n_cols} columns\n",
        "---",
        "\n## 1. Executive Summary & Data Health\n",
        f"- **Initial Quality Score:** `{score_before} / 100`",
        f"- **Post-Cleaning Quality Score:** `{score_after} / 100`",
        f"- **Improvement Delta:** `+{round(score_after - score_before, 1)} pts`\n",
        "### Column Classification Breakdown\n",
        "| Column | Detected Type |",
        "|---|---|",
    ]

    for col, ctype in col_types.items():
        lines.append(f"| `{col}` | `{ctype.upper()}` |")

    lines.extend([
        "\n---",
        "\n## 2. Applied Cleaning Transformations\n",
    ])

    if applied_log:
        lines.append("| Target | Action | Result / Value | Rows Affected | Status |")
        lines.append("|---|---|---|---|---|")
        for item in applied_log:
            lines.append(
                f"| `{item.get('target')}` | `{item.get('action')}` | "
                f"`{item.get('value_used')}` | {item.get('rows_affected')} | {item.get('status').capitalize()} |"
            )
    else:
        lines.append("_No cleaning transformations were applied to this dataset._\n")

    lines.extend([
        "\n---",
        "\n## 3. Key Statistical Insights\n",
    ])

    corrs = findings.get("correlations", [])
    if corrs:
        lines.append("### Top Pairwise Correlations")
        for c in corrs[:3]:
            lines.append(
                f"- **{c['col1']}** vs **{c['col2']}**: r = `{c['r']}` ({c['strength'].capitalize()} {c['direction']})"
            )
            if "caption" in c:
                lines.append(f"  > _{c['caption']}_\n")

    trends = findings.get("trends", [])
    if trends:
        lines.append("### Key Time Series Trends")
        for t in trends[:3]:
            lines.append(
                f"- **{t['metric_col']}** over `{t['date_col']}`: {t['direction'].capitalize()} trend ({t['pct_change_per_step']}% per {t['period_type']})"
            )
            if "caption" in t:
                lines.append(f"  > _{t['caption']}_\n")

    if model_res and "metrics_table" in model_res:
        lines.extend([
            "\n---",
            f"\n## 4. Baseline Machine Learning ({model_res.get('task', '').capitalize()})\n",
            f"**Target Column:** `{model_res.get('target_col')}`  ",
            f"**Verdict:** {model_res.get('verdict', {}).get('badge')} — _{model_res.get('verdict', {}).get('text')}_\n",
        ])

    return "\n".join(lines)


def generate_html_report(summary: Dict[str, Any]) -> str:
    """Generate self-contained, publication-ready HTML report with modern CSS."""
    md_content = generate_markdown_report(summary)
    dataset_name = html.escape(str(summary.get("dataset_name", "Dataset")))
    score_b = summary.get("score_before", 0.0)
    score_a = summary.get("score_after", score_b)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DataLens Report - {dataset_name}</title>
  <style>
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #0F172A;
      background: #F8FAFC;
      margin: 0;
      padding: 40px 20px;
    }}
    .container {{
      max-width: 860px;
      margin: 0 auto;
      background: #FFFFFF;
      padding: 40px;
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
      border: 1px solid #E2E8F0;
    }}
    h1 {{ color: #1E40AF; border-bottom: 2px solid #E2E8F0; padding-bottom: 12px; }}
    h2 {{ color: #1E3A8A; margin-top: 30px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ border: 1px solid #E2E8F0; padding: 10px 14px; text-align: left; font-size: 14px; }}
    th {{ background: #F1F5F9; color: #334155; font-weight: 600; }}
    code {{ font-family: 'Fira Code', monospace; background: #F1F5F9; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-family: monospace; }}
    .badge-score {{ background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>DataLens Quality & Analytics Report</h1>
    <p><strong>Dataset:</strong> {dataset_name} | <strong>Health Score:</strong> <span class="badge badge-score">{score_a}/100</span></p>
    <pre style="white-space: pre-wrap; font-family: inherit;">{html.escape(md_content)}</pre>
  </div>
</body>
</html>
"""
