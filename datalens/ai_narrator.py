"""
datalens/ai_narrator.py
Plain-English caption generator for DataLens.
Turns structured finding dicts into beginner-friendly, non-technical explanations.
Includes a 100% deterministic template fallback so the app functions completely offline / keyless.
Never sends raw rows to external APIs - only aggregated summary statistics.
"""
from typing import Dict, Any, Optional
import os
import json
import hashlib


def _generate_template_caption(finding: Dict[str, Any], finding_type: str) -> str:
    """Generate high-quality, factual template sentences without LLM dependency."""
    if finding_type == "correlation":
        c1 = finding.get("col1", "Feature A")
        c2 = finding.get("col2", "Feature B")
        r = finding.get("r", 0.0)
        direction = finding.get("direction", "positive")
        strength = finding.get("strength", "moderate")

        if direction == "positive":
            movement = "increase together"
        else:
            movement = "move in opposite directions (as one rises, the other tends to fall)"

        return (
            f"There is a {strength} {direction} correlation (r = {r}) between '{c1}' and '{c2}'. "
            f"This indicates that higher values of '{c1}' generally {movement}. "
            f"Remember that statistical correlation does not imply direct causation."
        )

    elif finding_type == "trend":
        metric = finding.get("metric_col", "Metric")
        period = finding.get("period_type", "daily")
        direction = finding.get("direction", "flat")
        start = finding.get("span_start", "start")
        end = finding.get("span_end", "end")
        pct = finding.get("pct_change_per_step", 0.0)

        if direction == "upward":
            desc = f"shows an upward trend, rising by approximately {pct:.1f}% per {period} interval"
        elif direction == "downward":
            desc = f"shows a downward trend, declining by approximately {abs(pct):.1f}% per {period} interval"
        else:
            desc = "remains relatively flat and stable across the observed timeline"

        return (
            f"Across the period from {start} to {end}, '{metric}' {desc}. "
            f"This represents the overall directional trajectory, not a guaranteed future forecast."
        )

    elif finding_type == "breakdown":
        cat_col = finding.get("category_col", "Category")
        metric = finding.get("metric_col", "Metric")
        top_cat = finding.get("top_category", "Top")
        top_sum = finding.get("top_category_sum", 0.0)
        top_pct = finding.get("top_category_share_pct", 0.0)
        bottom_cat = finding.get("bottom_category", "Bottom")
        bottom_sum = finding.get("bottom_category_sum", 0.0)

        return (
            f"In the breakdown of '{metric}' by '{cat_col}', '{top_cat}' leads with a total of "
            f"{top_sum:,.1f} ({top_pct:.1f}% of all observed volume). "
            f"In contrast, '{bottom_cat}' contributed the lowest volume ({bottom_sum:,.1f})."
        )

    elif finding_type == "quality":
        score = finding.get("quality_score", 100.0)
        miss = finding.get("total_missing_cells", 0)
        dups = finding.get("duplicate_rows", 0)
        outliers = finding.get("total_outlier_count", 0)

        if score >= 90:
            health_desc = "in strong structural health"
        elif score >= 75:
            health_desc = "moderately clean, with a few defects recommended for cleaning"
        else:
            health_desc = "severely impaired by missingness, duplicate rows, or extreme anomalies"

        return (
            f"The dataset scored {score}/100 and is {health_desc}. "
            f"The audit flagged {miss:,} missing entries, {dups:,} exact duplicate rows, "
            f"and {outliers:,} potential outlier values."
        )

    return "No plain-English summary available for this finding."


def generate_caption(
    finding: Dict[str, Any],
    finding_type: str,
    use_ai: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate an explainable caption for a finding.
    If use_ai is False or no API key is available, returns deterministic template text.
    """
    # Deterministic fallback check
    if not use_ai:
        return _generate_template_caption(finding, finding_type)

    resolved_key = (
        api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not resolved_key:
        return _generate_template_caption(finding, finding_type)

    # If Anthropic API key is present
    if "ANTHROPIC_API_KEY" in os.environ or (api_key and "sk-ant" in api_key):
        try:
            import urllib.request
            url = "https://api.anthropic.com/v1/messages"
            prompt = (
                f"You are a friendly data educator. Explain this statistical finding in 2 clear sentences "
                f"for a non-technical person. Use only the provided numbers; never invent statistics. "
                f"Tone: objective, beginner-friendly, clarify correlation is not causation.\n\n"
                f"Finding ({finding_type}): {json.dumps(finding)}"
            )
            data = json.dumps({
                "model": "claude-3-haiku-20240307",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "x-api-key": resolved_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                return res_json["content"][0]["text"].strip()
        except Exception:
            return _generate_template_caption(finding, finding_type)

    return _generate_template_caption(finding, finding_type)
