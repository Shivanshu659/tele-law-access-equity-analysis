"""
visualisation.py
================
Generates all charts and exports an HTML equity report for the
Tele-Law Access Equity Analysis project.

Dependencies: matplotlib, seaborn, plotly
All charts are saved to reports/charts/ and referenced in the HTML report.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logger = logging.getLogger(__name__)

CHART_DIR  = Path(os.getenv("REPORT_OUTPUT_DIR", "reports")) / "charts"
REPORT_DIR = Path(os.getenv("REPORT_OUTPUT_DIR", "reports"))

# Style
sns.set_theme(style="whitegrid", palette="muted")
ACCENT_COLOR   = "#3b82d4"
WARN_COLOR     = "#f59e0b"
DANGER_COLOR   = "#ef4444"
SUCCESS_COLOR  = "#22c55e"
PALETTE_CASTE  = ["#3b82d4", "#7c5cd8", "#f59e0b", "#22c55e"]


def _ensure_dirs() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(name: str) -> Path:
    path = CHART_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved -> %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Gender Distribution – National Pie
# ---------------------------------------------------------------------------

def plot_gender_pie(national: dict) -> Path:
    _ensure_dirs()
    labels = ["Male", "Female", "Other"]
    sizes  = [national["male_pct"], national["female_pct"], national["other_pct"]]
    colors = [ACCENT_COLOR, WARN_COLOR, SUCCESS_COLOR]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title("Gender-wise Case Registrations (National)", fontsize=13, pad=15)
    return _savefig("gender_pie.png")


# ---------------------------------------------------------------------------
# 2. Gender Trend – Line Chart
# ---------------------------------------------------------------------------

def plot_gender_trend(by_year: pd.DataFrame) -> Path:
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_year.index, by_year["male_pct"],   marker="o", label="Male %",   color=ACCENT_COLOR)
    ax.plot(by_year.index, by_year["female_pct"], marker="s", label="Female %", color=WARN_COLOR)
    ax.set_title("Gender Share Trend (Year-over-Year)", fontsize=13)
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Percentage (%)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    plt.xticks(rotation=30)
    return _savefig("gender_trend.png")


# ---------------------------------------------------------------------------
# 3. Caste Distribution – Stacked Bar by State (top 15 states)
# ---------------------------------------------------------------------------

def plot_caste_by_state(by_state: pd.DataFrame, top_n: int = 15) -> Path:
    _ensure_dirs()
    df = by_state.head(top_n)[["sc_pct", "st_pct", "obc_pct", "general_pct"]].copy()
    df.columns = ["SC", "ST", "OBC", "General"]

    ax = df.plot(
        kind="barh", stacked=True, figsize=(10, 6),
        color=PALETTE_CASTE, width=0.7,
    )
    ax.set_title(f"Caste-wise Case Distribution – Top {top_n} States", fontsize=13)
    ax.set_xlabel("Percentage of Cases (%)")
    ax.set_ylabel("State")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    return _savefig("caste_state_bar.png")


# ---------------------------------------------------------------------------
# 4. Marginalized % Heatmap (State × Year)
# ---------------------------------------------------------------------------

def plot_marginalized_heatmap(df_full: pd.DataFrame) -> Path:
    """Pivot table of avg marginalized_pct per state per year."""
    _ensure_dirs()
    pivot = (
        df_full.groupby(["state_name", "financial_year"])["marginalized_pct"]
               .mean()
               .unstack("financial_year")
               .fillna(0)
    )
    # Trim to manageable size
    pivot = pivot.iloc[:25] if len(pivot) > 25 else pivot

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Marginalized %"},
    )
    ax.set_title("Marginalized Group Share (%) – State × Financial Year", fontsize=13)
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("State")
    plt.xticks(rotation=30)
    return _savefig("marginalized_heatmap.png")


# ---------------------------------------------------------------------------
# 5. Cases per CSC – Top vs Bottom Districts
# ---------------------------------------------------------------------------

def plot_cases_per_csc(top20: pd.DataFrame, bot20: pd.DataFrame) -> Path:
    _ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, color in [
        (axes[0], top20, "Top 20 Districts – Cases/CSC", SUCCESS_COLOR),
        (axes[1], bot20, "Bottom 20 Districts – Cases/CSC", DANGER_COLOR),
    ]:
        labels = data["district_name"] + "\n(" + data["state_name"] + ")"
        ax.barh(range(len(data)), data["cases_per_csc"], color=color, edgecolor="white")
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Cases per CSC")
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()

    fig.suptitle("Regional Access Density – Cases per CSC", fontsize=13, y=1.02)
    return _savefig("cases_per_csc.png")


# ---------------------------------------------------------------------------
# 6. Advice Rate by State – Horizontal Bar
# ---------------------------------------------------------------------------

def plot_advice_rate(state_summary: pd.DataFrame) -> Path:
    _ensure_dirs()
    data = state_summary["advice_rate_pct"].sort_values()

    fig, ax = plt.subplots(figsize=(8, max(5, len(data) * 0.28)))
    colors = [SUCCESS_COLOR if v >= 80 else WARN_COLOR if v >= 60 else DANGER_COLOR
              for v in data]
    ax.barh(data.index, data.values, color=colors, edgecolor="white")
    ax.axvline(80, color="gray", linestyle="--", linewidth=1, label="80% target")
    ax.set_xlabel("Advice Delivery Rate (%)")
    ax.set_title("State-wise Advice Delivery Rate", fontsize=13)
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    return _savefig("advice_rate.png")


# ---------------------------------------------------------------------------
# 7. Equity Score – Top / Bottom 20 Districts
# ---------------------------------------------------------------------------

def plot_equity_scores(equity_df: pd.DataFrame) -> Path:
    _ensure_dirs()
    top    = equity_df.head(15)
    bottom = equity_df.tail(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, color in [
        (axes[0], top,    "Top 15 Equitable Districts",   SUCCESS_COLOR),
        (axes[1], bottom, "Bottom 15 Inequitable Districts", DANGER_COLOR),
    ]:
        labels = data["district_name"] + "\n(" + data["state_name"] + ")"
        ax.barh(range(len(data)), data["equity_score"], color=color, edgecolor="white")
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Equity Score (0–100)")
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()

    fig.suptitle("Composite Equity Index by District", fontsize=13, y=1.02)
    return _savefig("equity_scores.png")


# ---------------------------------------------------------------------------
# 8. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation(corr: pd.DataFrame) -> Path:
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation – Equity Indicators", fontsize=13)
    return _savefig("correlation.png")


# ---------------------------------------------------------------------------
# 9. Yearly Trend – Multi-metric Line
# ---------------------------------------------------------------------------

def plot_yearly_trend(yearly: pd.DataFrame) -> Path:
    _ensure_dirs()
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.bar(yearly.index, yearly["total_cases"] / 1000, alpha=0.4,
            color=ACCENT_COLOR, label="Total Cases (K)")
    ax2.plot(yearly.index, yearly["advice_rate_pct"], marker="o",
             color=SUCCESS_COLOR, label="Advice Rate %")
    ax2.plot(yearly.index, yearly["marginalized_pct"], marker="s",
             color=WARN_COLOR, label="Marginalized %")
    ax2.plot(yearly.index, yearly["female_pct"], marker="^",
             color=DANGER_COLOR, label="Female %")

    ax1.set_ylabel("Total Cases (Thousands)")
    ax2.set_ylabel("Percentage (%)")
    ax1.set_title("Year-over-Year Tele-Law Trends", fontsize=13)
    ax1.set_xlabel("Financial Year")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    plt.xticks(rotation=30)
    return _savefig("yearly_trend.png")


# ---------------------------------------------------------------------------
# 10. HTML Report
# ---------------------------------------------------------------------------

def _img_tag(path: Path, alt: str, width: str = "100%") -> str:
    rel = path.relative_to(REPORT_DIR) if path.is_relative_to(REPORT_DIR) else path
    return f'<img src="{rel.as_posix()}" alt="{alt}" style="width:{width};border-radius:6px;margin:8px 0;">'


def generate_html_report(results: dict, insights: dict, chart_paths: dict) -> Path:
    """
    Generate a full static HTML equity report from analysis results and
    Granite-generated insights.
    """
    _ensure_dirs()
    now = pd.Timestamp.now().strftime("%d %B %Y, %H:%M IST")
    gender_nat  = results["gender"]["national"]
    caste_nat   = results["caste"]["national"]
    gini        = results["caste"]["gini_coefficient"]
    trend_latest = results["trend"]["yearly"].iloc[-1].to_dict() \
                   if not results["trend"]["yearly"].empty else {}

    def section(title: str, content: str, bg: str = "#ffffff") -> str:
        return f"""
        <section style="background:{bg};border:1px solid #e5e7eb;border-radius:8px;
                         padding:24px;margin-bottom:20px;">
          <h2 style="color:#1f2328;border-bottom:2px solid #3b82d4;padding-bottom:6px;
                      margin-top:0;">{title}</h2>
          {content}
        </section>"""

    def kpi(label: str, value: str, color: str = "#3b82d4") -> str:
        return f"""<div style="display:inline-block;background:#f7f8fa;border:1px solid #e5e7eb;
                               border-radius:6px;padding:12px 20px;margin:6px;min-width:140px;
                               text-align:center;">
                     <div style="font-size:22px;font-weight:700;color:{color};">{value}</div>
                     <div style="font-size:12px;color:#57606a;margin-top:4px;">{label}</div>
                   </div>"""

    def insight_box(text: str) -> str:
        return f"""<div style="background:#f0f7ff;border-left:4px solid #3b82d4;
                               padding:14px 18px;margin-top:12px;border-radius:0 6px 6px 0;
                               font-style:italic;color:#374151;line-height:1.7;">
                    🤖 <strong>IBM Granite Insight:</strong><br>{text}
                  </div>"""

    # ── Build sections ──────────────────────────────────────────────────────
    kpis_html = (
        kpi("Total Cases",    f"{gender_nat['total_cases']:,}", "#3b82d4") +
        kpi("Female %",       f"{gender_nat['female_pct']}%",  "#f59e0b") +
        kpi("Marginalized %", f"{caste_nat['marginalized_pct']}%", "#7c5cd8") +
        kpi("Gini Index",     str(gini),  "#ef4444") +
        kpi("SC %",           f"{caste_nat['sc_pct']}%",  "#22c55e") +
        kpi("ST %",           f"{caste_nat['st_pct']}%",  "#06b6d4") +
        kpi("OBC %",          f"{caste_nat['obc_pct']}%", "#8b5cf6")
    )

    gender_section = section(
        "1. Gender-wise Analysis",
        (f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
         f'{_img_tag(chart_paths["gender_pie"], "Gender Pie", "280px")}'
         f'{_img_tag(chart_paths["gender_trend"], "Gender Trend", "calc(100% - 300px)")}'
         f'</div>'
         f'<p>Female participation nationally: <strong>{gender_nat["female_pct"]}%</strong>. '
         f'Districts with female share below 30%: <strong>{len(results["gender"]["skewed_districts"])}</strong></p>'
         + insight_box(insights.get("gender_insight", "—"))),
        "#ffffff"
    )

    caste_section = section(
        "2. Caste-wise / Marginalization Analysis",
        (f'{_img_tag(chart_paths["caste_state_bar"], "Caste by State")}'
         f'{_img_tag(chart_paths["marginalized_heatmap"], "Marginalized Heatmap")}'
         f'<p>National marginalized share: <strong>{caste_nat["marginalized_pct"]}%</strong> '
         f'(SC {caste_nat["sc_pct"]}% · ST {caste_nat["st_pct"]}% · OBC {caste_nat["obc_pct"]}%). '
         f'Gini coefficient: <strong>{gini}</strong>.</p>'
         + insight_box(insights.get("caste_insight", "—"))),
    )

    regional_section = section(
        "3. Geographic / Regional Disparity",
        (f'{_img_tag(chart_paths["cases_per_csc"], "Cases per CSC")}'
         f'{_img_tag(chart_paths["advice_rate"], "Advice Rate")}'
         + insight_box(insights.get("regional_insight", "—"))),
    )

    equity_section = section(
        "4. Composite Equity Index",
        f'{_img_tag(chart_paths["equity_scores"], "Equity Scores")}'
        f'<p>The Equity Index (0–100) combines marginalized group share (35%), '
        f'female participation (25%), advice delivery rate (25%), and CSC access density (15%).</p>',
        "#f7f8fa"
    )

    trend_section = section(
        "5. Year-over-Year Trends",
        f'{_img_tag(chart_paths["yearly_trend"], "Yearly Trend")}',
    )

    corr_section = section(
        "6. Correlation Analysis",
        f'{_img_tag(chart_paths["correlation"], "Correlation Heatmap")}',
        "#f7f8fa"
    )

    exec_summary = section(
        "Executive Summary",
        f'<p style="color:#57606a;font-size:13px;">Generated by IBM Granite AI</p>'
        + insight_box(insights.get("executive_summary", "—")),
        "#fffbeb"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Tele-Law Access Equity Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system,"Segoe UI",system-ui,sans-serif;
            background:#f7f8fa;color:#1f2328;margin:0;padding:20px;font-size:14px;line-height:1.6; }}
    .container {{ max-width:960px;margin:0 auto; }}
    header {{ background:#1f2328;color:#fff;padding:28px 32px;border-radius:8px;margin-bottom:24px; }}
    header h1 {{ margin:0 0 4px;font-size:22px; }}
    header p  {{ margin:0;color:#9ca3af;font-size:13px; }}
    .badge {{ display:inline-block;background:#3b82d4;color:#fff;font-size:11px;
              padding:3px 10px;border-radius:20px;margin-right:6px;margin-top:8px; }}
    .kpi-row {{ margin-bottom:20px; }}
    footer {{ text-align:center;padding:16px;color:#9ca3af;font-size:12px;
              border-top:1px solid #e5e7eb;margin-top:24px; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Tele-Law Access Equity Analysis</h1>
    <p>District-wise Case Registration &amp; Advice Enabled Data · FY 2021-22 to 2024-25</p>
    <span class="badge">IBM Cloud Lite</span>
    <span class="badge">IBM Granite AI</span>
    <span class="badge">data.gov.in</span>
    <p style="margin-top:10px;font-size:12px;color:#6b7280;">Report generated: {now}</p>
  </header>

  <div class="kpi-row">{kpis_html}</div>

  {exec_summary}
  {gender_section}
  {caste_section}
  {regional_section}
  {equity_section}
  {trend_section}
  {corr_section}

  <footer>Made with IBM Bob &nbsp;|&nbsp; IBM Cloud Lite &nbsp;|&nbsp; IBM Granite AI</footer>
</div>
</body>
</html>"""

    out_path = REPORT_DIR / "equity_report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def generate_all_charts(results: dict, df_full: pd.DataFrame) -> dict:
    """Generate all charts and return a dict of {name: Path}."""
    g  = results["gender"]
    c  = results["caste"]
    r  = results["regional"]
    eq = results["equity"]
    t  = results["trend"]
    co = results["correlation"]

    return {
        "gender_pie":          plot_gender_pie(g["national"]),
        "gender_trend":        plot_gender_trend(g["by_year"]),
        "caste_state_bar":     plot_caste_by_state(c["by_state"]),
        "marginalized_heatmap":plot_marginalized_heatmap(df_full),
        "cases_per_csc":       plot_cases_per_csc(r["district_top20"], r["district_bot20"]),
        "advice_rate":         plot_advice_rate(r["state_summary"]),
        "equity_scores":       plot_equity_scores(eq),
        "correlation":         plot_correlation(co),
        "yearly_trend":        plot_yearly_trend(t["yearly"]),
    }
