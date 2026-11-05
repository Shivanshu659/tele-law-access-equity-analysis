"""
analysis.py
===========
Core analytical engine for the Tele-Law equity study.

Provides:
  - Gender-wise disparity analysis
  - Caste-wise marginalization analysis
  - Geographic (state / district) disparity analysis
  - CSC-normalized regional comparison
  - Equity scoring (composite index)
  - Trend analysis across financial years
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan) * 100).round(2)


# ---------------------------------------------------------------------------
# 1. Gender-wise Analysis
# ---------------------------------------------------------------------------

def gender_disparity(df: pd.DataFrame) -> dict:
    """
    Compute national, state-level, and district-level gender distribution.

    Returns
    -------
    dict with keys:
        national   – overall male/female/other percentages
        by_state   – DataFrame indexed by state
        by_year    – DataFrame indexed by financial_year
        skewed_districts – districts where female_pct < 30 %
    """
    total = df["total_cases_registered"].sum()
    national = {
        "male_pct":   round(df["gender_male"].sum()   / total * 100, 2),
        "female_pct": round(df["gender_female"].sum() / total * 100, 2),
        "other_pct":  round(df["gender_other"].sum()  / total * 100, 2),
        "total_cases": int(total),
    }

    by_state = (
        df.groupby("state_name")[["gender_male", "gender_female",
                                  "gender_other", "total_cases_registered"]]
          .sum()
          .assign(
              male_pct   = lambda x: _safe_pct(x["gender_male"],   x["total_cases_registered"]),
              female_pct = lambda x: _safe_pct(x["gender_female"], x["total_cases_registered"]),
              other_pct  = lambda x: _safe_pct(x["gender_other"],  x["total_cases_registered"]),
          )
          .sort_values("total_cases_registered", ascending=False)
    )

    by_year = (
        df.groupby("financial_year")[["gender_male", "gender_female",
                                      "gender_other", "total_cases_registered"]]
          .sum()
          .assign(
              male_pct   = lambda x: _safe_pct(x["gender_male"],   x["total_cases_registered"]),
              female_pct = lambda x: _safe_pct(x["gender_female"], x["total_cases_registered"]),
          )
    )

    skewed = (
        df[df["female_pct"] < 30][["state_name", "district_name",
                                    "financial_year", "female_pct",
                                    "total_cases_registered"]]
          .sort_values("female_pct")
    )

    logger.info("Gender analysis complete. Skewed districts: %d", len(skewed))
    return {
        "national": national,
        "by_state": by_state,
        "by_year": by_year,
        "skewed_districts": skewed,
    }


# ---------------------------------------------------------------------------
# 2. Caste-wise / Marginalization Analysis
# ---------------------------------------------------------------------------

def caste_disparity(df: pd.DataFrame) -> dict:
    """
    Analyse SC / ST / OBC representation vs General category.

    Returns
    -------
    dict with keys:
        national           – national caste breakdown %
        by_state           – DataFrame
        by_year            – DataFrame
        underserved_districts – districts where marginalized_pct < equity threshold
        gini_coefficient   – float, inequality measure across districts
    """
    total = df["total_cases_registered"].sum()
    national = {
        "sc_pct":          round(df["caste_sc"].sum()      / total * 100, 2),
        "st_pct":          round(df["caste_st"].sum()      / total * 100, 2),
        "obc_pct":         round(df["caste_obc"].sum()     / total * 100, 2),
        "general_pct":     round(df["caste_general"].sum() / total * 100, 2),
        "marginalized_pct":round(df["marginalized_cases"].sum() / total * 100, 2),
    }

    by_state = (
        df.groupby("state_name")[["caste_sc", "caste_st", "caste_obc",
                                  "caste_general", "marginalized_cases",
                                  "total_cases_registered"]]
          .sum()
          .assign(
              sc_pct          = lambda x: _safe_pct(x["caste_sc"],          x["total_cases_registered"]),
              st_pct          = lambda x: _safe_pct(x["caste_st"],          x["total_cases_registered"]),
              obc_pct         = lambda x: _safe_pct(x["caste_obc"],         x["total_cases_registered"]),
              general_pct     = lambda x: _safe_pct(x["caste_general"],     x["total_cases_registered"]),
              marginalized_pct= lambda x: _safe_pct(x["marginalized_cases"],x["total_cases_registered"]),
          )
          .sort_values("marginalized_pct", ascending=False)
    )

    by_year = (
        df.groupby("financial_year")[["caste_sc", "caste_st", "caste_obc",
                                      "caste_general", "total_cases_registered"]]
          .sum()
          .assign(
              sc_pct  = lambda x: _safe_pct(x["caste_sc"],  x["total_cases_registered"]),
              st_pct  = lambda x: _safe_pct(x["caste_st"],  x["total_cases_registered"]),
              obc_pct = lambda x: _safe_pct(x["caste_obc"], x["total_cases_registered"]),
          )
    )

    underserved = df[~df["meets_equity_threshold"]][
        ["state_name", "district_name", "financial_year",
         "marginalized_pct", "total_cases_registered"]
    ].sort_values("marginalized_pct")

    # Gini coefficient for inequality across district-level marginalized_pct
    vals = df["marginalized_pct"].dropna().values
    gini = _gini(vals) if len(vals) > 1 else 0.0

    logger.info("Caste analysis complete. Underserved districts: %d | Gini: %.4f",
                len(underserved), gini)
    return {
        "national": national,
        "by_state": by_state,
        "by_year": by_year,
        "underserved_districts": underserved,
        "gini_coefficient": round(gini, 4),
    }


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient for a 1-D array (0 = perfect equality)."""
    arr = np.sort(arr.astype(float))
    n = len(arr)
    cumulative = np.cumsum(arr)
    return float((n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n)


# ---------------------------------------------------------------------------
# 3. Geographic / Regional Analysis
# ---------------------------------------------------------------------------

def regional_disparity(df: pd.DataFrame) -> dict:
    """
    State-level and district-level geographic analysis normalised by CSC count.

    Returns
    -------
    dict with keys:
        state_summary   – DataFrame with per-state aggregated metrics
        district_top20  – top 20 districts by cases_per_csc
        district_bot20  – bottom 20 districts by cases_per_csc
        state_advice_rate – DataFrame sorted by advice_rate_pct
        outlier_districts – districts with cases_per_csc > 2 SD from mean
    """
    min_csc = int(__import__("os").getenv("MIN_CSC_COUNT", 1))
    df_valid = df[df["total_cscs"] >= min_csc].copy()

    state_summary = (
        df_valid.groupby("state_name")
          .agg(
              total_cscs              = ("total_cscs",              "sum"),
              total_cases             = ("total_cases_registered",  "sum"),
              total_advised           = ("cases_advised",           "sum"),
              total_districts         = ("district_name",           "nunique"),
              avg_cases_per_csc       = ("cases_per_csc",           "mean"),
              avg_marginalized_pct    = ("marginalized_pct",        "mean"),
              avg_female_pct          = ("female_pct",              "mean"),
          )
          .assign(
              advice_rate_pct = lambda x: _safe_pct(x["total_advised"], x["total_cases"]),
          )
          .round(2)
          .sort_values("total_cases", ascending=False)
    )

    district_agg = (
        df_valid.groupby(["state_name", "district_name"])
          .agg(
              total_cscs    = ("total_cscs",             "sum"),
              total_cases   = ("total_cases_registered", "sum"),
              cases_per_csc = ("cases_per_csc",          "mean"),
              marginalized_pct = ("marginalized_pct",    "mean"),
              female_pct    = ("female_pct",             "mean"),
          )
          .round(2)
          .reset_index()
    )

    mean_cpc = district_agg["cases_per_csc"].mean()
    std_cpc  = district_agg["cases_per_csc"].std()

    outliers = district_agg[
        np.abs(district_agg["cases_per_csc"] - mean_cpc) > 2 * std_cpc
    ].copy()

    logger.info("Regional analysis complete. States: %d | Districts: %d | Outliers: %d",
                len(state_summary), len(district_agg), len(outliers))

    return {
        "state_summary":      state_summary,
        "district_top20":     district_agg.nlargest(20,  "cases_per_csc"),
        "district_bot20":     district_agg.nsmallest(20, "cases_per_csc"),
        "state_advice_rate":  state_summary[["advice_rate_pct"]].sort_values("advice_rate_pct"),
        "outlier_districts":  outliers,
    }


# ---------------------------------------------------------------------------
# 4. Trend Analysis
# ---------------------------------------------------------------------------

def trend_analysis(df: pd.DataFrame) -> dict:
    """
    Year-over-year trend in registrations, advice delivery, and equity metrics.

    Returns
    -------
    dict with yearly aggregates and growth rates.
    """
    yearly = (
        df.groupby("financial_year")
          .agg(
              total_cases        = ("total_cases_registered", "sum"),
              total_advised      = ("cases_advised",          "sum"),
              total_cscs         = ("total_cscs",             "sum"),
              marginalized_cases = ("marginalized_cases",     "sum"),
              gender_female      = ("gender_female",          "sum"),
          )
          .assign(
              advice_rate_pct    = lambda x: _safe_pct(x["total_advised"],      x["total_cases"]),
              marginalized_pct   = lambda x: _safe_pct(x["marginalized_cases"], x["total_cases"]),
              female_pct         = lambda x: _safe_pct(x["gender_female"],      x["total_cases"]),
          )
          .sort_index()
    )

    yearly["cases_yoy_growth_pct"] = yearly["total_cases"].pct_change() * 100
    yearly["advice_yoy_growth_pct"] = yearly["total_advised"].pct_change() * 100

    return {"yearly": yearly.round(2)}


# ---------------------------------------------------------------------------
# 5. Composite Equity Score
# ---------------------------------------------------------------------------

def compute_equity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite Equity Index (0–100) per district aggregated over all years.

    Components (equal weight):
      1. marginalized_pct      (higher = more equitable)
      2. female_pct            (higher = more equitable)
      3. advice_rate_pct       (higher = more equitable)
      4. cases_per_csc (inverted variance / normalised access density)

    Returns
    -------
    pd.DataFrame with equity_score column, sorted descending.
    """
    agg = (
        df.groupby(["state_name", "district_name"])
          .agg(
              marginalized_pct = ("marginalized_pct", "mean"),
              female_pct       = ("female_pct",       "mean"),
              advice_rate_pct  = ("advice_rate_pct",  "mean"),
              cases_per_csc    = ("cases_per_csc",    "mean"),
              total_cases      = ("total_cases_registered", "sum"),
          )
          .reset_index()
          .dropna(subset=["marginalized_pct", "female_pct",
                          "advice_rate_pct", "cases_per_csc"])
    )

    def _minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng * 100 if rng else pd.Series(50.0, index=s.index)

    agg["score_marginalized"] = _minmax(agg["marginalized_pct"])
    agg["score_female"]       = _minmax(agg["female_pct"])
    agg["score_advice"]       = _minmax(agg["advice_rate_pct"])
    agg["score_access"]       = _minmax(agg["cases_per_csc"])

    agg["equity_score"] = (
        agg["score_marginalized"] * 0.35 +
        agg["score_female"]       * 0.25 +
        agg["score_advice"]       * 0.25 +
        agg["score_access"]       * 0.15
    ).round(2)

    logger.info("Equity scores computed for %d districts", len(agg))
    return agg.sort_values("equity_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Statistical Correlation
# ---------------------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix among key equity indicators.
    """
    cols = [
        "total_cscs", "total_cases_registered", "cases_per_csc",
        "female_pct", "marginalized_pct", "advice_rate_pct",
        "caste_sc", "caste_st", "caste_obc",
    ]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr(method="pearson").round(3)
    logger.info("Correlation matrix computed (%d × %d)", len(available), len(available))
    return corr


# ---------------------------------------------------------------------------
# 7. Full pipeline runner
# ---------------------------------------------------------------------------

def run_full_analysis(df: pd.DataFrame) -> dict:
    """
    Run the complete analysis pipeline and return all results in a single dict.
    """
    logger.info("Starting full analysis pipeline …")
    results = {
        "gender":    gender_disparity(df),
        "caste":     caste_disparity(df),
        "regional":  regional_disparity(df),
        "trend":     trend_analysis(df),
        "equity":    compute_equity_score(df),
        "correlation": correlation_analysis(df),
    }
    logger.info("Full analysis pipeline complete.")
    return results
