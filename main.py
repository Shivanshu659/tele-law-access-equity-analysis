"""
main.py
=======
Entry point for the Tele-Law Access Equity Analysis pipeline.

Usage
-----
# With a local CSV (downloaded from data.gov.in):
  python main.py --csv data/raw/telelaw_data.csv

# Fetch directly from data.gov.in API:
  python main.py --fetch

# Skip IBM Granite AI (no API key yet):
  python main.py --csv data/raw/telelaw_data.csv --no-ai

# Generate synthetic demo data for testing:
  python main.py --demo
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reports/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("main")

# ── Lazy imports (after env loaded) ───────────────────────────────────────────
from src.data_ingestion import load_and_prepare       # noqa: E402
from src.analysis       import run_full_analysis      # noqa: E402
from src.granite_ai     import run_granite_insights   # noqa: E402
from src.visualisation  import generate_all_charts, generate_html_report  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic demo data generator
# ---------------------------------------------------------------------------

def _generate_demo_data() -> "pd.DataFrame":
    """
    Create synthetic but realistic Tele-Law data for all 28 states
    across 4 financial years without requiring a live download.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    states_districts = {
        "Uttar Pradesh":   ["Lucknow", "Agra", "Varanasi", "Kanpur", "Prayagraj"],
        "Madhya Pradesh":  ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Rewa"],
        "Bihar":           ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
        "Rajasthan":       ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
        "Maharashtra":     ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
        "West Bengal":     ["Kolkata", "Howrah", "Darjeeling", "Murshidabad", "Nadia"],
        "Odisha":          ["Bhubaneswar", "Cuttack", "Sambalpur", "Berhampur", "Koraput"],
        "Jharkhand":       ["Ranchi", "Dhanbad", "Jamshedpur", "Bokaro", "Hazaribagh"],
        "Chhattisgarh":    ["Raipur", "Bilaspur", "Durg", "Korba", "Jagdalpur"],
        "Gujarat":         ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
        "Tamil Nadu":      ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"],
        "Karnataka":       ["Bengaluru", "Mysuru", "Hubli", "Mangaluru", "Belagavi"],
        "Andhra Pradesh":  ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Kurnool"],
        "Telangana":       ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
        "Kerala":          ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
    }
    years = ["2021-22", "2022-23", "2023-24", "2024-25"]
    rows = []
    for yr_i, yr in enumerate(years):
        for state, districts in states_districts.items():
            for district in districts:
                cscs         = rng.integers(5, 50)
                total        = rng.integers(200, 4000)
                advised      = int(total * rng.uniform(0.55, 0.97))
                sc_pct       = rng.uniform(0.15, 0.35)
                st_pct       = rng.uniform(0.05, 0.25)
                obc_pct      = rng.uniform(0.15, 0.40)
                general_pct  = max(0, 1 - sc_pct - st_pct - obc_pct)
                female_pct   = rng.uniform(0.20, 0.55)
                male_pct     = 1 - female_pct - rng.uniform(0, 0.02)
                rows.append({
                    "state_name":           state,
                    "district_name":        district,
                    "financial_year":       yr,
                    "total_cscs":           int(cscs),
                    "total_cases_registered": int(total),
                    "cases_advised":        int(advised),
                    "gender_male":          int(total * male_pct),
                    "gender_female":        int(total * female_pct),
                    "gender_other":         max(0, total - int(total * male_pct) - int(total * female_pct)),
                    "caste_sc":             int(total * sc_pct),
                    "caste_st":             int(total * st_pct),
                    "caste_obc":            int(total * obc_pct),
                    "caste_general":        int(total * general_pct),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str | None, fetch: bool, use_ai: bool, demo: bool) -> None:
    import pandas as pd
    from src.data_ingestion import clean_and_standardise

    logger.info("=" * 60)
    logger.info("Tele-Law Access Equity Analysis – Pipeline Starting")
    logger.info("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    if demo:
        logger.info("[DEMO] Generating synthetic Tele-Law dataset …")
        raw = _generate_demo_data()
        df  = clean_and_standardise(raw)
    elif fetch:
        logger.info("[API] Fetching data from data.gov.in …")
        df = load_and_prepare()
    else:
        logger.info("[CSV] Loading data from %s …", csv_path)
        df = load_and_prepare(csv_path)

    logger.info("Dataset ready: %d rows × %d columns", *df.shape)

    # ── Step 2: Analysis ─────────────────────────────────────────────────────
    logger.info("Running analytical pipeline …")
    results = run_full_analysis(df)

    # ── Step 3: IBM Granite AI insights ─────────────────────────────────────
    if use_ai:
        logger.info("Requesting IBM Granite AI insights …")
        insights = run_granite_insights(results)
    else:
        logger.info("AI insights skipped (--no-ai flag).")
        insights = {k: "[AI disabled – run without --no-ai for IBM Granite insights]"
                    for k in ("gender_insight", "caste_insight",
                               "regional_insight", "executive_summary")}

    # ── Step 4: Visualisations ───────────────────────────────────────────────
    logger.info("Generating charts …")
    Path("reports").mkdir(exist_ok=True)
    chart_paths = generate_all_charts(results, df)

    # ── Step 5: HTML Report ──────────────────────────────────────────────────
    logger.info("Generating HTML equity report …")
    report_path = generate_html_report(results, insights, chart_paths)

    # ── Step 6: Console summary ──────────────────────────────────────────────
    _print_summary(results)
    logger.info("=" * 60)
    logger.info("Pipeline complete. Report -> %s", report_path.resolve())
    logger.info("=" * 60)


def _print_summary(results: dict) -> None:
    g  = results["gender"]["national"]
    c  = results["caste"]["national"]
    gini = results["caste"]["gini_coefficient"]

    print("\n" + "=" * 60)
    print("  TELE-LAW EQUITY ANALYSIS – SUMMARY")
    print("=" * 60)
    print(f"  Total cases registered : {g['total_cases']:>10,}")
    print(f"  Female participation   : {g['female_pct']:>9.2f} %")
    print(f"  Male participation     : {g['male_pct']:>9.2f} %")
    print("-" * 60)
    print(f"  SC representation      : {c['sc_pct']:>9.2f} %")
    print(f"  ST representation      : {c['st_pct']:>9.2f} %")
    print(f"  OBC representation     : {c['obc_pct']:>9.2f} %")
    print(f"  General category       : {c['general_pct']:>9.2f} %")
    print(f"  Marginalized total     : {c['marginalized_pct']:>9.2f} %")
    print(f"  Gini coefficient       : {gini:>9.4f}")
    print("-" * 60)
    top_state = results["regional"]["state_summary"].index[0] \
                if len(results["regional"]["state_summary"]) else "N/A"
    eq_top    = results["equity"].iloc[0] if len(results["equity"]) else None
    if eq_top is not None:
        print(f"  Highest equity district: {eq_top['district_name']} ({eq_top['state_name']}) "
              f"– score {eq_top['equity_score']}")
    print(f"  Top state by volume    : {top_state}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tele-Law Access Equity Analysis – IBM Cloud Lite + IBM Granite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv",   metavar="PATH",
                       help="Path to downloaded Tele-Law CSV file")
    group.add_argument("--fetch", action="store_true",
                       help="Fetch data live from data.gov.in API")
    group.add_argument("--demo",  action="store_true",
                       help="Generate synthetic demo data (no download needed)")
    parser.add_argument("--no-ai", dest="no_ai", action="store_true",
                        help="Skip IBM Granite AI insight generation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not (args.csv or args.fetch or args.demo):
        print("No data source specified. Using --demo mode.\n"
              "Run `python main.py --help` for all options.\n")
        args.demo = True

    Path("reports").mkdir(exist_ok=True)

    run_pipeline(
        csv_path=args.csv,
        fetch=args.fetch,
        use_ai=not args.no_ai,
        demo=args.demo,
    )
