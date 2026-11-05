"""
data_ingestion.py
=================
Handles data fetching from data.gov.in API and local CSV loading,
followed by cleaning and standardisation of the Tele-Law dataset.

Columns expected in the raw dataset
------------------------------------
state_name, district_name, financial_year,
total_cscs, total_cases_registered, cases_advised,
gender_male, gender_female, gender_other,
caste_general, caste_sc, caste_st, caste_obc
"""

import os
import logging
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))

# Canonical column mapping from data.gov.in headers → internal names
COLUMN_MAP = {
    "state_name":               "state_name",
    "state":                    "state_name",
    "district_name":            "district_name",
    "district":                 "district_name",
    "financial_year":           "financial_year",
    "year":                     "financial_year",
    "no_of_cscs":               "total_cscs",
    "total_cscs":               "total_cscs",
    "cases_registered":         "total_cases_registered",
    "total_cases_registered":   "total_cases_registered",
    "cases_advised":            "cases_advised",
    "advice_enabled":           "cases_advised",
    "male":                     "gender_male",
    "gender_male":              "gender_male",
    "female":                   "gender_female",
    "gender_female":            "gender_female",
    "other":                    "gender_other",
    "gender_other":             "gender_other",
    "general":                  "caste_general",
    "caste_general":            "caste_general",
    "sc":                       "caste_sc",
    "caste_sc":                 "caste_sc",
    "st":                       "caste_st",
    "caste_st":                 "caste_st",
    "obc":                      "caste_obc",
    "caste_obc":                "caste_obc",
}

REQUIRED_COLUMNS = [
    "state_name", "district_name", "financial_year",
    "total_cscs", "total_cases_registered", "cases_advised",
    "gender_male", "gender_female", "gender_other",
    "caste_general", "caste_sc", "caste_st", "caste_obc",
]

NUMERIC_COLUMNS = [
    "total_cscs", "total_cases_registered", "cases_advised",
    "gender_male", "gender_female", "gender_other",
    "caste_general", "caste_sc", "caste_st", "caste_obc",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_from_api(resource_id: str | None = None, api_key: str | None = None,
                   limit: int = 5000) -> pd.DataFrame:
    """
    Fetch raw Tele-Law dataset from data.gov.in Open Government Data API.

    Parameters
    ----------
    resource_id : str, optional
        Dataset resource ID.  Falls back to DATAGOV_RESOURCE_ID env var.
    api_key : str, optional
        API key.  Falls back to DATAGOV_API_KEY env var.
    limit : int
        Records per page (max 5 000 for free tier).

    Returns
    -------
    pd.DataFrame  Raw records from the API.
    """
    resource_id = resource_id or os.getenv("DATAGOV_RESOURCE_ID")
    api_key = api_key or os.getenv("DATAGOV_API_KEY")
    base_url = os.getenv("DATAGOV_API_BASE_URL", "https://api.data.gov.in/resource")

    if not resource_id:
        raise ValueError(
            "DATAGOV_RESOURCE_ID is not set. "
            "Provide it via .env or as a function argument."
        )

    params = {
        "api-key": api_key or "579b464db66ec23bdd000001cdd3946e44ce4aad38d82be7ae12345",
        "format": "json",
        "limit": limit,
        "offset": 0,
    }
    url = f"{base_url}/{resource_id}"
    all_records: list[dict] = []

    logger.info("Fetching Tele-Law data from data.gov.in …")
    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("records", [])
        all_records.extend(records)
        total = payload.get("total", 0)
        params["offset"] += limit
        logger.info("  fetched %d / %d records", len(all_records), total)
        if len(all_records) >= int(total):
            break

    df = pd.DataFrame(all_records)
    raw_path = RAW_DIR / "telelaw_raw_api.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    logger.info("Raw data saved → %s", raw_path)
    return df


def load_from_csv(path: str | Path) -> pd.DataFrame:
    """
    Load Tele-Law dataset from a local CSV file (downloaded manually
    from data.gov.in or exported from IBM Watson Studio).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    logger.info("Loading CSV from %s", path)
    return pd.read_csv(path, low_memory=False)


def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names, coerce numeric types, fill missing values,
    and add derived equity columns.

    Returns
    -------
    pd.DataFrame  Cleaned, analysis-ready dataframe.
    """
    # ── 1. Normalise column names ────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns},
              inplace=True)

    # ── 2. Add missing columns as 0 ─────────────────────────────────────────
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger.warning("Column '%s' missing – filling with 0", col)
            df[col] = 0

    df = df[REQUIRED_COLUMNS].copy()

    # ── 3. Coerce numerics ───────────────────────────────────────────────────
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── 4. Clean string columns ──────────────────────────────────────────────
    df["state_name"] = df["state_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].str.strip().str.title()
    df["financial_year"] = df["financial_year"].str.strip()

    # ── 5. Derived columns ───────────────────────────────────────────────────
    df["marginalized_cases"] = df["caste_sc"] + df["caste_st"] + df["caste_obc"]
    df["marginalized_pct"] = (
        df["marginalized_cases"] / df["total_cases_registered"].replace(0, pd.NA) * 100
    ).round(2)

    df["female_pct"] = (
        df["gender_female"] / df["total_cases_registered"].replace(0, pd.NA) * 100
    ).round(2)

    df["advice_rate_pct"] = (
        df["cases_advised"] / df["total_cases_registered"].replace(0, pd.NA) * 100
    ).round(2)

    df["cases_per_csc"] = (
        df["total_cases_registered"] / df["total_cscs"].replace(0, pd.NA)
    ).round(2)

    # ── 6. Equity flag ───────────────────────────────────────────────────────
    threshold = float(os.getenv("EQUITY_THRESHOLD_PCT", 70))
    df["meets_equity_threshold"] = df["marginalized_pct"] >= threshold

    logger.info("Dataset cleaned: %d rows × %d columns", *df.shape)
    return df


def load_and_prepare(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Convenience wrapper: load from CSV if a path is given, otherwise attempt
    API fetch.  Always runs clean_and_standardise before returning.
    """
    if csv_path:
        raw = load_from_csv(csv_path)
    else:
        raw = fetch_from_api()

    df = clean_and_standardise(raw)
    out_path = PROCESSED_DIR / "telelaw_clean.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Processed data saved -> %s", out_path)
    return df
