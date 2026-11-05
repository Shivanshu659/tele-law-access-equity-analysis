"""
tests/test_analysis.py
======================
Unit tests for the analysis pipeline using synthetic data.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import clean_and_standardise
from src.analysis import (
    gender_disparity,
    caste_disparity,
    regional_disparity,
    trend_analysis,
    compute_equity_score,
    correlation_analysis,
    _gini,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic Tele-Law dataframe for testing."""
    data = {
        "state_name":            ["Uttar Pradesh", "Bihar", "Maharashtra", "Uttar Pradesh"],
        "district_name":         ["Lucknow", "Patna", "Mumbai", "Agra"],
        "financial_year":        ["2021-22", "2021-22", "2022-23", "2022-23"],
        "total_cscs":            [10, 5, 20, 8],
        "total_cases_registered":[500, 200, 800, 300],
        "cases_advised":         [400, 150, 720, 240],
        "gender_male":           [300, 130, 450, 180],
        "gender_female":         [190, 65,  340, 115],
        "gender_other":          [10,  5,   10,  5],
        "caste_general":         [100, 50,  200, 70],
        "caste_sc":              [150, 60,  220, 90],
        "caste_st":              [80,  30,  100, 50],
        "caste_obc":             [170, 60,  280, 90],
    }
    df = pd.DataFrame(data)
    return clean_and_standardise(df)


# ---------------------------------------------------------------------------
# data_ingestion tests
# ---------------------------------------------------------------------------

class TestCleanAndStandardise:
    def test_required_columns_present(self, sample_df):
        required = [
            "state_name", "district_name", "financial_year",
            "total_cscs", "total_cases_registered", "cases_advised",
            "gender_male", "gender_female", "gender_other",
            "caste_general", "caste_sc", "caste_st", "caste_obc",
        ]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_derived_columns_added(self, sample_df):
        for col in ["marginalized_cases", "marginalized_pct",
                    "female_pct", "advice_rate_pct", "cases_per_csc"]:
            assert col in sample_df.columns

    def test_marginalized_cases_sum(self, sample_df):
        expected = sample_df["caste_sc"] + sample_df["caste_st"] + sample_df["caste_obc"]
        pd.testing.assert_series_equal(
            sample_df["marginalized_cases"], expected, check_names=False
        )

    def test_no_negative_values(self, sample_df):
        for col in ["total_cscs", "total_cases_registered", "cases_advised"]:
            assert (sample_df[col] >= 0).all(), f"Negative values in {col}"

    def test_state_name_title_case(self, sample_df):
        for val in sample_df["state_name"]:
            assert val == val.title(), f"Not title case: {val}"


# ---------------------------------------------------------------------------
# analysis tests
# ---------------------------------------------------------------------------

class TestGenderDisparity:
    def test_national_keys(self, sample_df):
        result = gender_disparity(sample_df)
        assert "national" in result
        nat = result["national"]
        for key in ("male_pct", "female_pct", "other_pct", "total_cases"):
            assert key in nat

    def test_percentages_sum_to_100(self, sample_df):
        nat = gender_disparity(sample_df)["national"]
        total = nat["male_pct"] + nat["female_pct"] + nat["other_pct"]
        assert abs(total - 100.0) < 0.5, f"Gender %s sum to {total}"

    def test_by_year_indexed(self, sample_df):
        by_year = gender_disparity(sample_df)["by_year"]
        assert "female_pct" in by_year.columns
        assert len(by_year) == sample_df["financial_year"].nunique()


class TestCasteDisparity:
    def test_national_keys(self, sample_df):
        result = caste_disparity(sample_df)
        for key in ("sc_pct", "st_pct", "obc_pct", "general_pct", "marginalized_pct"):
            assert key in result["national"]

    def test_gini_between_0_and_1(self, sample_df):
        gini = caste_disparity(sample_df)["gini_coefficient"]
        assert 0.0 <= gini <= 1.0, f"Gini out of range: {gini}"

    def test_underserved_is_dataframe(self, sample_df):
        under = caste_disparity(sample_df)["underserved_districts"]
        assert isinstance(under, pd.DataFrame)


class TestRegionalDisparity:
    def test_state_summary_has_rows(self, sample_df):
        result = regional_disparity(sample_df)
        assert len(result["state_summary"]) > 0

    def test_top20_not_empty(self, sample_df):
        result = regional_disparity(sample_df)
        assert len(result["district_top20"]) > 0

    def test_advice_rate_between_0_and_100(self, sample_df):
        sr = regional_disparity(sample_df)["state_summary"]
        assert (sr["advice_rate_pct"].dropna() <= 100).all()
        assert (sr["advice_rate_pct"].dropna() >= 0).all()


class TestEquityScore:
    def test_score_in_range(self, sample_df):
        eq = compute_equity_score(sample_df)
        assert "equity_score" in eq.columns
        assert (eq["equity_score"] >= 0).all()
        assert (eq["equity_score"] <= 100).all()

    def test_sorted_descending(self, sample_df):
        eq = compute_equity_score(sample_df)
        scores = eq["equity_score"].tolist()
        assert scores == sorted(scores, reverse=True)


class TestGiniCoefficient:
    def test_perfect_equality(self):
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        assert _gini(arr) == pytest.approx(0.0, abs=1e-6)

    def test_perfect_inequality(self):
        arr = np.array([0.0, 0.0, 0.0, 100.0])
        g = _gini(arr)
        assert g > 0.5


class TestTrendAnalysis:
    def test_yearly_has_growth_columns(self, sample_df):
        t = trend_analysis(sample_df)["yearly"]
        assert "cases_yoy_growth_pct" in t.columns

    def test_yearly_row_count(self, sample_df):
        t = trend_analysis(sample_df)["yearly"]
        assert len(t) == sample_df["financial_year"].nunique()


class TestCorrelation:
    def test_is_square(self, sample_df):
        corr = correlation_analysis(sample_df)
        assert corr.shape[0] == corr.shape[1]

    def test_diagonal_is_one(self, sample_df):
        corr = correlation_analysis(sample_df)
        for col in corr.columns:
            assert corr.loc[col, col] == pytest.approx(1.0, abs=1e-6)
