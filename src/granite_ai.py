"""
granite_ai.py
=============
IBM Granite AI integration via IBM watsonx.ai (IBM Cloud Lite tier).

Provides:
  - IBM IAM token management
  - Text generation via ibm/granite-13b-instruct-v2 (or configurable model)
  - Pre-built prompt templates for Tele-Law equity insights
  - Structured insight generation from analysis results
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _secret(key: str, default: str = "") -> str:
    """
    Read a secret from st.secrets (Streamlit Cloud) when available,
    otherwise fall back to os.environ / .env.
    """
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


# ── Config ────────────────────────────────────────────────────────────────────
IBM_CLOUD_API_KEY  = _secret("IBM_CLOUD_API_KEY")
WATSONX_URL        = _secret("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = _secret("WATSONX_PROJECT_ID")
GRANITE_MODEL_ID   = _secret("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
WATSONX_TEXT_URL = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"

# ── Token cache ───────────────────────────────────────────────────────────────
_token_cache: dict = {"token": None, "expires_at": datetime.min}


# ---------------------------------------------------------------------------
# IBM IAM Authentication
# ---------------------------------------------------------------------------

def _get_iam_token() -> str:
    """
    Obtain (or return cached) IBM IAM bearer token using the API key.
    Token is cached and refreshed 5 minutes before expiry.
    """
    global _token_cache
    if _token_cache["token"] and datetime.utcnow() < _token_cache["expires_at"]:
        return _token_cache["token"]

    if not IBM_CLOUD_API_KEY or IBM_CLOUD_API_KEY == "your_ibm_cloud_api_key_here":
        raise EnvironmentError(
            "IBM_CLOUD_API_KEY is not configured. "
            "Update your .env file with a valid IBM Cloud API key."
        )

    resp = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey":     IBM_CLOUD_API_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"]      = data["access_token"]
    _token_cache["expires_at"] = datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600) - 300)
    logger.debug("IBM IAM token refreshed.")
    return _token_cache["token"]


# ---------------------------------------------------------------------------
# Core generation wrapper
# ---------------------------------------------------------------------------

def generate_text(prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.3) -> str:
    """
    Send a prompt to IBM Granite via watsonx.ai and return generated text.

    Parameters
    ----------
    prompt          : str   Full prompt string.
    max_new_tokens  : int   Maximum tokens in the response.
    temperature     : float Sampling temperature (lower = more deterministic).

    Returns
    -------
    str  Generated text from Granite.
    """
    if not WATSONX_PROJECT_ID or WATSONX_PROJECT_ID == "your_watsonx_project_id_here":
        raise EnvironmentError(
            "WATSONX_PROJECT_ID is not configured. Update .env."
        )

    token = _get_iam_token()
    payload = {
        "model_id":   GRANITE_MODEL_ID,
        "project_id": WATSONX_PROJECT_ID,
        "input":      prompt,
        "parameters": {
            "decoding_method":  "greedy" if temperature == 0 else "sample",
            "max_new_tokens":   max_new_tokens,
            "temperature":      temperature,
            "repetition_penalty": 1.1,
            "stop_sequences":   ["\n\n---"],
        },
    }

    resp = requests.post(
        WATSONX_TEXT_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    return result["results"][0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

def _fmt_dict(d: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in d.items())


def build_gender_insight_prompt(national: dict, skewed_count: int) -> str:
    return f"""You are a legal-access equity analyst for India's Tele-Law program.
Below is a summary of gender-wise case registration data:

{_fmt_dict(national)}
Number of districts where female participation < 30%: {skewed_count}

Task: Write a concise policy-oriented insight paragraph (4-6 sentences) that:
1. Identifies the gender gap in Tele-Law access.
2. Highlights districts with critically low female participation.
3. Recommends targeted interventions to improve female outreach.

Insight:"""


def build_caste_insight_prompt(national: dict, gini: float, underserved_count: int) -> str:
    return f"""You are a legal-access equity analyst for India's Tele-Law program.
Below is a summary of caste-wise case registration data:

{_fmt_dict(national)}
Gini coefficient of marginalized group access across districts: {gini}
Districts not meeting the {os.getenv('EQUITY_THRESHOLD_PCT', 70)}% marginalized threshold: {underserved_count}

Task: Write a concise policy-oriented insight paragraph (4-6 sentences) that:
1. Evaluates the representation of SC, ST, OBC beneficiaries.
2. Interprets the Gini coefficient in context of equity.
3. Recommends policy actions to improve outreach to underserved communities.

Insight:"""


def build_regional_insight_prompt(top_state: str, bot_state: str,
                                  advice_rate_national: float) -> str:
    return f"""You are a legal-access equity analyst for India's Tele-Law program.
Below is a summary of regional case registration disparities:

Highest performing state: {top_state}
Lowest performing state: {bot_state}
National average advice delivery rate: {advice_rate_national}%

Task: Write a concise policy-oriented insight paragraph (4-6 sentences) that:
1. Describes the geographic disparity in Tele-Law utilization.
2. Identifies possible reasons for under-performance in lagging states.
3. Suggests strategies to bridge the regional gap and improve CSC effectiveness.

Insight:"""


def build_executive_summary_prompt(findings: dict) -> str:
    lines = []
    for section, content in findings.items():
        lines.append(f"[{section.upper()}]")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {content}")
        lines.append("")

    findings_text = "\n".join(lines)
    return f"""You are a senior policy advisor reviewing India's Tele-Law equity analysis.
The following findings summarise key disparities in legal aid access:

{findings_text}

Task: Write an executive summary (6-8 sentences) suitable for a government policy brief that:
1. States the overall equity situation in Tele-Law access.
2. Highlights the three most critical disparities found.
3. Proposes three actionable, evidence-based recommendations.
4. Concludes with the importance of data-driven monitoring.

Executive Summary:"""


# ---------------------------------------------------------------------------
# High-level insight functions
# ---------------------------------------------------------------------------

def generate_gender_insight(gender_results: dict) -> str:
    """Generate Granite-powered gender equity insight."""
    prompt = build_gender_insight_prompt(
        national=gender_results["national"],
        skewed_count=len(gender_results["skewed_districts"]),
    )
    logger.info("Requesting Granite gender insight …")
    return generate_text(prompt, max_new_tokens=300)


def generate_caste_insight(caste_results: dict) -> str:
    """Generate Granite-powered caste equity insight."""
    prompt = build_caste_insight_prompt(
        national=caste_results["national"],
        gini=caste_results["gini_coefficient"],
        underserved_count=len(caste_results["underserved_districts"]),
    )
    logger.info("Requesting Granite caste insight …")
    return generate_text(prompt, max_new_tokens=350)


def generate_regional_insight(regional_results: dict) -> str:
    """Generate Granite-powered regional equity insight."""
    state_summary = regional_results["state_summary"]
    top_state = state_summary.index[0]  if len(state_summary) > 0 else "N/A"
    bot_state = state_summary.index[-1] if len(state_summary) > 0 else "N/A"
    national_advice_rate = round(regional_results["state_summary"]["advice_rate_pct"].mean(), 2)

    prompt = build_regional_insight_prompt(top_state, bot_state, national_advice_rate)
    logger.info("Requesting Granite regional insight …")
    return generate_text(prompt, max_new_tokens=350)


def generate_executive_summary(all_results: dict) -> str:
    """
    Generate a full executive summary using Granite by distilling key findings
    from all analysis modules.
    """
    findings = {
        "Gender Equity": all_results["gender"]["national"],
        "Caste Equity":  all_results["caste"]["national"],
        "Gini Index":    all_results["caste"]["gini_coefficient"],
        "Trend (latest year)": (
            all_results["trend"]["yearly"].iloc[-1].to_dict()
            if not all_results["trend"]["yearly"].empty else {}
        ),
    }
    prompt = build_executive_summary_prompt(findings)
    logger.info("Requesting Granite executive summary …")
    return generate_text(prompt, max_new_tokens=500, temperature=0.4)


def run_granite_insights(all_results: dict) -> dict:
    """
    Run all Granite AI insight generators and return a dict of insights.
    Falls back gracefully if IBM credentials are not configured.
    """
    insights: dict = {}
    generators = {
        "gender_insight":     lambda: generate_gender_insight(all_results["gender"]),
        "caste_insight":      lambda: generate_caste_insight(all_results["caste"]),
        "regional_insight":   lambda: generate_regional_insight(all_results["regional"]),
        "executive_summary":  lambda: generate_executive_summary(all_results),
    }

    for name, fn in generators.items():
        try:
            insights[name] = fn()
            logger.info("✓ %s generated", name)
        except EnvironmentError as exc:
            insights[name] = f"[IBM Granite not configured – {exc}]"
            logger.warning("Skipped %s: %s", name, exc)
        except Exception as exc:                          # noqa: BLE001
            insights[name] = f"[Generation failed – {exc}]"
            logger.error("Error in %s: %s", name, exc)

    return insights
