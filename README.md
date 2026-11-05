# Tele-Law Access Equity Analysis

> **Problem Statement 37** — Analyzing Demographic and Regional Disparities in Tele-Law Case Registrations for Inclusive Legal Access

A data-driven analytics platform powered by **IBM Cloud Lite** and **IBM Granite AI** (via IBM watsonx.ai) that uncovers gender-wise, caste-wise, and geographic disparities in Tele-Law service utilization across CSCs (Common Service Centres) — FY 2021-22 to 2024-25.

---

## 🏗️ Project Structure

```
Tele-Law Access Equity Analysis/
├── .env                        ← IBM Cloud + data.gov.in credentials (DO NOT COMMIT)
├── env.example                 ← .env template (safe to commit)
├── main.py                     ← Pipeline entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py       ← data.gov.in API fetch + CSV loader + preprocessing
│   ├── analysis.py             ← Gender / caste / regional / equity / trend analysis
│   ├── granite_ai.py           ← IBM Granite AI insight generation (watsonx.ai)
│   └── visualisation.py        ← Charts (matplotlib / seaborn) + HTML report
├── data/
│   ├── raw/                    ← Downloaded / fetched raw CSV files
│   └── processed/              ← Cleaned, analysis-ready CSV
├── reports/
│   ├── charts/                 ← PNG charts (auto-generated)
│   ├── equity_report.html      ← Full HTML equity report (auto-generated)
│   └── pipeline.log
├── notebooks/                  ← Jupyter exploration notebooks
└── tests/                      ← Unit tests
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd "Tele-Law Access Equity Analysis"
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the template and fill in your credentials:

```bash
copy env.example .env
```

Open `.env` and replace the placeholder values:

| Variable | Where to get it |
|---|---|
| `IBM_CLOUD_API_KEY` | [IBM Cloud Console → Manage → Access (IAM) → API keys](https://cloud.ibm.com/iam/apikeys) |
| `WATSONX_PROJECT_ID` | [IBM watsonx.ai → Projects → your project → Manage → General](https://dataplatform.cloud.ibm.com/) |
| `WATSONX_URL` | Region-specific URL shown in your watsonx.ai project settings |
| `DATAGOV_RESOURCE_ID` | From the dataset URL on data.gov.in |
| `DATAGOV_API_KEY` | [data.gov.in → My Account → API keys](https://data.gov.in/) |

### 3. (Optional) Download dataset manually

Visit the dataset page and download as CSV:
```
https://www.data.gov.in/resource/district-wise-tele-law-case-registration-and-advice-enabled-data-fy-2021-22-2024-25
```
Save to `data/raw/telelaw_data.csv`.

---

## 🚀 Running the Pipeline

```bash
# Option A – Demo mode (synthetic data, no download needed)
python main.py --demo

# Option B – Local CSV file
python main.py --csv data/raw/telelaw_data.csv

# Option C – Live fetch from data.gov.in API
python main.py --fetch

# Run without IBM Granite AI (no API key required)
python main.py --demo --no-ai
```

Output files are written to `reports/`:
- `equity_report.html` — full interactive HTML report
- `charts/*.png` — individual chart images
- `pipeline.log` — execution log

---

## 📊 Analysis Modules

| Module | What it does |
|---|---|
| **Gender Disparity** | National and state-level male/female/other breakdown; identifies districts with <30% female participation |
| **Caste/Marginalization** | SC/ST/OBC vs General representation; Gini coefficient; underserved district flagging |
| **Regional Disparity** | Cases-per-CSC normalized comparison; top/bottom 20 districts; state advice delivery rates |
| **Trend Analysis** | Year-over-year growth in registrations, advice rates, and equity metrics |
| **Composite Equity Index** | District-level score (0–100) combining marginalized share, female participation, advice rate, and CSC density |
| **Correlation Analysis** | Pearson correlation matrix across all key equity indicators |

---

## 🤖 IBM Granite AI Integration

The project uses **IBM Granite 13B Instruct** (or `ibm/granite-3-8b-instruct` for Lite quota) via the **watsonx.ai Text Generation API** to generate:

- **Gender equity insight** — Policy-oriented narrative on gender gaps
- **Caste equity insight** — Analysis of marginalized group representation
- **Regional equity insight** — Geographic disparity narrative
- **Executive summary** — Full policy brief for government decision-makers

Authentication uses IBM Cloud IAM with automatic token refresh.

### IBM Cloud Lite Compatibility

| Service | Lite Tier Support |
|---|---|
| IBM watsonx.ai | ✅ Free tier with token quota |
| IBM Granite 13B Instruct | ✅ Available on Lite |
| IBM Watson Studio | ✅ Free tier for notebooks |
| data.gov.in API | ✅ Free, open access |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📋 Key Findings (Expected)

- **Gender gap**: Female beneficiaries typically represent ~30–40% of cases nationally, with significant state-level variation
- **Marginalization equity**: SC+ST+OBC combined should exceed 70% in well-served districts
- **Regional disparity**: 3–5× difference in cases-per-CSC between top and bottom districts
- **Advice delivery**: National average advice rate ~75–85%; lagging states below 60%
- **Gini coefficient**: Values above 0.3 indicate significant inequality in marginalized group access

---

## 📄 Dataset

**Source**: Ministry of Electronics & Information Technology / Department of Justice  
**URL**: https://www.data.gov.in/resource/district-wise-tele-law-case-registration-and-advice-enabled-data-fy-2021-22-2024-25  
**Coverage**: FY 2021-22 to FY 2024-25 · State/District/CSC level · Gender + Caste disaggregated

---

## 🔐 Security

- Never commit `.env` to version control (it is in `.gitignore`)
- Use `env.example` as the committed template
- Rotate IBM Cloud API keys regularly via the IBM Cloud Console

---

## 🏷️ Technologies

- **IBM Cloud Lite** — Hosting and identity management
- **IBM watsonx.ai** — Granite large language model inference
- **IBM Granite AI** — `ibm/granite-13b-instruct-v2` for policy insight generation
- **Python 3.11+** — Core language
- **pandas / numpy / scipy** — Data processing and statistics
- **matplotlib / seaborn** — Visualisation
- **data.gov.in Open API** — Source dataset

---

## 🚀 Deploy on Streamlit Community Cloud (Step-by-step)

### Prerequisites
- A **GitHub account** with this repository pushed to it
- A **Streamlit Community Cloud account** → [share.streamlit.io](https://share.streamlit.io) (free)
- Your **IBM Cloud API key** and **watsonx.ai Project ID** (from [cloud.ibm.com](https://cloud.ibm.com))

---

### Step 1 — Push the repo to GitHub

```powershell
# If you haven't added a remote yet:
git remote add origin https://github.com/<your-username>/tele-law-equity.git
git branch -M main
git push -u origin main
```

Make sure `.env` and `.streamlit/secrets.toml` are **never pushed** — they are already in `.gitignore`.

---

### Step 2 — Create the app on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**
2. Fill in:
   | Field | Value |
   |---|---|
   | **Repository** | `<your-username>/tele-law-equity` |
   | **Branch** | `main` |
   | **Main file path** | `app.py` |
3. Click **Advanced settings** → **Python version** → select `3.12`
4. Click **Deploy** — Streamlit will install `requirements.txt` automatically

---

### Step 3 — Add Secrets (IBM credentials)

After deploying, go to:  
**App menu (⋮) → Settings → Secrets**

Paste the following, replacing placeholder values with your real credentials:

```toml
IBM_CLOUD_API_KEY      = "your_ibm_cloud_api_key_here"
WATSONX_PROJECT_ID     = "your_watsonx_project_id_here"
WATSONX_URL            = "https://us-south.ml.cloud.ibm.com"
GRANITE_MODEL_ID       = "ibm/granite-13b-instruct-v2"

DATAGOV_RESOURCE_ID    = "your_datagov_resource_id_here"
DATAGOV_API_KEY        = "your_datagov_api_key_here"
DATAGOV_API_BASE_URL   = "https://api.data.gov.in/resource"

EQUITY_THRESHOLD_PCT   = "70"
MIN_CSC_COUNT          = "1"
```

> The app reads secrets from `st.secrets` on Streamlit Cloud and falls back to `.env` locally — no code changes needed.

Click **Save** → the app will automatically reboot with the secrets loaded.

---

### Step 4 — Verify the deployment

Once the app is live your public URL will be:
```
https://<your-username>-tele-law-equity-app-<hash>.streamlit.app
```

1. Open the URL — the dashboard loads in **Demo mode** (no credentials needed for the UI)
2. Go to the **🤖 AI Insights** tab — the IBM Granite credential badges should show **✓ Set** (green) if secrets were saved correctly
3. Toggle **Generate AI Insights** in the sidebar to trigger IBM Granite

---

### Step 5 — Load real Tele-Law data

To use the actual dataset from data.gov.in:

**Option A — API (automatic):**  
Set `DATAGOV_RESOURCE_ID` and `DATAGOV_API_KEY` in secrets → select **"Fetch from API"** in the sidebar Data Source selector.

**Option B — Upload CSV:**  
Download the CSV from [data.gov.in](https://www.data.gov.in/resource/district-wise-tele-law-case-registration-and-advice-enabled-data-fy-2021-22-2024-25) → use the **"Upload CSV"** option in the sidebar.

---

### Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` on deploy | Check `requirements.txt` is committed and all packages are listed |
| App shows "Demo Data" only | Data source selector defaults to Demo — switch to Upload CSV or API |
| IBM Granite badge shows ✗ red | Open App Settings → Secrets and verify keys are saved without quotes issues |
| `Memory limit exceeded` | Upgrade to Streamlit Teams, or reduce dataset size in `_generate_demo_data()` |
| Slow first load | Normal — Streamlit Cloud cold-starts can take 30–60 s; subsequent loads are cached |

---
