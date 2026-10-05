# Financial Distress Risk Score Dashboard

An interactive web dashboard for the Business Analytics project *Validating Financial Risk Scores: Classification, Portfolio Optimization and Stress-Period Evidence from NSE-Listed Firms*.

It is only a **viewer and explainer** for results the project has already produced. It does not retrain, rebuild or replace the statistical model. The distress probability for every company comes from the project's own `scores.parquet`.

---

## 1. Files

| File | What it does |
|---|---|
| `app.py` | The whole dashboard (search, score, indicators, chart, AI Analyst, About the model). |
| `requirements.txt` | The Python packages needed. |
| `README_DASHBOARD.md` | This guide. |
| `data/` | **Unmodified copies** of existing project files (listed below). Nothing new was created. |

Files in `data/` (all copied from your project ZIP, byte for byte):

| File | Why it is needed |
|---|---|
| `scores.parquet` | The model's predicted distress probability `p`, score date `reb`, accounts year `FY`, symbol `sym` (also `EQ`, `SH`, `n_train`). **Main source.** |
| `features.parquet` | Company **names** and the **15 financial ratios**. `scores.parquet` has no names or ratios, so this file is required. |
| `t3_coef.csv` | The model's coefficients (report Table 5.3). |
| `t3_models.csv`, `t3_naive_benchmark.csv`, `t3_robust.csv`, `t3_quintiles.csv`, `stage3.json` | Model performance and reference tables, used for the "About the model" section and AI answers. |

If the `data` folder is missing, put these 8 files next to `app.py` instead. The app looks in both places.

## 2. What you see, and where each number comes from

| On the page | Source |
|---|---|
| Probability of distress | `scores.parquet`, column `p` (shown as is) |
| Score date and accounts year | `reb` and `FY` |
| Company name and symbol | `sym` plus `name` from `features.parquet` |
| Risk quintile (Q1 lowest to Q5 highest) | The project's own rule (`stage3/4/5.py`): rank the scores and split into five equal groups. Applied here to **all companies scored on the selected date**. |
| Financial indicators | The 15 predictors in `features.parquet`, for the same company and accounts year |
| Observed outcome | The project's distress label `y` (shown only where it exists in the data) |
| Headline results and "About the model" table | `t3_*.csv` result files |

**Default score date: 1 Aug 2025.** It uses FY2025 accounts and matches the year evaluated on the locked test set in the report (AUC 0.846, which these stored scores reproduce: 0.8455). The 2024 and 2026 dates can be chosen too. The 2026 outcomes are not known yet.

**Two things the app calculates (nothing else):**
1. **Quintiles and percentiles** from the stored scores, as described above.
2. **"What is moving this company's score"**: the model's coefficient × the company's standardised ratio. The standardisation is recomputed from `features.parquet` with the project's own preprocessing (winsorise 1st/99th percentile, median fill, standardise; FY2023-FY2024 labelled rows). This breakdown is **exact for 1 Aug 2025** (it reproduces the stored probability within 0.004) and **approximate for the other dates**, because the model was refit at each date and those coefficients were not saved. The page says so. It shows association, not causation.

## 3. Run it on your computer

```
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Everything except the AI Analyst works without any setup.

To turn on the AI Analyst locally, either set environment variables, or create a file `.streamlit/secrets.toml` next to `app.py` containing the three lines shown in section 4. **Never upload `secrets.toml` to GitHub.**

## 4. How the AI Analyst works

1. You select a company and type (or click) a question.
2. The app builds a **small context** for that question: the selected company's score, quintile, 15 indicators and top score drivers, plus the project's methodology, plus only the extra facts the question needs (for example model performance for "How accurate is the model?", documented limitations for "limitations", portfolio or stress results if asked). Other companies' data and the full dataset are **not** sent.
3. The context and your question go to a language model with the project's system instructions (explain the existing model; don't invent numbers; don't give investment advice; say "not available in the dashboard" when information is missing; decline unrelated questions).
4. The answer is shown in the chat. The probability itself always comes from the project's model, not the AI.

Extra safeguards: requests for investment advice (buy/sell, price targets) are answered with a fixed message without calling the AI; each browser session is limited to 25 questions (change with `MAX_QUESTIONS_PER_SESSION`); the chat resets when you change company.

**If the AI is unavailable** (no key, no internet, service error) the page shows *"AI Analyst is currently unavailable. The financial risk score and model results remain available."* and everything else keeps working.

**Settings** (Streamlit secrets or environment variables; the key is never in the code):

```toml
LLM_PROVIDER = "anthropic"                    # or "openai" (any OpenAI-compatible service)
LLM_API_KEY  = "paste-your-key-here"
LLM_MODEL    = "claude-haiku-4-5-20251001"    # optional for anthropic; required for openai
# LLM_BASE_URL = "https://..."                # optional: another OpenAI-compatible provider
# MAX_QUESTIONS_PER_SESSION = "25"            # optional
```

To switch provider later, change only these settings. If the AI shows an HTTP error such as 404 or 401, check the key and the model name in your provider's documentation. An API key normally needs a funded account; check your provider's current pricing.

## 5. What was tested

Tested on this finished version:
- The app starts without errors; `scores.parquet` and `features.parquet` load; no hard-coded API key or company results.
- Five companies (lowest score, median, highest score, Reliance, TCS) match values computed independently from the data files, and two other score dates were checked. All 5,638 company-date combinations process without error, including companies with missing ratios and negative equity.
- **Company search in a real browser:** by name ("tata cons"), partial name ("reliance ind") and symbol ("infy"); an unknown name gives "No results". Changing the score date keeps the selected company.
- Risk probability, quintile badge, chart and all 15 indicator cards display for every company tested.
- **No API key:** the whole dashboard works; only the AI Analyst shows *"AI Analyst is currently unavailable..."* and its input is disabled.
- **AI Analyst plumbing**, tested against a **mock** AI server (31 checks): request format for both providers, only the needed context is sent, other companies' data is not sent, service errors and bad replies are handled, the 25-question limit works, the chat resets when the company changes. Through the real browser UI, *"Why is this company's risk at this level?"* and *"How accurate is the model?"* reach the AI, and *"Should I buy this stock?"* is declined with a fixed message without calling the AI.
- Runs under pandas 2.3.3 and 3.0.2.

**Not tested:** calls to a live AI service (no API key was available). The wording and quality of real AI answers, and how it handles unrelated or tricky questions, are **unverified**; try the questions in section 6 yourself before the demo. The default model name has not been checked against a live service; set `LLM_MODEL` if you get an error. The dashboard has not been deployed; it runs locally only.

## 6. Live demo checklist

- [ ] Open the dashboard link a few minutes early so the app is awake; check the AI Analyst is not showing "unavailable".
- [ ] Search a company by name (for example Reliance Industries). Point out the probability, the quintile badge and the chart.
- [ ] Show the financial indicators, then "What is moving this company's score".
- [ ] Ask: *"Why is this company's risk at this level?"*, then a model question such as *"How accurate is the model?"*
- [ ] Try a high-risk company and a low-risk one, to show the contrast.
- [ ] Expect questions on: what the model predicts (next-year distress, accounting-based label), why the date is 1 Aug 2025, how quintiles are defined, and that the AI explains the model but does not predict.
- [ ] Before the demo, also test: *"Should I buy this stock?"* (should decline), *"Who won the cricket match?"* (should say it only answers project questions), *"What is this company's CEO's name?"* (should say not available).
- [ ] Fallback if the AI or internet fails: the score, chart, indicators and About the model still work.
