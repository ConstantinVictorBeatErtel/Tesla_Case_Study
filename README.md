# Project Aurora - Model Inputs & Priors

This document describes how model inputs are configured in the simulation, comparing theoretical priors with data-driven Bayesian priors.

## FRED API Configuration

The application uses the Federal Reserve Economic Data (FRED) API to fetch the current Federal Funds Rate. If the API is unavailable or fails (e.g., missing API key, network issues), the application will automatically fall back to a default value of **4.5%**.

**Optional Setup:**
- To use live FRED data, add your API key to Streamlit Cloud secrets or create a `.env` file:
  ```
  FRED_API_KEY=your_api_key_here
  ```
- Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html

**Note:** The application will run successfully even without a FRED API key by using the fallback value.

---

## 1. Raw Material Cost ($/lamp)

**Theoretical prior in `app.py` (hand-crafted):**
- Normal by country: **US** N(40, 4), **MX** N(35, 3.5), **CN** N(30, 3) — mean and std given directly.

**Data-driven Bayesian prior in `bayesian_priors.py` (what replaces it):**
- Posterior-predictive **Student-t** centered near the baseline, fit via a Normal-Inverse-Gamma posterior on 24 months of _plastics PPI_ rescaled to the baseline. (Heavier tails than Normal.)

**Real-world data source & characteristics:**
- FRED series **PCU325211325211P** ("Producer Price Index by Industry: Plastics Material and Resin Manufacturing: Primary Products")
- **Monthly**, **Index (Dec 1980=100)**, **Not SA**; source **U.S. BLS**
- Code keeps last **24 months**
- [FRED Link](https://fred.stlouisfed.org/series/PCU325211325211P?utm_source=chatgpt.com "Producer Price Index by Industry: Plastics Material and Resin ...")

---

## 2. Labor Cost ($/lamp)

**Theoretical prior in `app.py`:**
- Normal: **US** N(12, 0.6), **MX** N(8, 0.4), **CN** N(4, 0.2)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement** — stays Normal with the baseline mean/std.

---

## 3. Indirect Cost ($/lamp)

**Theoretical prior in `app.py`:**
- Normal: **US** N(10, 0.5), **MX** N(8, 0.4), **CN** N(4, 0.2)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 4. Logistics Cost ($/lamp)

**Theoretical prior in `app.py`:**
- **US** Normal N(9, 0)
- **MX** Normal N(7, 0.056)
- **CN** **Lognormal** calibrated to mean = 12, std = 8 (code converts to μ, σ for lognormal sampling)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement** — still uses the specified Normal/Lognormal by country.

---

## 5. Electricity ($/lamp)

**Theoretical prior in `app.py`:**
- Normal: **US** N(4, 0.4), **MX** N(3, 0.3), **CN** N(4, 0.4)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 6. Depreciation ($/lamp)

**Theoretical prior in `app.py`:**
- Normal: **US** N(5, 0.25), **MX** N(1, 0.05), **CN** N(5, 0.25)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 7. Working Capital ($/lamp)

**Theoretical prior in `app.py`:**
- Normal: **US** N(5, 0.5), **MX** N(6, 0.6), **CN** N(10, 1)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 8. Manufacturing Yield (0–1)

**Theoretical prior in `app.py`:**
- **Beta(a,b):** **US** Beta(79, 20) (~0.80 mean), **MX** Beta(12, 1) (~0.92), **CN** Beta(49, 3) (~0.95)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **Beta posterior** using pseudo-counts set by "uncertainty":
  - **US** _(medium)_ → total 50 ⇒ ≈Beta(40, 10)
  - **MX** _(high)_ → total 15 ⇒ ≈Beta(13.8, 1.2)
  - **CN** _(low)_ → total 100 ⇒ ≈Beta(94.2, 5.8)
- (No external API; this formalizes confidence around the baseline.)

---

## 9. FX Impact (multiplicative)

**Theoretical prior in `app.py`:**
- Multiplier **1 + Normal(0, σ)** where σ is fixed: **US** 0 (no FX), **MX** 0.08, **CN** 0.03

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **FX multiplier from returns**: take **log returns** of last **12 months** of spot rates; fit Normal-Inverse-Gamma; sample **Student-t** returns; use **1 + return** as the multiplier. Applies to **MX** and **CN** only.

**Real-world data source & characteristics:**
- **DEXMXUS** ("Mexican Pesos to One U.S. Dollar")
  - **Daily**, **Not SA**, H.10 release, **Board of Governors of the Fed**
  - Code keeps **12 months**
  - [FRED Link](https://fred.stlouisfed.org/series/DEXMXUS?utm_source=chatgpt.com "Mexican Pesos to U.S. Dollar Spot Exchange Rate (DEXMXUS)")
- **DEXCHUS** ("Chinese Yuan Renminbi to U.S. Dollar Spot Exchange Rate")
  - **Daily**, **Not SA**, H.10 release, **Board of Governors**
  - Code keeps **12 months**
  - [FRED Link](https://fred.stlouisfed.org/series/DEXCHUS?utm_source=chatgpt.com "Chinese Yuan Renminbi to U.S. Dollar Spot Exchange Rate")

---

## 10. Tariff (additive)

**Theoretical prior in `app.py`:**
- Fixed by country: **US** $0; **MX** $15.5; **CN** $15

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement** (still fixed; optional normal "escalation" added below)

---

## 11. Tariff Escalation (additive)

**Theoretical prior in `app.py`:**
- Normal: mean/stdev by country — **US** (0, 0), **MX** (0, 2), **CN** (0, 2)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 12. Currency "σ" (if no FX series)

**Theoretical prior in `app.py`:**
- As above (only used if FX fetch fails)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- If FRED fetch fails, **falls back** to 1 + Normal(0, σ)

**Real-world data source:**
- See FX rows above

---

## 13. Discrete Risks (Bernoulli add-ons)

**Theoretical prior in `app.py`:**
- **Disruption:** add $10 with prob **US** 0.05/**MX** 0.10/**CN** 0.20
- **Damage:** add $20 with prob **US** 0.01/**MX** 0.015/**CN** 0.02
- **Cancellation (CN only):** add $50 with prob 0.30

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 14. Border Delay Cost (add-on)

**Theoretical prior in `app.py`:**
- Border time ~ Normal(mean, std); cost = max(0, time − 2 hrs) × $10/hr
- **US** (0, 0), **MX** (0.83, 0.67), **CN** (0, 0)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## 15. Skills Factor (multiplicative)

**Theoretical prior in `app.py`:**
- Multiplier **1 + Normal(μ, σ)**: **US** (0, 0), **MX** (0, 0.05), **CN** (0, 0)

**Data-driven Bayesian prior in `bayesian_priors.py`:**
- **No data-driven replacement**

---

## Summary: Uncertainty Sources Comparison

| Uncertainty Type | `app.py` | `bayesian_priors/` |
|------------------|----------|-------------------|
| **Mean** | Fixed (your guess) | Uncertain (estimated ± margin) |
| **Variance** | Fixed (your guess) | Uncertain (estimated ± margin) |
| **Shape** | Fixed (Normal/Beta/Lognormal) | Adapts (Student-t has fatter tails) |
| **Distribution types** | Normal, Beta, Lognormal | **Student-t**, Beta (adjusted), Lognormal |
| **Small sample penalty** | None | Yes (wider intervals) |
| **Tail risk** | Standard | **Enhanced** (more extreme scenarios) |