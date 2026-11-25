# SWYNEX-Final-Data-Science-Project

**Customer churn prediction for a telecom operator: an end-to-end data-science case study.**
Task 4 (final project) of the SWYNEX Technologies Data Science Internship. It combines the problem statement, data preparation, exploratory analysis, modeling, and conclusions into one reproducible, documented project.

**Author:** Karthik · **Dataset:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (7,043 customers, 21 columns)

---

## 1. Problem statement
A telecom operator loses revenue every time a customer leaves, and retention offers cost money. The business needs to answer two questions:

1. **Who is likely to churn?** A model that ranks customers by churn risk.
2. **Why do they churn?** Drivers the business can act on.

About **26.5%** of customers churn, so accuracy is misleading (predicting "no churn" for everyone is 73.5% accurate). Success is measured with **PR-AUC** (primary), ROC-AUC, recall/precision at a chosen threshold, and **expected net retention benefit** under an explicit cost model.

## 2. Pipeline at a glance
| Stage | What was done | Where |
|---|---|---|
| Data preparation | Found 11 hidden blanks in `TotalCharges` (all tenure-0 customers), imputed `0` with a safeguard, merged redundant levels, typed columns, 9/9 validation checks, 9 unit tests | `src/`, `tests/`, `notebooks/01_data_preparation.ipynb`, `reports/data_quality_report.md` |
| EDA | Target balance, numeric and categorical drivers, statistical tests (Mann-Whitney, chi-square/Cramér's V), interactions, a high-risk segment | `notebooks/02_eda.ipynb` |
| Modeling | Dummy, Logistic Regression, Random Forest, Gradient Boosting; 5-fold CV; tuning; held-out test with bootstrap CIs; calibration; cost-based threshold; lift; permutation importance; segment error analysis | `notebooks/03_modeling_evaluation.ipynb` |
| Conclusions | Findings, recommendations, limitations, next steps | `reports/final_report.md` |

## 3. Key results

**What drives churn (EDA)**
- **Contract is the strongest driver:** month-to-month 42.7% churn vs two-year 2.8%.
- **Churn is front-loaded:** 47% of first-year customers churn; **55% of all churners leave within 12 months**.
- **Fiber optic customers churn most** (41.9% vs 19.0% DSL) despite paying the most; **electronic-check payers** churn at 45.3%.
- **Online security and tech support** go with ~27 points lower churn among internet customers (association, not proof of cause).
- **Gender and phone service carry no signal.**
- One segment (month-to-month + fiber + electronic check) is **18.6% of customers but 42% of churners**, with 60% churn.

**How well can we predict it (held-out test set, n = 1,409)**

| Metric | Value | 95% CI |
|---|---|---|
| PR-AUC (primary) | **0.662** | 0.609 – 0.713 |
| ROC-AUC | **0.846** | 0.823 – 0.868 |
| Brier score | 0.136 | – |

- **Model:** tuned Gradient Boosting (CV PR-AUC 0.667), against a dummy baseline of 0.265. Logistic Regression is statistically close (0.662) and a transparent alternative.
- **Targeting:** the top 10% of customers by predicted risk hold **29% of all churners** (2.9x lift).
- **Operating point:** cost-optimised threshold **0.24** gives **recall 0.83, precision 0.50**, contacting 618 of 1,409 customers.
- **Business value (illustrative assumptions):** net benefit **$51.7k** vs $39.3k at the default 0.5 threshold and $27.5k for contacting everyone.
- **Main model drivers (permutation importance):** tenure, contract, internet service, consistent with the EDA.

Full discussion: [`reports/final_report.md`](reports/final_report.md).

## 4. Repository structure
```
SWYNEX-Final-Data-Science-Project/
├── README.md
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/telco_churn_raw.csv            # untouched source file
│   └── processed/telco_churn_clean.csv    # pipeline output (7,043 x 21)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling_evaluation.ipynb
├── src/
│   ├── data_prep.py                       # clean() + validate() + CLI
│   ├── features.py                        # add_features(), used by the saved model
│   └── report.py                          # generates the data-quality report
├── tests/test_data_prep.py                # 9 unit tests
├── models/
│   ├── churn_model.joblib                 # trained sklearn pipeline
│   └── metrics.json                       # test metrics, threshold, cost assumptions
├── reports/
│   ├── final_report.md                    # findings, recommendations, limitations
│   ├── data_quality_report.md             # auto-generated
│   └── figures/                           # all charts
└── docs/data_dictionary.md
```

## 5. How to run
```bash
pip install -r requirements.txt
python -m src.data_prep        # raw -> validated clean CSV (add --download to fetch raw data)
python -m src.report           # regenerate the data-quality report
pytest                         # 9 unit tests
jupyter notebook               # run notebooks 01 -> 02 -> 03 in order
```
All randomness is seeded (`SEED = 42`). The modeling notebook runs the Task 1 pipeline itself if the cleaned CSV is missing.

**Using the saved model** (run from the repository root, so `src` is importable)
```python
import joblib, pandas as pd
model = joblib.load("models/churn_model.joblib")
df = pd.read_csv("data/processed/telco_churn_clean.csv")
X = df.drop(columns=["customer_id", "churn"])
risk = model.predict_proba(X)[:, 1]      # contact customers with risk >= 0.24
```

## 6. Design choices
- **Logic lives in `src/`, not in notebooks**, so it is tested and reusable. The report is generated from the pipeline's audit log, so documentation cannot drift from code.
- **Fail loudly:** unexpected categories or blanks raise errors instead of being silently patched.
- **Leak-free modeling:** scaling and encoding sit inside an sklearn `Pipeline`; the test set is used once; the threshold is chosen on out-of-fold training predictions.
- **Decision-focused evaluation:** the threshold comes from a cost model, not the default 0.5.

## 7. Limitations
1. The **cost assumptions** ($50 offer, 30% save rate, 12 months of value) are illustrative and must be replaced with real business figures.
2. **Static snapshot:** no time dimension, so no temporal validation or drift monitoring; a random split may be optimistic.
3. **Association, not causation:** acting on drivers (e.g. contract upgrades) needs an A/B test.
4. **One public dataset**; no usage, complaint, or network-quality data.
5. **Fairness not assessed** (the data includes `gender` and `senior_citizen`).
6. At the chosen threshold the model flags **no** two-year-contract customers (only 2.7% of them churn) and catches just 19% of one-year churners; it works best on the month-to-month base (recall 0.92), where most churn happens.

## 8. Next steps
Replace cost assumptions with real numbers; add behavioural features and a time-based split; try LightGBM/XGBoost with SHAP explanations; run a randomised retention pilot to measure true uplift; monitor calibration and drift after launch.
