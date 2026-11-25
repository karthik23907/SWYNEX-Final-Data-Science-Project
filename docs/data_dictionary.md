# Data Dictionary (cleaned dataset)

File: `data/processed/telco_churn_clean.csv` (7,043 rows x 21 columns, 0 missing values)

| Column | Type | Description | Values / range |
|---|---|---|---|
| customer_id | string | Unique customer identifier | e.g. `7590-VHVEG` |
| gender | category | Customer gender | Female, Male |
| senior_citizen | int8 | Customer is a senior citizen | 0 / 1 |
| partner | int8 | Has a partner | 0 / 1 |
| dependents | int8 | Has dependents | 0 / 1 |
| tenure | int16 | Months with the company | 0 - 72 |
| phone_service | int8 | Has phone service | 0 / 1 |
| multiple_lines | int8 | Has multiple phone lines (0 if no phone service) | 0 / 1 |
| internet_service | category | Internet plan type | DSL, Fiber optic, No |
| online_security | int8 | Online security add-on (0 if no internet) | 0 / 1 |
| online_backup | int8 | Online backup add-on (0 if no internet) | 0 / 1 |
| device_protection | int8 | Device protection add-on (0 if no internet) | 0 / 1 |
| tech_support | int8 | Tech support add-on (0 if no internet) | 0 / 1 |
| streaming_tv | int8 | Streaming TV add-on (0 if no internet) | 0 / 1 |
| streaming_movies | int8 | Streaming movies add-on (0 if no internet) | 0 / 1 |
| contract | ordered category | Contract term | Month-to-month < One year < Two year |
| paperless_billing | int8 | Uses paperless billing | 0 / 1 |
| payment_method | category | Payment method | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| monthly_charges | float64 | Current monthly bill | 18.25 - 118.75 |
| total_charges | float64 | Lifetime billed amount (0 for brand-new customers) | 0 - ~8,684 |
| churn | int8 | **Target.** Customer left in the last month | 0 / 1 |

Notes: for the add-on columns, "No internet service" / "No phone service" were merged into 0 because
`internet_service` and `phone_service` already carry that information.
