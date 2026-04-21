# Data Cleaning and EDA Summary

## Proposed Data-Driven Problem

Predict whether a borrower will be labeled as high credit risk (`target = 1`) using demographic, financial, behavioral, and internal scoring features.

- Target variable: `target`
- Problem type: binary classification
- Practical motivation: identify risky applicants early so lending decisions and intervention policies can be improved.

## Dataset Snapshot

- Total rows: 15,000
- Total columns: 13
- Positive class rate: 7.720%

## Missing Values

| Column | Missing Values |
| --- | --- |
| age | 0 |
| monthly_income | 0 |
| debt_ratio | 0 |
| credit_utilization | 0 |
| transaction_count_30d | 0 |
| avg_transaction_amount | 0 |
| employment_type | 0 |
| education_level | 0 |
| region | 0 |
| device_type | 0 |
| last_payment_delay_days | 0 |
| internal_score_v2 | 0 |
| target | 0 |

No missing values were detected, so no imputation or row deletion was required.

## Descriptive Statistics

| Feature | Count | Mean | Median | Std | Min | Q1 | Q3 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 15000 | 45.558 | 45.600 | 11.014 | 18.000 | 38.200 | 52.900 | 75.000 |
| monthly_income | 15000 | 43234.147 | 39141.395 | 21168.674 | 4024.160 | 28488.085 | 53188.735 | 205203.980 |
| debt_ratio | 15000 | 0.335 | 0.328 | 0.155 | 0.000 | 0.229 | 0.431 | 1.000 |
| credit_utilization | 15000 | 0.401 | 0.392 | 0.196 | 0.000 | 0.265 | 0.521 | 1.000 |
| transaction_count_30d | 15000 | 43.334 | 42.000 | 8.584 | 18.000 | 38.000 | 47.000 | 95.000 |
| avg_transaction_amount | 15000 | 92.103 | 91.655 | 35.616 | 10.000 | 67.320 | 115.683 | 239.500 |
| last_payment_delay_days | 15000 | 2.586 | 1.620 | 3.416 | 0.000 | 0.650 | 3.260 | 50.430 |
| internal_score_v2 | 15000 | 587.514 | 592.600 | 57.424 | 276.500 | 557.700 | 624.825 | 750.200 |

## Outlier Detection and Handling

| Feature | Lower Bound | Upper Bound | Outliers | Outlier Share | Capped Rows |
| --- | --- | --- | --- | --- | --- |
| age | 16.150 | 74.950 | 56 | 0.373% | 56 |
| monthly_income | -8562.890 | 90239.710 | 514 | 3.427% | 514 |
| debt_ratio | -0.074 | 0.734 | 181 | 1.207% | 181 |
| credit_utilization | -0.119 | 0.905 | 213 | 1.420% | 213 |
| transaction_count_30d | 24.500 | 60.500 | 694 | 4.627% | 694 |
| avg_transaction_amount | -5.224 | 188.226 | 69 | 0.460% | 69 |
| last_payment_delay_days | -3.265 | 7.175 | 906 | 6.040% | 906 |
| internal_score_v2 | 457.013 | 725.512 | 501 | 3.340% | 501 |

Outliers were identified with the IQR rule. Instead of dropping observations, numeric outliers were capped to the IQR bounds in the model-ready dataset. This keeps all 15,000 records while reducing the influence of extreme synthetic values.

## Group-By Analysis

Default rates differ meaningfully across borrower groups, which makes these fields useful during EDA and modeling.

### `employment_type`

| Group | Count | Default Rate |
| --- | --- | --- |
| contract | 2158 | 9.361% |
| full_time | 8783 | 5.226% |
| self_employed | 2755 | 11.361% |
| unemployed | 1304 | 14.110% |

### `education_level`

| Group | Count | Default Rate |
| --- | --- | --- |
| bachelor | 6011 | 7.403% |
| high_school | 5415 | 7.904% |
| master | 2846 | 8.117% |
| phd | 728 | 7.418% |

### `region`

| Group | Count | Default Rate |
| --- | --- | --- |
| rural | 2706 | 7.650% |
| suburban | 4529 | 7.264% |
| urban | 7765 | 8.010% |

### `device_type`

| Group | Count | Default Rate |
| --- | --- | --- |
| desktop | 4218 | 7.563% |
| mobile | 9433 | 7.802% |
| tablet | 1349 | 7.635% |

## Recommended Interpretation

- The strongest early risk indicators appear to be employment instability, lower internal score, and larger payment delays.
- The target is imbalanced, so later modeling should use metrics such as precision, recall, F1, and ROC-AUC instead of accuracy alone.
- Keep the original raw dataset unchanged and use the cleaned dataset for modeling to preserve reproducibility.

Cleaned dataset written to `data\processed\synthetic_credit_risk_cleaned.csv`.
