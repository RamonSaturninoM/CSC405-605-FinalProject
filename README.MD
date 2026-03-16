# Synthetic Credit Risk Final Project

## Team Information

- Team members:
  - Ramon Saturnino

## Project Title

Predicting Synthetic Credit Risk with Financial and Behavioral Features

## Dataset Source

- Dataset file: `data/synthetic_credit_risk.csv`
- Source link: https://www.kaggle.com/datasets/emirhanakku/synthetic-credit-risk-dataset-extreme-imbalance



## Project Description

This project studies whether a borrower should be classified as high credit risk using demographic, financial, transaction, and internal scoring variables. The main response variable is `target`, where `1` indicates higher credit risk and `0` indicates lower credit risk.

## Data-Driven Problem

- Target variable: `target`
- Problem type: binary classification
- Core question: Which borrower characteristics are most associated with elevated credit risk, and how accurately can that risk be predicted?

## Repository Structure

- `data/` raw and processed datasets
- `notebooks/` analysis notebooks
- `src/` reusable project scripts
- `figures/` charts and exported visuals
- `report/` final report and supporting writeups

## Data Cleaning and EDA

Run the cleaning and EDA script:

```bash
.venv/bin/python src/data_cleaning.py
```

This script:

- checks for missing values
- computes descriptive statistics
- detects outliers with the IQR rule
- caps numeric outliers for a model-ready dataset instead of dropping rows
- uses `pandas` and `numpy` for the cleaning workflow
- writes a summary to `report/data_cleaning_eda_summary.md`

## Notes

Version control history in this repository should reflect each team member's contribution.
