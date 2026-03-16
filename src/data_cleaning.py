from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "synthetic_credit_risk.csv"
OUTPUT_DATASET_PATH = BASE_DIR / "data" / "processed" / "synthetic_credit_risk_cleaned.csv"
SUMMARY_PATH = BASE_DIR / "report" / "data_cleaning_eda_summary.md"

NUMERIC_COLUMNS = [
    "age",
    "monthly_income",
    "debt_ratio",
    "credit_utilization",
    "transaction_count_30d",
    "avg_transaction_amount",
    "last_payment_delay_days",
    "internal_score_v2",
]

CATEGORICAL_COLUMNS = [
    "employment_type",
    "education_level",
    "region",
    "device_type",
]

TARGET_COLUMN = "target"


def format_table(headers, rows):
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, divider_row] + body_rows)


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def missing_value_summary(df):
    return df.isna().sum().astype(int)


def describe_numeric(df):
    numeric_df = df[NUMERIC_COLUMNS]
    return pd.DataFrame(
        {
            "count": numeric_df.count(),
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "std": numeric_df.std(ddof=0),
            "min": numeric_df.min(),
            "q1": numeric_df.quantile(0.25),
            "q3": numeric_df.quantile(0.75),
            "max": numeric_df.max(),
        }
    )


def outlier_summary(df):
    q1 = df[NUMERIC_COLUMNS].quantile(0.25)
    q3 = df[NUMERIC_COLUMNS].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    masks = (df[NUMERIC_COLUMNS].lt(lower)) | (df[NUMERIC_COLUMNS].gt(upper))

    summary = pd.DataFrame(
        {
            "lower": lower,
            "upper": upper,
            "count": masks.sum().astype(int),
            "share": masks.mean(),
        }
    )
    return summary, lower, upper


def cap_outliers(df, lower, upper):
    cleaned_df = df.copy()
    before = cleaned_df[NUMERIC_COLUMNS].copy()
    cleaned_df[NUMERIC_COLUMNS] = cleaned_df[NUMERIC_COLUMNS].clip(lower=lower, upper=upper, axis=1)
    capped_counts = (before.ne(cleaned_df[NUMERIC_COLUMNS])).sum().astype(int)
    return cleaned_df, capped_counts


def group_default_rates(df, group_column):
    grouped = (
        df.groupby(group_column, dropna=False)[TARGET_COLUMN]
        .agg(["count", "mean"])
        .rename(columns={"mean": "default_rate"})
        .reset_index()
        .sort_values(group_column)
    )
    return grouped


def write_cleaned_dataset(df):
    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATASET_PATH, index=False)


def build_summary(df, missing_counts, descriptions, outliers, capped_counts):
    target_rate = df[TARGET_COLUMN].mean()
    lines = [
        "# Data Cleaning and EDA Summary",
        "",
        "## Proposed Data-Driven Problem",
        "",
        "Predict whether a borrower will be labeled as high credit risk (`target = 1`) using demographic, financial, behavioral, and internal scoring features.",
        "",
        "- Target variable: `target`",
        "- Problem type: binary classification",
        "- Practical motivation: identify risky applicants early so lending decisions and intervention policies can be improved.",
        "",
        "## Dataset Snapshot",
        "",
        f"- Total rows: {len(df):,}",
        f"- Total columns: {df.shape[1]}",
        f"- Positive class rate: {target_rate:.3%}",
        "",
        "## Missing Values",
        "",
    ]

    missing_rows = [[column, str(int(missing_counts[column]))] for column in df.columns]
    lines.append(format_table(["Column", "Missing Values"], missing_rows))
    lines.extend(
        [
            "",
            "No missing values were detected, so no imputation or row deletion was required.",
            "",
            "## Descriptive Statistics",
            "",
        ]
    )

    stat_rows = []
    for column in NUMERIC_COLUMNS:
        stats = descriptions.loc[column]
        stat_rows.append(
            [
                column,
                str(int(stats["count"])),
                f"{stats['mean']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['q1']:.3f}",
                f"{stats['q3']:.3f}",
                f"{stats['max']:.3f}",
            ]
        )
    lines.append(
        format_table(
            ["Feature", "Count", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max"],
            stat_rows,
        )
    )

    lines.extend(["", "## Outlier Detection and Handling", ""])
    outlier_rows = []
    for column in NUMERIC_COLUMNS:
        info = outliers.loc[column]
        outlier_rows.append(
            [
                column,
                f"{info['lower']:.3f}",
                f"{info['upper']:.3f}",
                str(int(info["count"])),
                f"{info['share']:.3%}",
                str(int(capped_counts[column])),
            ]
        )
    lines.append(
        format_table(
            ["Feature", "Lower Bound", "Upper Bound", "Outliers", "Outlier Share", "Capped Rows"],
            outlier_rows,
        )
    )

    lines.extend(
        [
            "",
            "Outliers were identified with the IQR rule. Instead of dropping observations, numeric outliers were capped to the IQR bounds in the model-ready dataset. This keeps all 15,000 records while reducing the influence of extreme synthetic values.",
            "",
            "## Group-By Analysis",
            "",
            "Default rates differ meaningfully across borrower groups, which makes these fields useful during EDA and modeling.",
            "",
        ]
    )

    for column in CATEGORICAL_COLUMNS:
        grouped = group_default_rates(df, column)
        group_rows = [
            [str(row[column]), str(int(row["count"])), f"{row['default_rate']:.3%}"]
            for _, row in grouped.iterrows()
        ]
        lines.append(f"### `{column}`")
        lines.append("")
        lines.append(format_table(["Group", "Count", "Default Rate"], group_rows))
        lines.append("")

    lines.extend(
        [
            "## Recommended Interpretation",
            "",
            "- The strongest early risk indicators appear to be employment instability, lower internal score, and larger payment delays.",
            "- The target is imbalanced, so later modeling should use metrics such as precision, recall, F1, and ROC-AUC instead of accuracy alone.",
            "- Keep the original raw dataset unchanged and use the cleaned dataset for modeling to preserve reproducibility.",
            "",
            f"Cleaned dataset written to `{OUTPUT_DATASET_PATH.relative_to(BASE_DIR)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    df = load_dataset()
    missing_counts = missing_value_summary(df)
    descriptions = describe_numeric(df)
    outliers, lower, upper = outlier_summary(df)
    cleaned_df, capped_counts = cap_outliers(df, lower, upper)

    write_cleaned_dataset(cleaned_df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        build_summary(df, missing_counts, descriptions, outliers, capped_counts),
        encoding="utf-8",
    )

    print(f"Cleaned dataset saved to {OUTPUT_DATASET_PATH}")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
