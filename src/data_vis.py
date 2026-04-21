import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "synthetic_credit_risk_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
TARGET_COLUMN = "target"

sns.set_style("whitegrid")


#Debt Ratio Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=TARGET_COLUMN, y="debt_ratio", data=df)
plt.title("Debt Ratio by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Debt Ratio")
plt.xticks([0, 1], ["Low Risk (0)", "High Risk (1)"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "debt_ratio_boxplot.png")
plt.show()

#Credit Utilization Histogram

plt.figure(figsize=(8, 5))
plt.hist(
    df[df[TARGET_COLUMN] == 0]["credit_utilization"],
    bins=20,
    alpha=0.6,
    label="Low Risk (0)",
    edgecolor="black",
    density=True
)
plt.hist(
    df[df[TARGET_COLUMN] == 1]["credit_utilization"],
    bins=20,
    alpha=0.6,
    label="High Risk (1)",
    edgecolor="black",
    density=True
)
plt.xlabel("Credit Utilization")
plt.ylabel("Density")
plt.title("Credit Utilization by Risk Group")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "credit_utilization_histogram.png")
plt.show()

#Last Payment Delay Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=TARGET_COLUMN, y="last_payment_delay_days", data=df)
plt.title("Last Payment Delay by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Last Payment Delay (Days)")
plt.xticks([0, 1], ["Low Risk (0)", "High Risk (1)"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "last_payment_delay_boxplot.png")
plt.show()

#Average Monthly Income and Debt Ratio
avg_by_target = df.groupby(TARGET_COLUMN)[["monthly_income", "debt_ratio"]].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(
    ["Low Risk (0)", "High Risk (1)"],
    avg_by_target["monthly_income"],
    edgecolor="black"
)
axes[0].set_title("Average Monthly Income by Risk Group")
axes[0].set_ylabel("Average Monthly Income")

axes[1].bar(
    ["Low Risk (0)", "High Risk (1)"],
    avg_by_target["debt_ratio"],
    edgecolor="black"
)
axes[1].set_title("Average Debt Ratio by Risk Group")
axes[1].set_ylabel("Average Debt Ratio")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_income_debt_ratio.png")
plt.show()

#Credit Utilization vs Debt Ratio Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

low_risk = df[df[TARGET_COLUMN] == 0]
high_risk = df[df[TARGET_COLUMN] == 1]

axes[0].scatter(low_risk["credit_utilization"], low_risk["debt_ratio"], alpha=0.5)
axes[0].set_title("Low Risk (0)")
axes[0].set_xlabel("Credit Utilization")
axes[0].set_ylabel("Debt Ratio")

axes[1].scatter(high_risk["credit_utilization"], high_risk["debt_ratio"], alpha=0.5)
axes[1].set_title("High Risk (1)")
axes[1].set_xlabel("Credit Utilization")
axes[1].set_ylabel("Debt Ratio")

plt.suptitle("Credit Utilization vs Debt Ratio by Risk Group")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "credit_utilization_vs_debt_ratio_scatter.png")
plt.show()