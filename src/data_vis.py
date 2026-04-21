import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "synthetic_credit_risk.csv"
OUTPUT_DIR = BASE_DIR / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

TARGET_COLUMN = "target"

sns.set_style("whitegrid")


# Debt Ratio Boxplot
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





