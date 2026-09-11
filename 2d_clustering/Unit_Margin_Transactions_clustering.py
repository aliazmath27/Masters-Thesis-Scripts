# Input: ILE_Modified.xlsx | Output: unit_margin_transactions_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "unit_margin_transactions_clustering"
OUTDIR.mkdir(exist_ok=True)
DATE_COL   = "Posting_Date"

CATEGORY_COLS = ["itemCategoryCode", "thc_product_group", "product_family"]
K_MAP = {"itemCategoryCode": 3, "thc_product_group": 3, "product_family": 4}

print("Loading data...")
df = pd.read_excel(INPUT_FILE)
df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
]
df["unit_price"]  = pd.to_numeric(df["unit_price"],  errors="coerce")
df["unit_cost"]   = pd.to_numeric(df["unit_cost"],   errors="coerce")
df["unit_margin"] = df["unit_price"] + df["unit_cost"]

def run_clustering(category_col):
    print(f"\nProcessing: {category_col}")

    folder = OUTDIR / f"{category_col}_clusters"
    folder.mkdir(exist_ok=True)

    agg = (
        df.groupby(category_col)
        .agg(
            margin_per_unit   =("unit_margin", "mean"),
            total_transactions=("unit_margin", "count"),
        )
        .reset_index()
        .dropna()
    )

    print("\nAggregated Data:")
    print(agg.to_string())

    X        = agg[["margin_per_unit", "total_transactions"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k   = min(len(agg), 8)
    inertia = []
    for ki in range(1, max_k + 1):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia, marker="o", color="#2563EB", linewidth=2)
    plt.title(f"Elbow Method — {category_col} (Margin per Unit vs Transactions)")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(folder / f"{category_col}_elbow.png", dpi=300)
    plt.close()
    print(f"  ✔ Saved elbow plot")

    k              = K_MAP[category_col]
    km_final       = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = km_final.fit_predict(X_scaled)

    agg.to_csv(folder / f"{category_col}_clusters.csv", index=False)
    print(f"  ✔ Saved cluster CSV: {category_col}_clusters.csv")

    COLORS = ["#2563EB", "#DC2626", "#16A34A", "#CA8A04", "#7C3AED"]
    plt.figure(figsize=(10, 6))
    for ci in sorted(agg["cluster"].unique()):
        sub = agg[agg["cluster"] == ci]
        plt.scatter(sub["total_transactions"], sub["margin_per_unit"],
                    color=COLORS[ci % len(COLORS)], s=150,
                    label=f"Cluster {ci}", edgecolors="white", linewidth=1)
    for _, row in agg.iterrows():
        plt.text(row["total_transactions"], row["margin_per_unit"],
                 str(row[category_col]), fontsize=9)
    plt.xlabel("Total Transactions (count)")
    plt.ylabel("Average Margin per Unit (€)")
    plt.title(f"{category_col} Clusters (Margin per Unit vs Transactions, k={k})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(folder / f"{category_col}_scatter.png", dpi=300)
    plt.close()
    print(f"  ✔ Completed: {category_col}")

for cat in CATEGORY_COLS:
    if cat in df.columns:
        run_clustering(cat)

print("\n✅ ALL CLUSTERING COMPLETED")
print(f"📁 Output folder: {OUTDIR}")
