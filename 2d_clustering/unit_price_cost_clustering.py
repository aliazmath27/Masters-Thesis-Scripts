# Input: ILE_Modified.xlsx | Output: unit_price_cost_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "unit_price_cost_clustering"
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
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["unit_cost"]  = pd.to_numeric(df["unit_cost"],  errors="coerce")
df = df.dropna(subset=["unit_price", "unit_cost"])

def run_clustering(category_col):
    print(f"\nProcessing: {category_col}")
    folder = OUTDIR / category_col
    folder.mkdir(exist_ok=True)

    agg = (
        df.groupby(category_col)
        .agg(
            avg_unit_price=("unit_price", "mean"),
            avg_unit_cost =("unit_cost",  "mean"),
        )
        .reset_index()
        .dropna()
    )
    agg["avg_unit_cost"] = agg["avg_unit_cost"].abs()

    print("\nAggregated Data:")
    print(agg.to_string())

    X        = agg[["avg_unit_price", "avg_unit_cost"]].values
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
    plt.title(f"Elbow Method — {category_col} (Unit Price vs Unit Cost)")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(folder / "elbow.png", dpi=300)
    plt.close()
    print("✔ Saved elbow plot")

    k              = K_MAP[category_col]
    km_final       = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = km_final.fit_predict(X_scaled)

    agg.to_csv(folder / "clusters.csv", index=False)
    print("✔ Saved cluster CSV")

    COLORS = ["#2563EB", "#DC2626", "#16A34A", "#CA8A04", "#7C3AED"]
    plt.figure(figsize=(10, 6))
    for ci in sorted(agg["cluster"].unique()):
        sub = agg[agg["cluster"] == ci]
        plt.scatter(sub["avg_unit_price"], sub["avg_unit_cost"],
                    color=COLORS[ci % len(COLORS)], s=150,
                    label=f"Cluster {ci}", edgecolors="white", linewidth=1)
    for _, row in agg.iterrows():
        plt.text(row["avg_unit_price"], row["avg_unit_cost"],
                 str(row[category_col]), fontsize=9)
    plt.xlabel("Average Unit Price (€)")
    plt.ylabel("Average Unit Cost (€, abs)")
    plt.title(f"{category_col} Clusters (Unit Price vs Unit Cost, k={k})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(folder / "scatter.png", dpi=300)
    plt.close()
    print(f"✔ Completed: {category_col}")

for cat in CATEGORY_COLS:
    if cat in df.columns:
        run_clustering(cat)

print("\n✅ ALL CLUSTERING COMPLETED")
print(f"📁 Output folder: {OUTDIR}")
