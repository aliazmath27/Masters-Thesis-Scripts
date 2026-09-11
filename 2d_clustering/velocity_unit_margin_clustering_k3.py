# Input: ILE_Modified.xlsx | Output: velocity_unit_margin_clustering_k3/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTDIR = BASE / "velocity_unit_margin_clustering_k3"
OUTDIR.mkdir(exist_ok=True)

DATE_COL = "Posting_Date"


print("Loading data...")

df = pd.read_excel(INPUT_FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
]

df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")

df["unit_margin"] = df["unit_price"] + df["unit_cost"]

df["Quantity"] = df["Quantity"].abs()


def run_clustering(category_col):

    print(f"\nProcessing: {category_col}")

    folder = OUTDIR / category_col
    folder.mkdir(exist_ok=True)

    agg = (
        df.groupby(category_col)
        .agg(
            total_quantity=("Quantity", "sum"),
            active_days=(DATE_COL, "nunique"),
            avg_unit_margin=("unit_margin", "mean")
        )
        .reset_index()
    )

    agg["velocity"] = agg["total_quantity"] / agg["active_days"]

    print("\nAggregated Data:")
    print(agg)

    X = agg[["velocity", "avg_unit_margin"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    k_range = range(1, len(agg) + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(k_range, inertia, marker="o")

    plt.title(f"Elbow Method - {category_col}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(folder / "elbow.png", dpi=300)
    plt.close()

    print("✔ Saved elbow plot")

    k = 3

    kmeans = KMeans(n_clusters=k, random_state=42)
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    agg.to_csv(folder / "clusters.csv", index=False)

    print("✔ Saved cluster table")

    plt.figure(figsize=(10,6))

    for cluster in agg["cluster"].unique():

        sub = agg[agg["cluster"] == cluster]

        plt.scatter(
            sub["velocity"],
            sub["avg_unit_margin"],
            label=f"Cluster {cluster}",
            s=150
        )

    for _, row in agg.iterrows():
        plt.text(
            row["velocity"],
            row["avg_unit_margin"],
            str(row[category_col]),
            fontsize=9
        )

    plt.xlabel("Velocity (Units per Active Day)")
    plt.ylabel("Average Unit Margin")

    plt.title(f"{category_col} Clusters (Velocity vs Unit Margin, k=3)")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(folder / "scatter.png", dpi=300)
    plt.close()

    print(f"✔ Completed: {category_col}")


run_clustering("product_family")
run_clustering("itemCategoryCode")
run_clustering("thc_product_group")

print("\n✅ ALL CLUSTERING COMPLETED")
print(f"Output folder: {OUTDIR}")
