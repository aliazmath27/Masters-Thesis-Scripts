# Input: ILE_Modified.xlsx | Output: inventory_velocity_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTDIR = BASE / "inventory_velocity_clustering"
OUTDIR.mkdir(exist_ok=True)

DATE_COL = "Posting_Date"


print("Loading data...")

df = pd.read_excel(INPUT_FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df = df.dropna(subset=[DATE_COL, "Quantity"])


def run_clustering(category_col):

    print(f"\nProcessing: {category_col}")

    folder = OUTDIR / category_col
    folder.mkdir(exist_ok=True)

    daily = (
        df.groupby([category_col, DATE_COL])["Quantity"]
        .sum()
        .reset_index()
    )

    rows = []

    categories = daily[category_col].dropna().unique()

    for cat in categories:

        sub = daily[daily[category_col] == cat].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(DATE_COL)

        sub["Inventory_On_Hand"] = sub["Quantity"].cumsum()

        avg_inventory = sub["Inventory_On_Hand"].mean()

        total_quantity = sub["Quantity"].sum()
        active_days = sub[DATE_COL].nunique()

        velocity = total_quantity / active_days if active_days > 0 else 0

        rows.append({
            category_col: cat,
            "inventory": avg_inventory,
            "velocity": velocity
        })

    agg = pd.DataFrame(rows)

    print("\nAggregated Data:")
    print(agg)

    X = agg[["inventory", "velocity"]]

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

    plt.title(f"Elbow Method - {category_col} (Inventory vs Velocity)")
    plt.xlabel("Clusters")
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
            sub["inventory"],
            label=f"Cluster {cluster}",
            s=150
        )

    for _, row in agg.iterrows():
        plt.text(
            row["velocity"],
            row["inventory"],
            str(row[category_col]),
            fontsize=9
        )

    plt.xlabel("Velocity (Units per Active Day)")
    plt.ylabel("Average Inventory On Hand")

    plt.title(f"{category_col} Clusters (Inventory vs Velocity, k=3)")

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
