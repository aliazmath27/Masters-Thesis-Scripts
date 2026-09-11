# Input: ILE_Modified.xlsx | Output: inventory_cv_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTDIR = BASE / "inventory_cv_clustering"
OUTDIR.mkdir(exist_ok=True)

DATE_COL = "Posting_Date"


print("Loading data...")

df = pd.read_excel(INPUT_FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df = df.dropna(subset=[DATE_COL, "Quantity"])


sales_df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
].copy()

sales_df["Quantity"] = sales_df["Quantity"].abs()


def run_clustering(category_col):

    print(f"\nProcessing: {category_col}")

    folder = OUTDIR / category_col
    folder.mkdir(exist_ok=True)

    daily_inv = (
        df.groupby([category_col, DATE_COL])["Quantity"]
        .sum()
        .reset_index()
    )

    daily_sales = (
        sales_df.groupby([category_col, DATE_COL])["Quantity"]
        .sum()
        .reset_index()
    )

    rows = []

    categories = daily_inv[category_col].dropna().unique()

    for cat in categories:

        inv = daily_inv[daily_inv[category_col] == cat].copy()
        sales = daily_sales[daily_sales[category_col] == cat].copy()

        if inv.empty:
            continue

        inv = inv.sort_values(DATE_COL)

        inv["Inventory_On_Hand"] = inv["Quantity"].cumsum()
        avg_inventory = inv["Inventory_On_Hand"].mean()

        if not sales.empty:
            mean_sales = sales["Quantity"].mean()
            std_sales = sales["Quantity"].std()

            if mean_sales == 0 or pd.isna(mean_sales):
                cv = 0
            else:
                cv = std_sales / mean_sales
        else:
            cv = 0

        rows.append({
            category_col: cat,
            "inventory": avg_inventory,
            "cv": cv
        })

    agg = pd.DataFrame(rows)

    print("\nAggregated Data:")
    print(agg)

    X = agg[["inventory", "cv"]]

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

    plt.title(f"Elbow Method - {category_col} (Inventory vs CV)")
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
            sub["inventory"],
            sub["cv"],
            label=f"Cluster {cluster}",
            s=150
        )

    for _, row in agg.iterrows():
        plt.text(
            row["inventory"],
            row["cv"],
            str(row[category_col]),
            fontsize=9
        )

    plt.xlabel("Average Inventory On Hand")
    plt.ylabel("Coefficient of Variation (CV)")

    plt.title(f"{category_col} Clusters (Inventory vs CV, k=3)")

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
