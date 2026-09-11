# Input: ILE_Modified.xlsx | Output: category_sales_inventory_clustering/

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTPUT_DIR = BASE / "category_sales_inventory_clustering"
OUTPUT_DIR.mkdir(exist_ok=True)


print("Loading dataset...")

df = pd.read_excel(INPUT_FILE)

df["Sales_Amount_Actual"] = pd.to_numeric(
    df["Sales_Amount_Actual"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df = df.dropna(subset=["Sales_Amount_Actual", "Quantity"])


def run_clustering(category_column):

    print(f"\nProcessing category: {category_column}")

    outdir = OUTPUT_DIR / category_column
    outdir.mkdir(exist_ok=True)


    agg = (
        df.groupby(category_column)
        .agg(
            Sales=("Sales_Amount_Actual", "sum"),
            Inventory_Movement=("Quantity", lambda x: x.abs().sum())
        )
        .reset_index()
    )

    print("\nAggregated Data:")
    print(agg)


    X = agg[["Sales", "Inventory_Movement"]]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    agg["Cluster"] = kmeans.fit_predict(X_scaled)


    agg.to_csv(
        outdir / f"{category_column}_clusters.csv",
        index=False
    )


    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        agg["Sales"],
        agg["Inventory_Movement"],
        c=agg["Cluster"],
        cmap="Set1",
        s=120
    )

    for i, row in agg.iterrows():

        plt.text(
            row["Sales"],
            row["Inventory_Movement"],
            str(row[category_column]),
            fontsize=9
        )

    plt.xlabel("Total Sales Amount")
    plt.ylabel("Total Inventory Movement")

    plt.title(
        f"Clustering: Sales vs Inventory\nCategory: {category_column}"
    )

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        outdir / f"{category_column}_clusters.png",
        dpi=300
    )

    plt.close()

    print("✔ Clustering complete")


run_clustering("itemCategoryCode")
run_clustering("thc_product_group")
run_clustering("product_family")

print("\nAll clustering completed.")
