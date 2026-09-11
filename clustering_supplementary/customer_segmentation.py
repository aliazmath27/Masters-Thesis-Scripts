# Input: ItemLedgerEntriesCustom_all_rows.xlsx | Output: customer_clustering/

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

FILE_PATH = r"C:\Users\aliaz\IU\Thesis\ItemLedgerEntriesCustom_all_rows.xlsx"

OUTPUT_DIR = Path(r"C:\Users\aliaz\IU\Thesis\customer_clustering")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_K = 10
FINAL_K = 4
RANDOM_STATE = 42

def de_number_to_float(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    mask = s.str.contains(",", na=False)
    s.loc[mask] = (
        s.loc[mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")

df = pd.read_excel(FILE_PATH)

df["Entry_Type"] = df["Entry_Type"].astype(str).str.strip().str.title()
df["Source_Type"] = df["Source_Type"].astype(str).str.strip().str.title()

df = df[
    (df["Entry_Type"] == "Sale") &
    (df["Source_Type"] == "Customer")
].copy()

df["Sales_Amount_Actual"] = de_number_to_float(df["Sales_Amount_Actual"])
df["Cost_Amount_Actual"] = de_number_to_float(df["Cost_Amount_Actual"])
df["Quantity"] = de_number_to_float(df["Quantity"])

df["Margin"] = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]

customer_df = (
    df.groupby("Source_No")
      .agg(
          total_sales=("Sales_Amount_Actual", "sum"),
          total_margin=("Margin", "sum"),
          total_quantity=("Quantity", lambda x: x.abs().sum()),
          n_transactions=("Entry_No", "count")
      )
)

features = [
    "total_sales",
    "total_margin",
    "total_quantity",
    "n_transactions"
]

X = customer_df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for k in range(1, MAX_K + 1):
    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, MAX_K + 1), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Customer Clustering")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "elbow_plot.png")
plt.close()

print("\n📉 Elbow Method (WCSS):")
for k, val in enumerate(wcss, start=1):
    print(f"K={k}: WCSS={val:.2f}")

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=RANDOM_STATE,
    n_init=10
)

customer_df["cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=customer_df.index
)

pca_df["cluster"] = customer_df["cluster"]

explained_var = pca.explained_variance_ratio_

print(
    f"\n📐 PCA explained variance:"
    f"\nPC1: {explained_var[0]*100:.2f}%"
    f"\nPC2: {explained_var[1]*100:.2f}%"
    f"\nTotal: {(explained_var[0]+explained_var[1])*100:.2f}%"
)

plt.figure()
for c in sorted(pca_df["cluster"].unique()):
    subset = pca_df[pca_df["cluster"] == c]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {c}",
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Customer Clusters")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_clusters.png")
plt.close()

cluster_summary = (
    customer_df
    .groupby("cluster")
    .agg(
        n_customers=("cluster", "count"),
        avg_sales=("total_sales", "mean"),
        avg_margin=("total_margin", "mean"),
        avg_quantity=("total_quantity", "mean"),
        avg_transactions=("n_transactions", "mean")
    )
    .round(2)
)

customer_df.to_csv(OUTPUT_DIR / "customer_clustering.csv")
cluster_summary.to_csv(OUTPUT_DIR / "customer_clustering_summary.csv")

print("\n📊 CUSTOMER CLUSTERING SUMMARY:\n")
print(cluster_summary.to_string())

print("\n📁 Files written to:")
print(str(OUTPUT_DIR))

print("\n✅ Customer clustering analysis completed successfully.")

