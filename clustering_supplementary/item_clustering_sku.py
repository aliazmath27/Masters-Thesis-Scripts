# Input: ItemLedgerEntriesCustom_all_rows.xlsx | Output: item_clustering_SKU/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

FILE_PATH = r"C:\Users\aliaz\IU\Thesis\ItemLedgerEntriesCustom_all_rows.xlsx"
OUTDIR = r"C:\Users\aliaz\IU\Thesis\item_clustering_SKU"
os.makedirs(OUTDIR, exist_ok=True)

K_RANGE = range(2, 8)
K_UNIVARIATE = 3
K_MULTIVARIATE = 4

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

def scale_df(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    return model.fit_predict(X)

def plot_univariate(series, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(series)), series, c=labels, cmap="tab10", alpha=0.7)
    plt.title(title)
    plt.xlabel("Item index")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300)
    plt.close()

df = pd.read_excel(FILE_PATH)

df["Posting_Date"] = pd.to_datetime(
    df["Posting_Date"], errors="coerce", dayfirst=True
)

for col in ["Sales_Amount_Actual", "Cost_Amount_Actual"]:
    if col in df.columns:
        df[col] = de_number_to_float(df[col])

df["Entry_Type"] = df["Entry_Type"].astype(str).str.strip().str.title()
df = df[df["Entry_Type"] == "Sale"].copy()

df["Margin"] = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]

item_df = (
    df.groupby("Item_No")
      .agg(
          total_sales=("Sales_Amount_Actual", "sum"),
          total_margin=("Margin", "sum"),
          n_transactions=("Margin", "count"),
      )
      .fillna(0)
)

total_items = item_df.shape[0]
print(f"Total unique items (Item_No): {total_items}")

FEATURES = ["total_sales", "total_margin", "n_transactions"]
X_scaled = scale_df(item_df[FEATURES])

elbow, silhouette = [], []

for k in K_RANGE:
    labels = run_kmeans(X_scaled, k)
    model = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X_scaled)
    elbow.append(model.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))

pd.DataFrame({
    "k": list(K_RANGE),
    "inertia": elbow,
    "silhouette": silhouette
}).to_csv(os.path.join(OUTDIR, "k_diagnostics.csv"), index=False)

plt.figure()
plt.plot(K_RANGE, elbow, marker="o")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.savefig(os.path.join(OUTDIR, "elbow_curve.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(K_RANGE, silhouette, marker="o")
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
plt.savefig(os.path.join(OUTDIR, "silhouette_curve.png"), dpi=300)
plt.close()

item_df["cluster_sales"] = run_kmeans(
    scale_df(item_df[["total_sales"]]), K_UNIVARIATE
)
item_df["cluster_transactions"] = run_kmeans(
    scale_df(item_df[["n_transactions"]]), K_UNIVARIATE
)
item_df["cluster_margin"] = run_kmeans(
    scale_df(item_df[["total_margin"]]), K_UNIVARIATE
)

plot_univariate(
    item_df["total_sales"].values,
    item_df["cluster_sales"].values,
    "Univariate Clustering — Total Sales",
    "Total Sales",
    "univariate_cluster_total_sales.png"
)

plot_univariate(
    item_df["n_transactions"].values,
    item_df["cluster_transactions"].values,
    "Univariate Clustering — Number of Transactions",
    "Number of Transactions",
    "univariate_cluster_n_transactions.png"
)

plot_univariate(
    item_df["total_margin"].values,
    item_df["cluster_margin"].values,
    "Univariate Clustering — Total Margin",
    "Total Margin",
    "univariate_cluster_total_margin.png"
)

item_df["cluster_combined"] = run_kmeans(X_scaled, K_MULTIVARIATE)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=item_df["cluster_combined"],
    cmap="tab10",
    alpha=0.8
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Multivariate Item Clusters (PCA Projection)")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pca_clusters.png"), dpi=300)
plt.close()

item_df.reset_index().to_csv(
    os.path.join(OUTDIR, "items_with_all_clusters.csv"),
    index=False
)

cluster_summary = (
    item_df.groupby("cluster_combined")[FEATURES]
    .mean()
    .round(2)
)
cluster_summary.to_csv(os.path.join(OUTDIR, "cluster_summary_combined.csv"))

with open(os.path.join(OUTDIR, "cluster_overview.txt"), "w") as f:
    f.write("FINAL ITEM CLUSTERING OVERVIEW\n")
    f.write("==============================\n\n")
    f.write(f"Entry_Type used: Sale only\n")
    f.write(f"Total unique items (Item_No): {total_items}\n\n")
    f.write("Features used:\n")
    for feat in FEATURES:
        f.write(f"- {feat}\n")
    f.write("\nClustering steps:\n")
    f.write("- Univariate clustering (sales, transactions, margin)\n")
    f.write("- Multivariate clustering (combined)\n")
    f.write(f"- Final number of clusters (multivariate): {K_MULTIVARIATE}\n")

print("✅ FINAL clustering pipeline completed successfully")
print("All outputs saved to:", OUTDIR)

