# Input: ItemLedgerEntriesCustom_all_rows.xlsx + workflowItems_all_rows.csv | Output: item_description_clustering_by_group/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

BASE = r"C:\Users\aliaz\IU\Thesis"

LEDGER_PATH = os.path.join(BASE, "ItemLedgerEntriesCustom_all_rows.xlsx")
WORKFLOW_PATH = os.path.join(BASE, "workflowItems_all_rows.csv")

OUTDIR = os.path.join(BASE, "item_description_clustering_by_group")
os.makedirs(OUTDIR, exist_ok=True)

TARGET_GROUPS = ["CANNABISBLÜTEN", "CANNABISEXTRAKTE"]
K_RANGE = range(2, 8)
K_FINAL = 4

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

def scale(X: pd.DataFrame):
    return StandardScaler().fit_transform(X)

ledger = pd.read_excel(LEDGER_PATH)
workflow = pd.read_csv(WORKFLOW_PATH, sep=";", encoding="utf-8-sig")

ledger["Posting_Date"] = pd.to_datetime(
    ledger["Posting_Date"], errors="coerce", dayfirst=True
)

for col in ["Sales_Amount_Actual", "Cost_Amount_Actual"]:
    ledger[col] = de_number_to_float(ledger[col])

ledger["Entry_Type"] = ledger["Entry_Type"].astype(str).str.strip().str.title()
ledger = ledger[ledger["Entry_Type"] == "Sale"].copy()

ledger["Margin"] = ledger["Sales_Amount_Actual"] + ledger["Cost_Amount_Actual"]

ledger["Item_No"] = (
    pd.to_numeric(ledger["Item_No"], errors="coerce")
    .astype("Int64")
    .astype(str)
)

workflow["number"] = (
    pd.to_numeric(workflow["number"], errors="coerce")
    .astype("Int64")
    .astype(str)
)

workflow["itemCategoryCode"] = (
    workflow["itemCategoryCode"]
    .astype(str)
    .str.strip()
)

df = ledger.merge(
    workflow[["number", "itemCategoryCode"]],
    left_on="Item_No",
    right_on="number",
    how="left"
)

print("\n🔍 itemCategoryCode after merge:")
print(df["itemCategoryCode"].value_counts(dropna=False))

for group in TARGET_GROUPS:

    print(f"\n🔹 Processing product group: {group}")

    sub = df[df["itemCategoryCode"] == group].copy()

    if sub.empty:
        print(f"⚠ No data found for {group}")
        continue

    item_df = (
        sub.groupby("Item_Description")
        .agg(
            total_sales=("Sales_Amount_Actual", "sum"),
            total_margin=("Margin", "sum"),
            n_transactions=("Margin", "count"),
        )
        .fillna(0)
    )

    print(f"✅ Unique Item_Descriptions: {item_df.shape[0]}")

    FEATURES = ["total_sales", "total_margin", "n_transactions"]
    X = scale(item_df[FEATURES])

    group_dir = os.path.join(OUTDIR, group)
    os.makedirs(group_dir, exist_ok=True)

    inertia, silhouette = [], []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(X, labels))

    pd.DataFrame({
        "k": list(K_RANGE),
        "inertia": inertia,
        "silhouette": silhouette
    }).to_csv(os.path.join(group_dir, "k_diagnostics.csv"), index=False)

    plt.figure()
    plt.plot(K_RANGE, inertia, marker="o")
    plt.title(f"Elbow Method – {group}")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig(os.path.join(group_dir, "elbow_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(K_RANGE, silhouette, marker="o")
    plt.title(f"Silhouette Score – {group}")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.savefig(os.path.join(group_dir, "silhouette_curve.png"), dpi=300)
    plt.close()

    kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20)
    item_df["cluster"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=item_df["cluster"],
        cmap="tab10",
        alpha=0.8
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"{group} – Item_Description Clusters (PCA)")
    plt.colorbar(label="Cluster")
    plt.savefig(os.path.join(group_dir, "pca_clusters.png"), dpi=300)
    plt.close()

    item_df.reset_index().to_csv(
        os.path.join(group_dir, "item_descriptions_with_clusters.csv"),
        index=False
    )

    cluster_summary = (
        item_df.groupby("cluster")[FEATURES]
        .mean()
        .round(2)
    )

    cluster_summary.to_csv(
        os.path.join(group_dir, "cluster_summary.csv")
    )

    print(f"📁 Saved results to: {group_dir}")

print("\n🎯 All product-group clusterings completed.")

