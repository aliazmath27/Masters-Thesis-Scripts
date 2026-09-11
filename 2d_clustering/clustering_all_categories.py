# Input: ILE_Modified.xlsx | Output: clustering_allCatergories_quantity_margin/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "clustering_allCatergories_quantity_margin"
OUTDIR.mkdir(exist_ok=True)
DATE_COL   = "Posting_Date"

CATEGORY_COLS = [
    "itemCategoryCode",
    "thc_product_group",
    "product_family"
]

K_MAP = {
    "itemCategoryCode":  3,
    "thc_product_group": 3,
    "product_family":    4,
}

def de_number_to_float(s):
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    mask = s.str.contains(",", na=False)
    s2 = s.copy()
    s2.loc[mask] = (
        s2.loc[mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce")

print("Loading data...")
df = pd.read_excel(INPUT_FILE)
df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
]
df["Sales_Amount_Actual"]     = de_number_to_float(df["Sales_Amount_Actual"])
df["Cost_Amount_Actual"]      = de_number_to_float(df["Cost_Amount_Actual"])
df["Quantity"]                = pd.to_numeric(df["Quantity"], errors="coerce").abs()
df["total_physical_quantity"] = pd.to_numeric(df["total_physical_quantity"], errors="coerce")
df["Margin"]                  = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]

def cluster_category(cat_col, k):
    print(f"\n{'='*50}")
    print(f"Clustering: {cat_col}  (k={k})")

    agg = (
        df.groupby(cat_col, dropna=True)
        .agg(
            total_quantity=("total_physical_quantity", "sum"),
            total_margin=("Margin", "sum"),
            total_sales=("Sales_Amount_Actual", "sum"),
        )
        .reset_index()
        .dropna()
    )
    print(agg.to_string())

    X        = agg[["total_quantity", "total_margin"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k   = min(len(agg), 8)
    inertia = []
    for ki in range(1, max_k + 1):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    ax.plot(range(1, max_k + 1), inertia, marker='o', color='#2563EB',
            linewidth=2.5, markersize=8)
    ax.set_title(f'Elbow Method — {cat_col}', fontsize=13,
                 fontweight='bold', color='#1E293B')
    ax.set_xlabel('Number of Clusters (k)', fontsize=10, color='#64748B')
    ax.set_ylabel('Inertia', fontsize=10, color='#64748B')
    ax.set_xticks(range(1, max_k + 1))
    ax.tick_params(colors='#64748B', labelsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_edgecolor('#E2E8F0')
    ax.grid(color='#E2E8F0', lw=0.7, ls='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{cat_col}_elbow.png", dpi=180,
                bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  ✔ Saved elbow plot")

    km_final    = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["Cluster"] = km_final.fit_predict(X_scaled)

    agg.to_csv(OUTDIR / f"{cat_col}_clusters.csv", index=False)
    print(f"  ✔ Saved cluster CSV")
    print(agg[[cat_col, "total_quantity", "total_margin", "Cluster"]].to_string())

    CLUSTER_COLORS = ['#2563EB', '#DC2626', '#16A34A', '#CA8A04', '#7C3AED']

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')

    for ci in sorted(agg["Cluster"].unique()):
        sub = agg[agg["Cluster"] == ci]
        ax.scatter(sub["total_quantity"], sub["total_margin"],
                   color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)],
                   s=220, zorder=3, edgecolors='white', linewidth=1.5,
                   label=f'Cluster {ci}')

    for _, row in agg.iterrows():
        ax.annotate(
            row[cat_col],
            xy=(row["total_quantity"], row["total_margin"]),
            xytext=(8, 6), textcoords='offset points',
            fontsize=9, color='#1E293B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec='#CBD5E1', alpha=0.8)
        )

    if agg["total_margin"].min() < 0:
        ax.axhline(0, color='#94A3B8', lw=1, ls='--', alpha=0.6)

    ax.set_xlabel('Total Physical Quantity (g / ml)', fontsize=10, color='#64748B')
    ax.set_ylabel('Total Margin (€)',                 fontsize=10, color='#64748B')
    ax.set_title(f'Product Clusters — {cat_col}\n(Quantity vs Margin)',
                 fontsize=13, fontweight='bold', color='#1E293B')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{int(x/1e6)}M' if abs(x) >= 1e6 else f'{int(x/1e3)}K'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'€{x/1e6:.1f}M' if abs(x) >= 1e6 else f'€{x/1e3:.0f}K'))
    ax.tick_params(colors='#64748B', labelsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_edgecolor('#E2E8F0')
    ax.grid(color='#E2E8F0', lw=0.7, ls='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, facecolor='white', edgecolor='#CBD5E1')
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{cat_col}_clusters_scatter.png", dpi=180,
                bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  ✔ Saved scatter plot")

    return agg

for cat in CATEGORY_COLS:
    if cat in df.columns:
        cluster_category(cat, K_MAP[cat])

print("\n✅ All clustering completed.")
print(f"📁 Output folder: {OUTDIR}")
