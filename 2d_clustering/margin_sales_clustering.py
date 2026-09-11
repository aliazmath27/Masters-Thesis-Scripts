# Input: ILE_Modified.xlsx | Output: margin_sales_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "margin_sales_clustering"
OUTDIR.mkdir(exist_ok=True)

CATEGORY_COLS = [
    "itemCategoryCode",
    "thc_product_group",
    "product_family",
]

K_MAP = {
    "itemCategoryCode":  3,
    "thc_product_group": 3,
    "product_family":    4,
}

def de_number_to_float(s: pd.Series) -> pd.Series:
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
df["Sales_Amount_Actual"] = de_number_to_float(df["Sales_Amount_Actual"])
df["Cost_Amount_Actual"]  = de_number_to_float(df["Cost_Amount_Actual"])
df["Margin"]              = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]

def run_clustering(cat_col, k):
    print(f"\n{'='*50}")
    print(f"Clustering: {cat_col}  (k={k})")

    agg = (
        df.groupby(cat_col, dropna=True)
        .agg(
            Sales=("Sales_Amount_Actual", "sum"),
            Margin=("Margin", "sum"),
        )
        .reset_index()
        .dropna()
    )
    print(agg.to_string())

    X        = agg[["Sales", "Margin"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k   = min(len(agg), 8)
    inertia = []
    for ki in range(1, max_k + 1):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    km_final       = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["Cluster"] = km_final.fit_predict(X_scaled)
    agg.to_csv(OUTDIR / f"{cat_col}_clusters.csv", index=False)
    print(f"  ✔ Saved cluster CSV")

    fig, (ax_elbow, ax_scatter) = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor('#F8F9FA')
    ax_elbow.set_facecolor('#FFFFFF')
    ax_scatter.set_facecolor('#FFFFFF')

    C_TEXT = '#1E293B'; C_SUB = '#64748B'; C_GRID = '#E2E8F0'
    COLORS = ['#2563EB', '#DC2626', '#16A34A', '#CA8A04', '#7C3AED']

    ks = list(range(1, max_k + 1))
    ax_elbow.plot(ks, inertia, marker='o', color='#2563EB',
                  linewidth=2.5, markersize=9, zorder=3)
    ax_elbow.set_title(f'Elbow Method — {cat_col}',
                       fontsize=12, fontweight='bold', color=C_TEXT, pad=10)
    ax_elbow.set_xlabel('Number of Clusters (k)', fontsize=10.5, color=C_SUB)
    ax_elbow.set_ylabel('Inertia', fontsize=10.5, color=C_SUB)
    ax_elbow.set_xticks(ks)
    ax_elbow.tick_params(colors=C_SUB, labelsize=9)
    for sp in ['top','right']: ax_elbow.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax_elbow.spines[sp].set_edgecolor(C_GRID)
    ax_elbow.grid(color=C_GRID, lw=0.7, ls='--')
    ax_elbow.set_axisbelow(True)

    for ci in sorted(agg["Cluster"].unique()):
        sub = agg[agg["Cluster"] == ci]
        ax_scatter.scatter(sub["Sales"], sub["Margin"],
                           color=COLORS[ci % len(COLORS)],
                           s=240, zorder=3, edgecolors='white', linewidth=2,
                           label=f'Cluster {ci}')

    texts = []
    for _, r in agg.iterrows():
        t = ax_scatter.text(r["Sales"], r["Margin"], str(r[cat_col]),
                            fontsize=9.5, color=C_TEXT, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      ec='#CBD5E1', alpha=0.9))
        texts.append(t)

    adjust_text(texts, ax=ax_scatter,
                arrowprops=dict(arrowstyle='-', color='#94A3B8', lw=0.8),
                expand_points=(2.0, 2.0), expand_text=(1.8, 1.8),
                force_points=(0.8, 0.8))

    if agg["Margin"].min() < 0:
        ax_scatter.axhline(0, color='#94A3B8', lw=1, ls='--', alpha=0.7)

    ax_scatter.set_title(f'Sales vs Margin — {cat_col}',
                         fontsize=12, fontweight='bold', color=C_TEXT, pad=10)
    ax_scatter.set_xlabel('Total Sales (€)', fontsize=10.5, color=C_SUB)
    ax_scatter.set_ylabel('Total Margin (€)', fontsize=10.5, color=C_SUB)
    ax_scatter.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f'€{x/1e6:.1f}M' if abs(x)>=1e6 else f'€{x/1e3:.0f}K'))
    ax_scatter.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f'€{x/1e6:.1f}M' if abs(x)>=1e6 else f'€{x/1e3:.0f}K'))
    ax_scatter.tick_params(colors=C_SUB, labelsize=9)
    for sp in ['top','right']: ax_scatter.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax_scatter.spines[sp].set_edgecolor(C_GRID)
    ax_scatter.grid(color=C_GRID, lw=0.7, ls='--')
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(fontsize=9.5, facecolor='white', edgecolor='#CBD5E1',
                      labelcolor=C_TEXT, loc='upper left')

    fig.suptitle(f'K-Means Clustering — {cat_col}  (Sales vs Margin)',
                 fontsize=14, fontweight='bold', color=C_TEXT, y=1.01)

    plt.tight_layout(w_pad=4)
    plt.savefig(OUTDIR / f"{cat_col}_combined.png", dpi=180,
                bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  ✔ Saved combined figure")

for cat in CATEGORY_COLS:
    if cat in df.columns:
        run_clustering(cat, K_MAP[cat])

print("\n✅ All clustering completed.")
print(f"📁 Output folder: {OUTDIR}")
