# Input: ILE_Modified.xlsx | Output: velocity_cv_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "velocity_cv_clustering"
OUTDIR.mkdir(exist_ok=True)
DATE_COL   = "Posting_Date"

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
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df = df.dropna(subset=[DATE_COL, "Quantity"])

sales_df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
].copy()
sales_df["Quantity"] = sales_df["Quantity"].abs()

def run_clustering(cat_col, k):
    print(f"\n{'='*50}")
    print(f"Clustering: {cat_col}  (k={k})")

    daily = (
        sales_df.groupby([cat_col, DATE_COL])["Quantity"]
        .sum()
        .reset_index()
    )

    rows = []
    for cat in daily[cat_col].dropna().unique():
        sub = daily[daily[cat_col] == cat].copy()
        if sub.empty:
            continue

        total_quantity = sub["Quantity"].sum()
        active_days    = sub[DATE_COL].nunique()
        velocity       = total_quantity / active_days if active_days > 0 else 0

        mean_sales = sub["Quantity"].mean()
        std_sales  = sub["Quantity"].std()
        if mean_sales == 0 or pd.isna(mean_sales):
            cv = 0
        else:
            cv = std_sales / mean_sales

        rows.append({cat_col: cat, "velocity": velocity, "cv": cv})

    agg = pd.DataFrame(rows)
    print(agg.to_string())

    X        = agg[["velocity", "cv"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k   = min(len(agg), 8)
    inertia = []
    for ki in range(1, max_k + 1):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    km_final       = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = km_final.fit_predict(X_scaled)
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

    for ci in sorted(agg["cluster"].unique()):
        sub = agg[agg["cluster"] == ci]
        ax_scatter.scatter(sub["velocity"], sub["cv"],
                           color=COLORS[ci % len(COLORS)],
                           s=240, zorder=3, edgecolors='white', linewidth=2,
                           label=f'Cluster {ci}')

    texts = []
    for _, r in agg.iterrows():
        t = ax_scatter.text(r["velocity"], r["cv"], str(r[cat_col]),
                            fontsize=9.5, color=C_TEXT, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      ec='#CBD5E1', alpha=0.9))
        texts.append(t)

    adjust_text(texts, ax=ax_scatter,
                arrowprops=dict(arrowstyle='-', color='#94A3B8', lw=0.8),
                expand_points=(2.0, 2.0), expand_text=(1.8, 1.8),
                force_points=(0.8, 0.8))

    ax_scatter.set_title(f'Velocity vs CV — {cat_col}',
                         fontsize=12, fontweight='bold', color=C_TEXT, pad=10)
    ax_scatter.set_xlabel('Velocity (Units per Active Day)', fontsize=10.5, color=C_SUB)
    ax_scatter.set_ylabel('Coefficient of Variation (CV)', fontsize=10.5, color=C_SUB)
    ax_scatter.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f'{x/1e3:.0f}K' if abs(x)>=1e3 else f'{x:.0f}'))
    ax_scatter.tick_params(colors=C_SUB, labelsize=9)
    for sp in ['top','right']: ax_scatter.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax_scatter.spines[sp].set_edgecolor(C_GRID)
    ax_scatter.grid(color=C_GRID, lw=0.7, ls='--')
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(fontsize=9.5, facecolor='white', edgecolor='#CBD5E1',
                      labelcolor=C_TEXT, loc='upper right')

    fig.suptitle(f'K-Means Clustering — {cat_col}  (Velocity vs CV)',
                 fontsize=14, fontweight='bold', color=C_TEXT, y=1.01)

    plt.tight_layout(w_pad=4)
    plt.savefig(OUTDIR / f"{cat_col}_combined.png", dpi=180,
                bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  ✔ Saved combined figure")

for cat in CATEGORY_COLS:
    if cat in sales_df.columns:
        run_clustering(cat, K_MAP[cat])

print("\n✅ All clustering completed.")
print(f"📁 Output folder: {OUTDIR}")
