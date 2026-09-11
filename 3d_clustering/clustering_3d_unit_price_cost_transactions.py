# Input: ILE_Modified.xlsx | Output: unit_price_cost_transactions_3d_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "unit_price_cost_transactions_3d_clustering"
OUTDIR.mkdir(exist_ok=True)

CATEGORY_COLS = ["itemCategoryCode", "thc_product_group", "product_family"]
K_MAP = {"itemCategoryCode": 3, "thc_product_group": 3, "product_family": 4}

COLORS     = ["#2563EB", "#DC2626", "#16A34A", "#CA8A04", "#7C3AED"]
BG_FIG     = "#F8F9FA"
BG_AX      = "#FFFFFF"
TEXT_MAIN  = "#1E293B"
TEXT_SUB   = "#64748B"
GRID_COLOR = "#E2E8F0"
SPINE_COLOR= "#E2E8F0"

print("Loading data...")
df = pd.read_excel(INPUT_FILE)
df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
].copy()

df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["unit_cost"]  = pd.to_numeric(df["unit_cost"],  errors="coerce")

print(f"Rows after filter: {len(df)}")

def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.grid(True, linestyle="--", color=GRID_COLOR, alpha=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_SUB, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SUB)
    ax.yaxis.label.set_color(TEXT_SUB)


def scatter_2d(ax, df, xcol, ycol, cat_col, xlabel, ylabel, title):
    style_ax(ax)
    texts = []
    for ci in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == ci]
        ax.scatter(sub[xcol], sub[ycol],
                   color=COLORS[ci % len(COLORS)], s=220,
                   edgecolors="white", linewidth=2, zorder=3,
                   label=f"Cluster {ci}")
        for _, row in sub.iterrows():
            texts.append(ax.text(
                row[xcol], row[ycol], str(row[cat_col]),
                fontsize=7.5, fontweight="bold", color=TEXT_MAIN,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.9)
            ))
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.7))
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SUB)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SUB)
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_MAIN, pad=8)
    leg = ax.legend(fontsize=8, loc="best", frameon=True,
                    facecolor="white", edgecolor="#CBD5E1")
    for t in leg.get_texts():
        t.set_color(TEXT_MAIN)


def run_clustering(category_col):
    print(f"\nProcessing: {category_col}")

    folder = OUTDIR / f"{category_col}_clusters"
    folder.mkdir(exist_ok=True)

    agg = (
        df.groupby(category_col)
        .agg(
            avg_unit_price   =("unit_price", "mean"),
            avg_unit_cost    =("unit_cost",  lambda x: x.abs().mean()),
            total_transactions=(category_col, "count"),
        )
        .reset_index()
        .dropna()
    )

    print("\nAggregated Data:")
    print(agg.to_string())

    X        = agg[["avg_unit_price", "avg_unit_cost", "total_transactions"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k   = min(len(agg), 8)
    inertia = []
    for ki in range(1, max_k + 1):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    k              = K_MAP[category_col]
    km_final       = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = km_final.fit_predict(X_scaled)

    agg.to_csv(folder / f"{category_col}_clusters.csv", index=False)
    print(f"  ✔ Saved CSV")

    label = category_col.replace("_", " ").title()

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle(
        f"3D Clustering Summary — {label}\n(Unit Price + Unit Cost + Total Transactions, k={k})",
        fontsize=15, fontweight="bold", color=TEXT_MAIN, y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    ax_elbow = fig.add_subplot(gs[0, 0])
    style_ax(ax_elbow)
    ax_elbow.plot(range(1, max_k + 1), inertia,
                  color="#2563EB", linewidth=2.5, marker="o", markersize=9,
                  markerfacecolor="#2563EB", markeredgecolor="white", markeredgewidth=1.5)
    ax_elbow.set_xlabel("Number of Clusters (k)", fontsize=9, color=TEXT_SUB)
    ax_elbow.set_ylabel("Inertia", fontsize=9, color=TEXT_SUB)
    ax_elbow.set_title(f"Elbow Method — {label}", fontsize=10,
                       fontweight="bold", color=TEXT_MAIN, pad=8)
    ax_elbow.set_xticks(range(1, max_k + 1))

    ax3d = fig.add_subplot(gs[0, 1:], projection="3d")
    ax3d.set_facecolor(BG_AX)
    for ci in sorted(agg["cluster"].unique()):
        sub = agg[agg["cluster"] == ci]
        ax3d.scatter(sub["avg_unit_price"], sub["avg_unit_cost"], sub["total_transactions"],
                     color=COLORS[ci % len(COLORS)], s=200,
                     edgecolors="white", linewidth=1.5,
                     label=f"Cluster {ci}", depthshade=True)
        for _, row in sub.iterrows():
            ax3d.text(row["avg_unit_price"], row["avg_unit_cost"], row["total_transactions"],
                      str(row[category_col]), fontsize=8, fontweight="bold", color=TEXT_MAIN,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.85))
    ax3d.set_xlabel("Avg Unit Price (€)", fontsize=8, color=TEXT_SUB, labelpad=8)
    ax3d.set_ylabel("Avg Unit Cost (€)", fontsize=8, color=TEXT_SUB, labelpad=8)
    ax3d.set_zlabel("Total Transactions", fontsize=8, color=TEXT_SUB, labelpad=8)
    ax3d.set_title(f"3D Scatter — {label}", fontsize=10,
                   fontweight="bold", color=TEXT_MAIN, pad=12)
    ax3d.view_init(elev=20, azim=45)
    leg3d = ax3d.legend(fontsize=8, loc="upper left", frameon=True,
                        facecolor="white", edgecolor="#CBD5E1")
    for t in leg3d.get_texts():
        t.set_color(TEXT_MAIN)

    ax1  = fig.add_subplot(gs[1, 0])
    ax2  = fig.add_subplot(gs[1, 1])
    ax3p = fig.add_subplot(gs[1, 2])

    scatter_2d(ax1,  agg, "avg_unit_price", "avg_unit_cost",       category_col,
               "Avg Unit Price (€)",    "Avg Unit Cost (€)",    "Unit Price vs Unit Cost")
    scatter_2d(ax2,  agg, "avg_unit_price", "total_transactions",   category_col,
               "Avg Unit Price (€)",    "Total Transactions",   "Unit Price vs Transactions")
    scatter_2d(ax3p, agg, "avg_unit_cost",  "total_transactions",   category_col,
               "Avg Unit Cost (€)",     "Total Transactions",   "Unit Cost vs Transactions")

    plt.tight_layout()
    out_fig = folder / f"{category_col}_3d_summary.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close()
    print(f"  ✔ Saved summary figure: {out_fig}")

    fig_e, ax_e = plt.subplots(figsize=(8, 5))
    fig_e.patch.set_facecolor(BG_FIG)
    style_ax(ax_e)
    ax_e.plot(range(1, max_k + 1), inertia,
              color="#2563EB", linewidth=2.5, marker="o", markersize=9,
              markerfacecolor="#2563EB", markeredgecolor="white", markeredgewidth=1.5)
    ax_e.set_title(f"Elbow Method — {label}\n(Unit Price + Unit Cost + Total Transactions)",
                   fontsize=12, fontweight="bold", color=TEXT_MAIN, pad=10)
    ax_e.set_xlabel("Number of Clusters (k)", fontsize=10, color=TEXT_SUB)
    ax_e.set_ylabel("Inertia", fontsize=10, color=TEXT_SUB)
    ax_e.set_xticks(range(1, max_k + 1))
    fig_e.tight_layout()
    fig_e.savefig(folder / f"{category_col}_elbow.png", dpi=150,
                  bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig_e)
    print(f"  ✔ Saved standalone elbow")
    print(f"  ✔ Completed: {category_col}")


for cat in CATEGORY_COLS:
    if cat in df.columns:
        run_clustering(cat)

print("\n✅ ALL 3D CLUSTERING COMPLETED")
print(f"📁 Output folder: {OUTDIR}")
