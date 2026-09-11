# Input: ILE_Modified.xlsx | Output: sales_margin_transactions_3d_clustering/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR     = BASE / "sales_margin_transactions_3d_clustering"
OUTDIR.mkdir(exist_ok=True)

CATEGORY_COLS = ["itemCategoryCode", "thc_product_group", "product_family"]
K_MAP = {"itemCategoryCode": 3, "thc_product_group": 3, "product_family": 4}

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#CA8A04", "#7C3AED"]
BG_FIG     = "#F8F9FA"
BG_AX      = "#FFFFFF"
TEXT_MAIN  = "#1E293B"
TEXT_SUB   = "#64748B"
GRID_COLOR = "#E2E8F0"

print("Loading data...")
df = pd.read_excel(INPUT_FILE)
df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
    (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
].copy()

df["Sales_Amount_Actual"] = pd.to_numeric(df["Sales_Amount_Actual"], errors="coerce")
df["unit_price"]          = pd.to_numeric(df["unit_price"],          errors="coerce")
df["unit_cost"]           = pd.to_numeric(df["unit_cost"],           errors="coerce")
df["unit_margin"]         = df["unit_price"] + df["unit_cost"]

print(f"Rows after filter: {len(df)}")

def style_ax2d(ax):
    ax.set_facecolor(BG_AX)
    ax.grid(True, linestyle="--", color=GRID_COLOR, alpha=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SUB, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SUB)
    ax.yaxis.label.set_color(TEXT_SUB)


def add_labels_3d(ax, agg, cat_col):
    for _, row in agg.iterrows():
        ax.text(row["total_sales"], row["margin_per_unit"], row["total_transactions"],
                str(row[cat_col]), fontsize=8, fontweight="bold", color=TEXT_MAIN,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.85))


def run_clustering(category_col):
    print(f"\nProcessing: {category_col}")

    folder = OUTDIR / f"{category_col}_clusters"
    folder.mkdir(exist_ok=True)

    agg = (
        df.groupby(category_col)
        .agg(
            total_sales      =("Sales_Amount_Actual", "sum"),
            margin_per_unit  =("unit_margin",         "mean"),
            total_transactions=("unit_margin",         "count"),
        )
        .reset_index()
        .dropna()
    )

    print("\nAggregated Data:")
    print(agg.to_string())

    X        = agg[["total_sales", "margin_per_unit", "total_transactions"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k   = min(len(agg), 8)
    inertia = []
    for ki in range(1, max_k + 1):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig_e, ax_e = plt.subplots(figsize=(8, 5))
    fig_e.patch.set_facecolor(BG_FIG)
    style_ax2d(ax_e)
    ax_e.plot(range(1, max_k + 1), inertia,
              marker="o", color="#2563EB", linewidth=2.5,
              markersize=9, markerfacecolor="#2563EB",
              markeredgecolor="white", markeredgewidth=1.5)
    ax_e.set_title(f"Elbow Method — {category_col}\n(Sales + Margin per Unit + Transactions)",
                   fontsize=12, fontweight="bold", color=TEXT_MAIN, pad=10)
    ax_e.set_xlabel("Number of Clusters (k)", fontsize=10, color=TEXT_SUB)
    ax_e.set_ylabel("Inertia", fontsize=10, color=TEXT_SUB)
    ax_e.set_xticks(range(1, max_k + 1))
    fig_e.tight_layout()
    fig_e.savefig(folder / f"{category_col}_elbow.png", dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig_e)
    print(f"  ✔ Saved elbow plot")

    k              = K_MAP[category_col]
    km_final       = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = km_final.fit_predict(X_scaled)

    agg.to_csv(folder / f"{category_col}_clusters.csv", index=False)
    print(f"  ✔ Saved CSV")

    fig_3d = plt.figure(figsize=(12, 8))
    fig_3d.patch.set_facecolor(BG_FIG)
    ax3d = fig_3d.add_subplot(111, projection="3d")
    ax3d.set_facecolor(BG_AX)

    for ci in sorted(agg["cluster"].unique()):
        sub = agg[agg["cluster"] == ci]
        ax3d.scatter(sub["total_sales"], sub["margin_per_unit"], sub["total_transactions"],
                     color=COLORS[ci % len(COLORS)], s=200,
                     edgecolors="white", linewidth=1.5,
                     label=f"Cluster {ci}", depthshade=True)

    add_labels_3d(ax3d, agg, category_col)

    ax3d.set_xlabel("Total Sales (€)", fontsize=9, color=TEXT_SUB, labelpad=8)
    ax3d.set_ylabel("Margin per Unit (€)", fontsize=9, color=TEXT_SUB, labelpad=8)
    ax3d.set_zlabel("Total Transactions", fontsize=9, color=TEXT_SUB, labelpad=8)
    ax3d.set_title(f"3D Clustering — {category_col}\n(Sales + Margin per Unit + Transactions, k={k})",
                   fontsize=12, fontweight="bold", color=TEXT_MAIN, pad=15)
    ax3d.legend(fontsize=9, loc="upper left",
                frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax3d.view_init(elev=20, azim=45)

    fig_3d.tight_layout()
    fig_3d.savefig(folder / f"{category_col}_3d_scatter.png", dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig_3d)
    print(f"  ✔ Saved 3D scatter")

    fig_p, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig_p.patch.set_facecolor(BG_FIG)
    fig_p.suptitle(f"2D Projections — {category_col}  (Sales + Margin per Unit + Transactions, k={k})",
                   fontsize=13, fontweight="bold", color=TEXT_MAIN, y=1.01)

    projection_pairs = [
        ("total_sales",        "margin_per_unit",   "Total Sales (€)",    "Margin per Unit (€)",  "Sales vs Margin per Unit"),
        ("total_sales",        "total_transactions","Total Sales (€)",    "Total Transactions",   "Sales vs Transactions"),
        ("margin_per_unit",    "total_transactions","Margin per Unit (€)","Total Transactions",   "Margin per Unit vs Transactions"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, projection_pairs):
        style_ax2d(ax)
        has_negative = (agg[ycol] < 0).any() or (agg[xcol] < 0).any()
        if has_negative:
            ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="-", zorder=1)

        for ci in sorted(agg["cluster"].unique()):
            sub = agg[agg["cluster"] == ci]
            ax.scatter(sub[xcol], sub[ycol],
                       color=COLORS[ci % len(COLORS)], s=200,
                       edgecolors="white", linewidth=1.5,
                       label=f"Cluster {ci}", zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(str(row[category_col]),
                            xy=(row[xcol], row[ycol]),
                            fontsize=8, fontweight="bold", color=TEXT_MAIN,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.9),
                            xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SUB)
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SUB)
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT_MAIN, pad=8)
        ax.legend(fontsize=8, loc="best", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    fig_p.tight_layout()
    fig_p.savefig(folder / f"{category_col}_2d_projections.png", dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig_p)
    print(f"  ✔ Saved 2D projections")
    print(f"  ✔ Completed: {category_col}")


for cat in CATEGORY_COLS:
    if cat in df.columns:
        run_clustering(cat)

print("\n✅ ALL 3D CLUSTERING COMPLETED")
print(f"📁 Output folder: {OUTDIR}")
