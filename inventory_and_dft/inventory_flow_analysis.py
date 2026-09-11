# Input: ILE_Modified.xlsx | Output: inventory_flow_analysis/ (charts + csv_outputs, by category/THC group/product family)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTPUT_DIR = BASE / "inventory_flow_analysis"

DIR_OVERALL = OUTPUT_DIR / "overall"
DIR_ITEM    = OUTPUT_DIR / "itemCategoryCode"
DIR_THC     = OUTPUT_DIR / "thc_product_group"
DIR_FAM     = OUTPUT_DIR / "product_family"
DIR_TOP     = OUTPUT_DIR / "top_products_margin"

CSV_DIR         = OUTPUT_DIR / "csv_outputs"
CSV_OVERALL     = CSV_DIR / "overall"
CSV_ITEM        = CSV_DIR / "itemCategoryCode"
CSV_THC         = CSV_DIR / "thc_product_group"
CSV_FAM         = CSV_DIR / "product_family"
CSV_TOP         = CSV_DIR / "top_products_margin"

for d in [DIR_OVERALL, DIR_ITEM, DIR_THC, DIR_FAM, DIR_TOP,
          CSV_OVERALL, CSV_ITEM, CSV_THC, CSV_FAM, CSV_TOP]:
    d.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_excel(INPUT_FILE)
df["Posting_Date"]        = pd.to_datetime(df["Posting_Date"], errors="coerce")
df["Quantity"]            = pd.to_numeric(df["Quantity"],            errors="coerce")
df["Sales_Amount_Actual"] = pd.to_numeric(df["Sales_Amount_Actual"], errors="coerce")
df["Cost_Amount_Actual"]  = pd.to_numeric(df["Cost_Amount_Actual"],  errors="coerce")
df = df.dropna(subset=["Posting_Date", "Quantity"])

df["Margin"] = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]

sales_df = df[df["Quantity"] < 0].copy()
sales_df["Sales_Quantity"] = -sales_df["Quantity"]

purchase_df = df[df["Quantity"] > 0].copy()
purchase_df["Bought_Quantity"] = df["Quantity"]

def safe_filename(name):
    if pd.isna(name):
        return "Unknown"
    return "".join(c if c.isalnum() else "_" for c in str(name))[:150]

def compute_overall():
    print("\nProcessing: OVERALL DATA")

    daily_inv      = df.groupby("Posting_Date")["Quantity"].sum().reset_index(name="Qty_Move")
    daily_inv["Inventory_On_Hand"] = daily_inv["Qty_Move"].cumsum()

    daily_sales    = sales_df.groupby("Posting_Date")["Sales_Quantity"].sum().reset_index()
    daily_sales["Sales_Cumulative"] = daily_sales["Sales_Quantity"].cumsum()

    daily_purchase = purchase_df.groupby("Posting_Date")["Bought_Quantity"].sum().reset_index()
    daily_purchase["Purchase_Cumulative"] = daily_purchase["Bought_Quantity"].cumsum()

    all_dates = pd.date_range(
        start=daily_inv["Posting_Date"].min(),
        end=daily_inv["Posting_Date"].max(),
        freq="D"
    )

    inv      = daily_inv.set_index("Posting_Date").reindex(all_dates)
    sales    = daily_sales.set_index("Posting_Date").reindex(all_dates)
    purchase = daily_purchase.set_index("Posting_Date").reindex(all_dates)

    inv["Inventory_On_Hand"]        = inv["Inventory_On_Hand"].ffill().fillna(0)
    sales["Sales_Cumulative"]       = sales["Sales_Cumulative"].ffill().fillna(0)
    purchase["Purchase_Cumulative"] = purchase["Purchase_Cumulative"].ffill().fillna(0)

    out_csv = pd.DataFrame({
        "date":                all_dates,
        "inventory_on_hand":   inv["Inventory_On_Hand"].values,
        "sales_cumulative":    sales["Sales_Cumulative"].values,
        "purchase_cumulative": purchase["Purchase_Cumulative"].values,
    })
    out_csv.to_csv(CSV_OVERALL / "OVERALL_FLOW.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(inv.index, inv["Inventory_On_Hand"],        label="Inventory",         linewidth=2.5)
    plt.plot(sales.index, sales["Sales_Cumulative"],     label="Cumulative Sold",   linestyle="--")
    plt.plot(purchase.index, purchase["Purchase_Cumulative"], label="Cumulative Bought", linestyle=":")
    plt.title("Inventory Flow - Entire Dataset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(DIR_OVERALL / "OVERALL_FLOW.png", dpi=300)
    plt.close()
    print("✔ OVERALL DONE")

def compute_flow(category_column, output_dir, csv_dir):
    print(f"\nProcessing: {category_column}")

    daily_inv = (
        df.groupby([category_column, "Posting_Date"])["Quantity"]
        .sum().reset_index(name="Qty_Move")
    )
    daily_inv["Inventory_On_Hand"] = (
        daily_inv.groupby(category_column)["Qty_Move"].cumsum()
    )

    daily_sales = (
        sales_df.groupby([category_column, "Posting_Date"])["Sales_Quantity"]
        .sum().reset_index()
    )
    daily_sales["Sales_Cumulative"] = (
        daily_sales.groupby(category_column)["Sales_Quantity"].cumsum()
    )

    daily_purchase = (
        purchase_df.groupby([category_column, "Posting_Date"])["Bought_Quantity"]
        .sum().reset_index()
    )
    daily_purchase["Purchase_Cumulative"] = (
        daily_purchase.groupby(category_column)["Bought_Quantity"].cumsum()
    )

    categories = daily_inv[category_column].dropna().unique()

    for cat in categories:
        inv      = daily_inv[daily_inv[category_column] == cat]
        sales    = daily_sales[daily_sales[category_column] == cat]
        purchase = daily_purchase[daily_purchase[category_column] == cat]

        if inv.empty:
            continue

        all_dates = pd.date_range(
            start=inv["Posting_Date"].min(),
            end=inv["Posting_Date"].max(),
            freq="D"
        )

        inv      = inv.set_index("Posting_Date").reindex(all_dates)
        sales    = sales.set_index("Posting_Date").reindex(all_dates)
        purchase = purchase.set_index("Posting_Date").reindex(all_dates)

        inv["Inventory_On_Hand"]        = inv["Inventory_On_Hand"].ffill().fillna(0)
        sales["Sales_Cumulative"]       = sales["Sales_Cumulative"].ffill().fillna(0)
        purchase["Purchase_Cumulative"] = purchase["Purchase_Cumulative"].ffill().fillna(0)

        out_csv = pd.DataFrame({
            "date":                all_dates,
            "inventory_on_hand":   inv["Inventory_On_Hand"].values,
            "sales_cumulative":    sales["Sales_Cumulative"].values,
            "purchase_cumulative": purchase["Purchase_Cumulative"].values,
        })
        out_csv.to_csv(csv_dir / f"{safe_filename(cat)}.csv", index=False)

        plt.figure(figsize=(12, 5))
        plt.plot(inv.index, inv["Inventory_On_Hand"],             label="Inventory",         linewidth=2.5)
        plt.plot(sales.index, sales["Sales_Cumulative"],          label="Cumulative Sold",   linestyle="--")
        plt.plot(purchase.index, purchase["Purchase_Cumulative"], label="Cumulative Bought", linestyle=":")
        plt.title(f"{category_column}: {cat}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(output_dir / f"{safe_filename(cat)}.png", dpi=300)
        plt.close()
        print(f"  ✔ {cat}")

def compute_top_products():
    print("\nProcessing Top 10 Products by Margin...")

    top10 = (
        sales_df.groupby("Item_Description")["Margin"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    for product_name in top10.index:
        sub = df[df["Item_Description"] == product_name]
        if sub.empty:
            continue

        daily_inv = sub.groupby("Posting_Date")["Quantity"].sum().reset_index(name="Qty_Move")
        daily_inv["Inventory_On_Hand"] = daily_inv["Qty_Move"].cumsum()

        s = sales_df[sales_df["Item_Description"] == product_name]
        daily_sales = s.groupby("Posting_Date")["Sales_Quantity"].sum().reset_index()
        daily_sales["Sales_Cumulative"] = daily_sales["Sales_Quantity"].cumsum()

        p = purchase_df[purchase_df["Item_Description"] == product_name]
        daily_purchase = p.groupby("Posting_Date")["Bought_Quantity"].sum().reset_index()
        daily_purchase["Purchase_Cumulative"] = daily_purchase["Bought_Quantity"].cumsum()

        all_dates = pd.date_range(
            start=daily_inv["Posting_Date"].min(),
            end=daily_inv["Posting_Date"].max(),
            freq="D"
        )

        inv      = daily_inv.set_index("Posting_Date").reindex(all_dates)
        sales    = daily_sales.set_index("Posting_Date").reindex(all_dates)
        purchase = daily_purchase.set_index("Posting_Date").reindex(all_dates)

        inv["Inventory_On_Hand"]        = inv["Inventory_On_Hand"].ffill().fillna(0)
        sales["Sales_Cumulative"]       = sales["Sales_Cumulative"].ffill().fillna(0)
        purchase["Purchase_Cumulative"] = purchase["Purchase_Cumulative"].ffill().fillna(0)

        out_csv = pd.DataFrame({
            "date":                all_dates,
            "inventory_on_hand":   inv["Inventory_On_Hand"].values,
            "sales_cumulative":    sales["Sales_Cumulative"].values,
            "purchase_cumulative": purchase["Purchase_Cumulative"].values,
        })
        out_csv.to_csv(CSV_TOP / f"{safe_filename(product_name)}.csv", index=False)

        plt.figure(figsize=(12, 5))
        plt.plot(inv.index, inv["Inventory_On_Hand"],             label="Inventory",         linewidth=2.5)
        plt.plot(sales.index, sales["Sales_Cumulative"],          label="Cumulative Sold",   linestyle="--")
        plt.plot(purchase.index, purchase["Purchase_Cumulative"], label="Cumulative Bought", linestyle=":")
        plt.title(product_name)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(DIR_TOP / f"{safe_filename(product_name)}.png", dpi=300)
        plt.close()
        print(f"  ✔ {product_name}")

compute_overall()
compute_flow("itemCategoryCode",  DIR_ITEM, CSV_ITEM)
compute_flow("thc_product_group", DIR_THC,  CSV_THC)
compute_flow("product_family",    DIR_FAM,  CSV_FAM)
compute_top_products()

print("\nALL DONE 🚀")
print(f"📁 CSVs saved to: {CSV_DIR}")
