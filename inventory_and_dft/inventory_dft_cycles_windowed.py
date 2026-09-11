# Input: ILE_Modified.xlsx | Output: inventory_dft_cycles_windowed/ (charts + csv_outputs)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, detrend


BASE       = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTPUT_DIR = BASE / "inventory_dft_cycles_windowed"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_DIR = OUTPUT_DIR / "csv_outputs"
CSV_DIR.mkdir(exist_ok=True)


print("Loading dataset...")

df = pd.read_excel(INPUT_FILE)

df["Posting_Date"] = pd.to_datetime(df["Posting_Date"], errors="coerce")
df["Quantity"]     = pd.to_numeric(df["Quantity"], errors="coerce")

df = df.dropna(subset=["Posting_Date", "Quantity"])


sales_df = df[
    (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale")
].copy()

sales_df["Demand"] = -sales_df["Quantity"]


WINDOWS = [
    (2,  7),
    (7,  14),
    (14, 30),
    (30, 90),
]


def run_fft(sub_df, label, png_folder, csv_folder):

    if sub_df.empty:
        return

    daily = (
        sub_df.groupby("Posting_Date")["Demand"]
        .sum()
        .sort_index()
    )

    full_index = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    daily  = daily.reindex(full_index, fill_value=0)
    signal = daily.values

    fft_vals  = np.fft.fft(signal)
    magnitude = np.abs(fft_vals)
    freq      = np.fft.fftfreq(len(signal), d=1)

    mask      = freq > 0
    freq      = freq[mask]
    magnitude = magnitude[mask]
    cycles    = 1 / freq

    mask_cycles = (cycles >= 2) & (cycles <= 90)
    cycles      = cycles[mask_cycles]
    magnitude   = magnitude[mask_cycles]

    order     = np.argsort(cycles)
    cycles    = cycles[order]
    magnitude = magnitude[order]

    peaks, _    = find_peaks(magnitude, distance=5)
    peak_cycles = cycles[peaks]
    peak_mag    = magnitude[peaks]

    threshold      = np.percentile(magnitude, 90)
    dominant_mask  = peak_mag >= threshold
    dominant_cycles = peak_cycles[dominant_mask]
    dominant_mag    = peak_mag[dominant_mask]

    safe_name = "".join(x if x.isalnum() else "_" for x in label)


    for w_min, w_max in WINDOWS:

        mask_w   = (cycles >= w_min) & (cycles < w_max)
        w_cycles = cycles[mask_w]
        w_mag    = magnitude[mask_w]

        if len(w_cycles) == 0:
            continue

        mask_dom_w  = (dominant_cycles >= w_min) & (dominant_cycles < w_max)
        dom_cycles_w = dominant_cycles[mask_dom_w]
        dom_mag_w    = dominant_mag[mask_dom_w]

        print(f"\n{label} | Window {w_min}-{w_max} days")
        if len(dom_cycles_w) == 0:
            print("  No dominant cycles")
        else:
            for c in dom_cycles_w:
                print(f"  {c:.1f} days")

        is_dominant = np.isin(w_cycles, dom_cycles_w)
        csv_df = pd.DataFrame({
            "cycle_length_days": np.round(w_cycles, 4),
            "fft_magnitude":     np.round(w_mag,    4),
            "is_dominant":       is_dominant,
        })
        csv_path = csv_folder / f"{safe_name}_{w_min}_{w_max}.csv"
        csv_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(w_cycles, w_mag, linewidth=2)
        plt.scatter(dom_cycles_w, dom_mag_w, color="red")

        for c, m in zip(dom_cycles_w, dom_mag_w):
            plt.annotate(
                f"({c:.1f}, {m:.2f})",
                (c, m),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        plt.xlabel("Cycle Length (Days)")
        plt.ylabel("Normalized FFT Magnitude")
        plt.title(f"{label}\nWindow {w_min}-{w_max} days (Demand Only)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlim(w_min, w_max)
        plt.tight_layout()
        plt.savefig(png_folder / f"{safe_name}_{w_min}_{w_max}.png", dpi=300)
        plt.close()


def run_category(category_col):

    print(f"\nRunning category: {category_col}")

    png_folder = OUTPUT_DIR / category_col
    png_folder.mkdir(exist_ok=True)

    csv_folder = CSV_DIR / category_col
    csv_folder.mkdir(exist_ok=True)

    values = sales_df[category_col].dropna().unique()

    for val in values:
        sub = sales_df[sales_df[category_col] == val]
        run_fft(
            sub,
            f"{category_col}: {val}",
            png_folder,
            csv_folder,
        )


print("\nCalculating top 10 margin products...")

df["Margin"] = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]

margin_table = (
    df.groupby("Item_Description_norm")["Margin"]
    .sum()
    .sort_values(ascending=False)
)

top10 = margin_table.head(10).index

png_folder_top = OUTPUT_DIR / "top10_products"
png_folder_top.mkdir(exist_ok=True)

csv_folder_top = CSV_DIR / "top10_products"
csv_folder_top.mkdir(exist_ok=True)

for product in top10:
    sub = sales_df[sales_df["Item_Description_norm"] == product]
    run_fft(
        sub,
        f"Top Product: {product}",
        png_folder_top,
        csv_folder_top,
    )


run_category("product_family")
run_category("itemCategoryCode")
run_category("thc_product_group")

print("\n✅ CLEAN DEMAND CYCLE ANALYSIS DONE")
print(f"📁 CSVs saved to: {CSV_DIR}")
