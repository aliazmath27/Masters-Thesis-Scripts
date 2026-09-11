# Input: ILE_product_family_added.xlsx | Output: Filtered_Data/ILE_2023_onwards.xlsx

import os
import pandas as pd

BASE = r"C:\Users\aliaz\IU\Thesis"

INPUT_FILE = os.path.join(BASE, "ILE_product_family_added.xlsx")

OUTDIR = os.path.join(BASE, "Filtered_Data")
os.makedirs(OUTDIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTDIR, "ILE_2023_onwards.xlsx")

print("Loading dataset...")
df = pd.read_excel(INPUT_FILE)

if "Posting_Date" not in df.columns:
    raise KeyError("Posting_Date column not found in dataset.")

df["Posting_Date"] = pd.to_datetime(df["Posting_Date"], errors="coerce")

df_filtered = df[df["Posting_Date"] >= "2023-01-01"]

print("\nOriginal rows:", len(df))
print("Rows after filtering:", len(df_filtered))

df_filtered.to_excel(OUTPUT_FILE, index=False)

print("\n✅ Filtered dataset saved to:")
print(OUTPUT_FILE)
