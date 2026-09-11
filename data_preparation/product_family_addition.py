# Input: ILE_Modified.xlsx | Output: Product family addition 2/ILE_product_family_added.xlsx

import os
import pandas as pd

BASE = r"C:\Users\aliaz\IU\Thesis"

INPUT_FILE = os.path.join(BASE, "ILE_Modified.xlsx")

OUTDIR = os.path.join(BASE, "Product family addition 2")
os.makedirs(OUTDIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTDIR, "ILE_product_family_added.xlsx")

def classify_product_family(desc: str) -> str:

    if not isinstance(desc, str):
        return "Others"

    desc = desc.upper()

    if "ENUA" in desc:
        return "Enua"

    elif "CALAMA" in desc:
        return "Calama"

    elif "DROP" in desc:
        return "Drop"

    elif (
        "CORE" in desc
        or "ICC" in desc
        or "JM7" in desc
        or "MOONBOW-112" in desc
        or "OZK X CC" in desc
    ):
        return "Core"

    elif (
        "CRAFT" in desc
        or "BLUE Z" in desc
        or "TRICOLO" in desc
        or "PERMANENT MARKER" in desc
    ):
        return "Craft"

    elif "TRUU" in desc:
        return "Truu"

    elif "OKTONIA" in desc:
        return "Oktonia"

    elif "PURE" in desc:
        return "Pure"

    else:
        return "Others"


df = pd.read_excel(INPUT_FILE)

if "Item_Description_norm" not in df.columns:
    raise KeyError(
        "Item_Description_norm column not found in ILE_Modified.xlsx. "
        "Please ensure normalization was applied earlier."
    )

df["product_family"] = df["Item_Description_norm"].apply(
    classify_product_family
)

print("\n🔍 Product family distribution:")
print(df["product_family"].value_counts())

print("\n🔍 Sample mappings:")
print(
    df[["Item_Description_norm", "product_family"]]
    .drop_duplicates()
    .head(25)
)

df.to_excel(OUTPUT_FILE, index=False)

print("\n✅ File saved successfully:")
print(OUTPUT_FILE)
