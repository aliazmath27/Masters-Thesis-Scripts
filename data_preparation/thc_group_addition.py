# Input: ItemLedgerEntriesCustom_all_rows.xlsx + workflowItems_all_rows.csv | Output: Feature addition/ILE_thcProductgroup_added.xlsx

import os
import re
import numpy as np
import pandas as pd

BASE = r"C:\Users\aliaz\IU\Thesis"

LEDGER_PATH = os.path.join(BASE, "ItemLedgerEntriesCustom_all_rows.xlsx")
WORKFLOW_PATH = os.path.join(BASE, "workflowItems_all_rows.csv")

OUTDIR = os.path.join(BASE, "Feature addition")
os.makedirs(OUTDIR, exist_ok=True)

MAIN_OUTFILE = os.path.join(OUTDIR, "ILE_thcProductgroup_added.xlsx")
UNCAT_OUTFILE = os.path.join(OUTDIR, "flowers_still_not_categorized.xlsx")

MANUAL_THC_MAP = {
    "BB REGULAR DROP": 25,
    "BH EPIC DROP": 26,
    "CR EPIC DROP": 30,
    "CV REGULAR DROP": 25,
    "EPIC DROP": 30,

    "AG EPIC DROP": 30,
    "BB STRONG DROP": 27,
    "CB REGULAR DROP": 25,
    "CR REGULAR DROP": 25,

    "BEDICA": 14,
    "BEDIOL": 6,
    "BEDROBINOL": 14,
    "BEDROCAN": 22,
    "BEDROLITE": 1,

    "BH LEAN DROP": 22,
    "HS LEAN DROP": 20,
    "HS LIGHT DROP": 17,
    "HS LOW DROP": 18,

    "LEAN DROP": 24,
    "MC EPIC DROP": 29,
    "MC LEAN DROP": 24,
    "MC LIGHT DROP": 17,
    "MC LOW DROP": 17,
    "MC REGULAR DROP": 24,
    "MC STRONG DROP": 27,

    "PG REGULAR DROP": 30,
    "PG STRONG DROP": 27,

    "RF EPIC DROP": 31,
    "RF LEAN DROP": 22,
    "SC LIGHT DROP": 20,
    "SC LOW DROP": 17,
    "SC REGULAR DROP": 25,
    "SFF STRONG DROP": 27,

    "GBH EPIC DROP": 30,
    "GBH REGULAR DROP": 25,
    "GBH STRONG DROP": 26,

    "REGULAR DROP": 24,
    "STRONG DROP": 27,

    "MEDIZINAL-CANNABISBLÜTEN TYP 1 DEMECAN": 21,
    "MEDIZINAL-CANNABISBLÜTEN TYP 2 DEMECAN": 17,
    "TYP 1 DEMECAN FORTE": 22,
}

FORCE_EXTRACT_PRODUCTS = {
    "LABEL DEMECAN REZEPTURSET"
}

def normalize_description(desc):
    if not isinstance(desc, str):
        return ""

    desc = desc.upper()
    desc = re.sub(r"RÜCKSTELLMUSTER\s*", "", desc)
    desc = re.sub(r"\(.*?\)", "", desc)
    desc = re.sub(r"\b\d+\s*G\b", "", desc)
    desc = re.sub(r"\s+", " ", desc).strip()

    return desc


def extract_thc(description):
    if not isinstance(description, str):
        return np.nan

    desc = description.upper()

    if re.search(r"\bTYP\s*[12]\b", desc):
        return np.nan

    for pattern in [
        r"\b(\d{2})\s*:\s*\d{1,2}\b",
        r"\b(\d{2})\s*/\s*\d{1,2}\b",
        r"\bTHC\s*(\d{2})\b",
    ]:
        match = re.search(pattern, desc)
        if match:
            return float(match.group(1))

    return np.nan


def classify_thc_product_group(row):
    desc_norm = row["Item_Description_norm"]

    if desc_norm in FORCE_EXTRACT_PRODUCTS:
        return "Extracts or others"

    if row["itemCategoryCode"] != "CANNABISBLÜTEN":
        return "Extracts or others"

    thc = row["thc_final"]

    if np.isnan(thc):
        return "Not categorized flower"
    elif thc > 29:
        return "High thc flower"
    elif 22 <= thc <= 29:
        return "Middle thc flower"
    else:
        return "Low thc flower"


ledger = pd.read_excel(LEDGER_PATH)
workflow = pd.read_csv(WORKFLOW_PATH, sep=";", encoding="utf-8-sig")

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
    workflow["itemCategoryCode"].astype(str).str.strip()
)

df = ledger.merge(
    workflow[["number", "itemCategoryCode"]],
    left_on="Item_No",
    right_on="number",
    how="left"
)

df["Item_Description_norm"] = df["Item_Description"].apply(normalize_description)

df["thc_auto"] = df["Item_Description"].apply(extract_thc)
df["thc_manual"] = df["Item_Description_norm"].map(MANUAL_THC_MAP)

df["thc_final"] = df["thc_manual"].combine_first(df["thc_auto"])

df["thc_product_group"] = df.apply(classify_thc_product_group, axis=1)

uncategorized = (
    df[
        (df["itemCategoryCode"] == "CANNABISBLÜTEN") &
        (df["thc_product_group"] == "Not categorized flower")
    ][["Item_Description_norm"]]
    .drop_duplicates()
    .rename(columns={"Item_Description_norm": "Item_Description"})
    .sort_values("Item_Description")
)

uncategorized.to_excel(UNCAT_OUTFILE, index=False)

df.to_excel(MAIN_OUTFILE, index=False)

print("\n✅ Main file saved:")
print(MAIN_OUTFILE)

print("\n⚠ Flowers still not categorized:", len(uncategorized))
print("Saved to:")
print(UNCAT_OUTFILE)

