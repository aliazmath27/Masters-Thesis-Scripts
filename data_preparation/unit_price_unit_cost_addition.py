# Input: ILE_Modified.xlsx | Output: Feature addition/ILE_unit_added.xlsx

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\aliaz\IU\Thesis"

INPUT_FILE = os.path.join(BASE, "ILE_Modified.xlsx")
OUTDIR = os.path.join(BASE, "Feature addition")
os.makedirs(OUTDIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTDIR, "ILE_unit_added.xlsx")

def de_number_to_float(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    mask = s.str.contains(",", na=False)
    s.loc[mask] = (
        s.loc[mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")

UNIT_QUANTITY_MAP = {
    "BEUTEL 100": 100,
    "BEUTEL 400": 400,
    "STK": 1,
    "MENGE IN G": 1,
    "FLA. 10ML": 10,
    "FLA. 30ML": 30,
    "SPRITZE1ML": 1,
    "DOSE 5G": 5,
    "DOSE 15G": 15,
    "DOSE 20G": 20,
    "DOSE 30G": 30,
    "DOSE 40G": 40,
    "DOSE 50G": 50,
    "GLAS 10": 10,
    "GLAS 50": 50,
}

df = pd.read_excel(INPUT_FILE)

df["Sales_Amount_Actual"] = de_number_to_float(df["Sales_Amount_Actual"])
df["Cost_Amount_Actual"] = de_number_to_float(df["Cost_Amount_Actual"])
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df["Unit_of_Measure_Code_norm"] = (
    df["Unit_of_Measure_Code"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["unit_quantity"] = pd.to_numeric(
    df["Unit_of_Measure_Code_norm"].map(UNIT_QUANTITY_MAP),
    errors="coerce"
)

df["total_physical_quantity"] = (
    df["Quantity"].abs() * df["unit_quantity"]
)

df["unit_price"] = (
    df["Sales_Amount_Actual"] / df["total_physical_quantity"]
).astype(float)

df.loc[df["total_physical_quantity"] <= 0, "unit_price"] = np.nan

df["unit_cost"] = (
    df["Cost_Amount_Actual"] / df["total_physical_quantity"]
).astype(float)

df.loc[df["total_physical_quantity"] <= 0, "unit_cost"] = np.nan

NUMERIC_COLS = [
    "Sales_Amount_Actual",
    "Cost_Amount_Actual",
    "Quantity",
    "unit_quantity",
    "total_physical_quantity",
    "unit_price",
    "unit_cost",
]

for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_excel(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f"
)

print("\n✅ File saved successfully with dot decimals:")
print(OUTPUT_FILE)

