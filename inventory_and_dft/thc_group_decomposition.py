# Input: ILE_Modified.xlsx | Output: thc_group_decomposition/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"
OUTDIR = BASE / "thc_group_decomposition"
OUTDIR.mkdir(exist_ok=True)

DATE_COL = "Posting_Date"

TARGETS = [
    "Sales_Amount_Actual",
    "Cost_Amount_Actual",
    "Margin"
]


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

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])


df["Sales_Amount_Actual"] = de_number_to_float(df["Sales_Amount_Actual"])
df["Cost_Amount_Actual"] = de_number_to_float(df["Cost_Amount_Actual"])

df["Margin"] = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]


groups = df["thc_product_group"].dropna().unique()

for group in groups:

    print(f"\nProcessing group: {group}")

    sub = df[df["thc_product_group"] == group]

    daily = (
        sub
        .groupby(pd.Grouper(key=DATE_COL, freq="D"))
        .agg({
            "Sales_Amount_Actual": "sum",
            "Cost_Amount_Actual": "sum",
            "Margin": "sum"
        })
        .sort_index()
        .fillna(0)
    )

    for target in TARGETS:

        series = daily[target]

        if len(series) < 30:
            print(f"Skipping {group} {target} (not enough data)")
            continue

        result = seasonal_decompose(
            series,
            model="additive",
            period=7
        )

        fig = result.plot()

        fig.set_size_inches(10, 8)

        plt.suptitle(
            f"{target} Decomposition\n"
            f"thc_product_group = {group}"
        )

        safe_name = str(group).replace(" ", "_")

        plt.savefig(
            OUTDIR / f"{safe_name}_{target}_decomposition.png",
            dpi=150
        )

        plt.close()

        print(f"✔ Saved {group} {target}")


print("\nDecomposition completed.")
