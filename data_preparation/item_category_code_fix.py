# Input: ILE_Modified.xlsx | Output: ItemcategoryFixed/ILE_Modified_ItemCategoryFixed.xlsx

from pathlib import Path
import pandas as pd


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTPUT_FOLDER = BASE / "ItemcategoryFixed"
OUTPUT_FOLDER.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "ILE_Modified_ItemCategoryFixed.xlsx"


df = pd.read_excel(INPUT_FILE)


before_na = df["itemCategoryCode"].isna().sum()
before_empty = (df["itemCategoryCode"].astype(str).str.strip() == "").sum()

df["itemCategoryCode"] = df["itemCategoryCode"].fillna("Others")

df["itemCategoryCode"] = df["itemCategoryCode"].replace(r'^\s*$', "Others", regex=True)


df.to_excel(OUTPUT_FILE, index=False)

print("✔ itemCategoryCode blanks replaced with 'Others'")
print(f"✔ NaN values replaced: {before_na}")
print(f"✔ Empty strings replaced: {before_empty}")
print(f"✔ File saved to: {OUTPUT_FILE}")

