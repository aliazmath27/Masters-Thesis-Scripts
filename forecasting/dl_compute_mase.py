# Input: existing dl_forecast_summary*/gemini_forecast_summary*/ICDP_gemini_forecast_summary*.csv files (from the other forecasting scripts) | Output: mase_all_models.csv

import sys
import numpy as np
import pandas as pd
from pathlib import Path

SEARCH_ROOT = Path(r"C:\Users\aliaz\IU\Thesis")

INPUT_FILE_NAME = "ILE_Modified.xlsx"

RESULT_FILES = {
    "dl_forecast_summary.csv":         "DL (untuned baseline)",
    "dl_forecast_summary_tuned.csv":   "DL (tuned)",
    "dl_forecast_summary_tuned2.csv":  "DL (tuned, 2-run avg)",
    "gemini_forecast_summary.csv":     "Agent (plain DP)",
    "ICDP_gemini_forecast_summary.csv": "Agent (IC-DP)",
}

OUTPUT_PATH = SEARCH_ROOT / "mase_all_models.csv"

TARGETS = ["Cost_Amount_Actual", "Margin", "Sales_Amount_Actual"]
TARGET_LABELS = {"Cost_Amount_Actual": "Cost", "Margin": "Margin",
                 "Sales_Amount_Actual": "Sales"}
SEASONAL_PERIOD = 5


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


def load_daily_series(input_path: Path) -> pd.DataFrame:
    print(f"Loading daily series from {input_path.name} ...")
    df = pd.read_excel(input_path)
    df["Posting_Date"] = pd.to_datetime(df["Posting_Date"], errors="coerce")
    df["Sales_Amount_Actual"] = de_number_to_float(df["Sales_Amount_Actual"])
    df["Cost_Amount_Actual"] = de_number_to_float(df["Cost_Amount_Actual"])
    df["Margin"] = df["Sales_Amount_Actual"] + df["Cost_Amount_Actual"]
    df = df.dropna(subset=["Posting_Date"])

    df = df[
        (df["Entry_Type"].astype(str).str.strip().str.title() == "Sale") &
        (df["Source_Type"].astype(str).str.strip().str.title() == "Customer")
    ]

    daily = (
        df.groupby("Posting_Date")
        .agg({"Sales_Amount_Actual": "sum", "Cost_Amount_Actual": "sum", "Margin": "sum"})
        .sort_index()
        .fillna(0)
    )
    print(f"[OK] {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}")
    return daily


def find_file(root: Path, name: str):
    hits = list(root.rglob(name))
    if not hits:
        return None
    return sorted(hits, key=lambda p: len(p.parts))[0]


def build_denominators(daily: pd.DataFrame, horizons) -> dict:
    denoms = {}
    for h in horizons:
        train = daily.iloc[:-h]
        for t in TARGETS:
            v = train[t].values
            d1 = float(np.mean(np.abs(np.diff(v))))
            d5 = float(np.mean(np.abs(v[SEASONAL_PERIOD:] - v[:-SEASONAL_PERIOD])))
            denoms[(h, t)] = (d1, d5)
    return denoms


def main():
    if not SEARCH_ROOT.exists():
        print(f"ERROR: SEARCH_ROOT does not exist: {SEARCH_ROOT}", file=sys.stderr)
        sys.exit(1)

    input_path = find_file(SEARCH_ROOT, INPUT_FILE_NAME)
    if input_path is None:
        print(f"ERROR: could not find {INPUT_FILE_NAME} anywhere under {SEARCH_ROOT}", file=sys.stderr)
        sys.exit(1)

    daily = load_daily_series(input_path)

    frames = []
    print("\nLocating result files:")
    for fname, label in RESULT_FILES.items():
        path = find_file(SEARCH_ROOT, fname)
        if path is None:
            print(f"  [skip]  {fname}  -- not found (fine if you haven't run it)")
            continue
        d = pd.read_csv(path)
        if d.empty:
            print(f"  [skip]  {fname}  -- file is empty")
            continue
        missing = [c for c in ["horizon_days", "lookback_days", "target", "MAE"] if c not in d.columns]
        if missing:
            print(f"  [skip]  {fname}  -- missing required column(s): {missing}")
            continue
        if "model" not in d.columns:
            d["model"] = label
        d["source"] = label
        d["source_file"] = str(path)
        frames.append(d)
        print(f"  [ok]    {fname:<34} {len(d):>4} rows   ({path})")

    if not frames:
        print("\nERROR: no usable result files found. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    horizons = sorted(combined["horizon_days"].unique())
    denoms = build_denominators(daily, horizons)

    print(f"\nMASE denominators (in-sample naive MAE on the training split, per horizon x target):")
    print(f"  {'horizon':>8} {'target':<8} {'lag-1':>12} {'lag-5':>12}")
    for h in horizons:
        for t in TARGETS:
            d1, d5 = denoms[(h, t)]
            print(f"  {str(h)+'d':>8} {TARGET_LABELS[t]:<8} {d1:>12,.0f} {d5:>12,.0f}")

    def mase_row(r, which):
        key = (r["horizon_days"], r["target"])
        if key not in denoms:
            return np.nan
        denom = denoms[key][0 if which == "lag1" else 1]
        if denom == 0 or pd.isna(r["MAE"]):
            return np.nan
        return r["MAE"] / denom

    combined["MASE"] = combined.apply(lambda r: mase_row(r, "lag1"), axis=1).round(3)
    combined["MASE_seasonal5"] = combined.apply(lambda r: mase_row(r, "seas"), axis=1).round(3)
    combined["beats_naive"] = combined["MASE"] < 1.0

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Wrote {len(combined)} rows to {OUTPUT_PATH}")

    print("\n" + "=" * 104)
    print("BEST MODEL PER (HORIZON x TARGET), RANKED BY MASE   -- lower is better, <1.00 beats naive")
    print("=" * 104)
    print(f"{'H':>4} {'Target':<7} | {'Best model':<22} {'Source':<24} {'LB':>5} | {'MASE':>6} {'MASEs5':>7} {'MAPE':>7} | verdict")
    print("-" * 104)

    valid = combined.dropna(subset=["MASE"])
    n_beat = 0
    for h in horizons:
        for t in TARGETS:
            sub = valid[(valid.horizon_days == h) & (valid.target == t)]
            if sub.empty:
                continue
            r = sub.loc[sub["MASE"].idxmin()]
            verdict = "BEATS naive" if r["MASE"] < 1 else "worse than naive"
            if r["MASE"] < 1:
                n_beat += 1
            mape = r["MAPE"] if "MAPE" in r and not pd.isna(r["MAPE"]) else float("nan")
            print(f"{str(h)+'d':>4} {TARGET_LABELS[t]:<7} | {str(r['model']):<22} {str(r['source']):<24} "
                  f"{int(r['lookback_days']):>4}d | {r['MASE']:>6.2f} {r['MASE_seasonal5']:>7.2f} "
                  f"{mape:>6.1f}% | {verdict}")
    print("-" * 104)
    total = sum(1 for h in horizons for t in TARGETS
                if not valid[(valid.horizon_days == h) & (valid.target == t)].empty)
    print(f"{n_beat} of {total} (horizon x target) cases beat the naive forecast.")

    print("\n" + "=" * 104)
    print("PER-EXPERIMENT ROLLUP  (median MASE across all that experiment's rows; "
          "and its best-per-cell win count)")
    print("=" * 104)
    print(f"{'Source':<24} {'rows':>6} {'median MASE':>12} {'% rows beating naive':>22}")
    print("-" * 104)
    for src, g in valid.groupby("source"):
        pct = 100.0 * (g["MASE"] < 1).mean()
        print(f"{src:<24} {len(g):>6} {g['MASE'].median():>12.2f} {pct:>21.1f}%")

    print("\nNote: median MASE per experiment mixes all lookbacks together, including")
    print("deliberately poor ones, so it reflects the whole sweep rather than the")
    print("best achievable result. Use the per-cell table above for best-case claims.")


if __name__ == "__main__":
    main()
