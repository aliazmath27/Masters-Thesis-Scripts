# Input: dl_forecast_summary.csv + demecan-analyst/gemini_forecast_summary.csv + demecan-analyst/ICDP_backtest_reasoning_forecast_full/ICDP_gemini_forecast_summary.csv | Output: best_window_analysis/best_window_summary.csv, best_window_full_ranking.csv

import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\aliaz\IU\Thesis")

DL_SUMMARY = BASE / "dl_forecast_summary.csv"
AGENT_SUMMARY = BASE / "demecan-analyst" / "gemini_forecast_summary.csv"
ICDP_AGENT_SUMMARY = (
    BASE / "demecan-analyst" / "ICDP_backtest_reasoning_forecast_full" / "ICDP_gemini_forecast_summary.csv"
)

OUTPUT_DIR = BASE / "best_window_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_WINDOW_PATH = OUTPUT_DIR / "best_window_summary.csv"
FULL_RANKING_PATH = OUTPUT_DIR / "best_window_full_ranking.csv"

RANK_METRIC = "MAPE"

TARGET_ORDER = ["Cost_Amount_Actual", "Margin", "Sales_Amount_Actual"]

REQUIRED_COLS = ["horizon_days", "lookback_days", "target", "model", "MAE", "RMSE", "MAPE"]

SOURCES = {
    "DL (MLP/CNN/LSTM)": DL_SUMMARY,
    "Agent (plain)": AGENT_SUMMARY,
    "Agent (IC-DP)": ICDP_AGENT_SUMMARY,
}


def load_combined() -> pd.DataFrame:
    frames = []
    for label, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} summary not found at expected path: {path}")
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{label} summary ({path.name}) is missing expected columns: {missing}")

        subset = df[REQUIRED_COLS].copy()
        if "icdp_example_included" in df.columns:
            subset["icdp_example_included"] = df["icdp_example_included"]
        else:
            subset["icdp_example_included"] = "N/A"
        frames.append(subset)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def analyze(combined: pd.DataFrame):
    full_rows = []
    best_rows = []

    for target in TARGET_ORDER:
        target_df = combined[combined["target"] == target]
        if target_df.empty:
            print(f"  (no rows found for target={target} — skipping)")
            continue

        for horizon_days in sorted(target_df["horizon_days"].unique()):
            horizon_df = target_df[target_df["horizon_days"] == horizon_days].copy()

            ranked = (
                horizon_df[["model", "lookback_days", "MAE", "RMSE", "MAPE", "icdp_example_included"]]
                .sort_values(RANK_METRIC)
                .reset_index(drop=True)
            )
            ranked.insert(0, "rank", ranked.index + 1)
            ranked.insert(0, "horizon_days", horizon_days)
            ranked.insert(0, "target", target)

            full_rows.append(ranked)
            best_rows.append(ranked.iloc[0])

    if not full_rows:
        raise ValueError("No matching rows found for any target — check that all three "
                          "summary CSVs contain data for Cost_Amount_Actual, Margin, "
                          "and Sales_Amount_Actual.")

    full_ranking = pd.concat(full_rows, ignore_index=True)
    best_window = pd.DataFrame(best_rows).reset_index(drop=True)
    best_window = best_window.drop(columns=["rank"])
    best_window = best_window.rename(columns={
        "lookback_days": "best_lookback_days",
        "model": "best_model",
    })

    return best_window, full_ranking


if __name__ == "__main__":
    print("Loading DL + agent (plain) + agent (IC-DP) summaries...")
    for label, path in SOURCES.items():
        print(f"  {label}: {path}")

    combined = load_combined()
    print(f"\n[OK] {len(combined)} total result rows loaded "
          f"({combined['model'].nunique()} distinct models: {sorted(combined['model'].unique())})")

    best_window, full_ranking = analyze(combined)

    best_window.to_csv(BEST_WINDOW_PATH, index=False)
    full_ranking.to_csv(FULL_RANKING_PATH, index=False)

    print(f"\nSaved: {BEST_WINDOW_PATH.name} ({len(best_window)} rows — one winner per target/horizon)")
    print(f"Saved: {FULL_RANKING_PATH.name} ({len(full_ranking)} rows — full ranking, all 5 models)")

    print(f"\n=== BEST MODEL + LOOKBACK PER TARGET / HORIZON (ranked by {RANK_METRIC}, 5 models) ===")
    print(best_window.to_string(index=False))

    fallback_wins = best_window[
        (best_window["best_model"] == "agent_reasoning_icdp")
        & (best_window["icdp_example_included"] == False)
    ]
    if not fallback_wins.empty:
        print(f"\n⚠ WARNING: {len(fallback_wins)} winning row(s) show best_model=agent_reasoning_icdp "
              f"but icdp_example_included=False. This means IC-DP fell back to a plain prompt for "
              f"that specific combo (not enough history for a worked example) -- so this win is "
              f"functionally identical to a plain agent_reasoning win, not evidence IC-DP helped:")
        print(fallback_wins[["target", "horizon_days", "best_lookback_days", "MAPE"]].to_string(index=False))
