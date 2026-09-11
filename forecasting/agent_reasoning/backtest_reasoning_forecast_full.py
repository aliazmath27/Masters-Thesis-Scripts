# Input: ILE_Modified.xlsx (same folder as this script) + GOOGLE_API_KEY in a local .env | Output: gemini_forecast_summary.csv, gemini_forecast_daily.csv

import os
import sys
import json
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(Path(__file__).resolve().parent / ".env")

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found. Check that .env exists in this folder "
          "and contains a line like GOOGLE_API_KEY=your-key", file=sys.stderr)
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = Path(BASE_DIR) / "ILE_Modified.xlsx"
OUTDIR = Path(BASE_DIR) / "agent_forecasting_experiments"
OUTDIR.mkdir(exist_ok=True)

SUMMARY_PATH = Path(BASE_DIR) / "gemini_forecast_summary.csv"
DAILY_PATH = Path(BASE_DIR) / "gemini_forecast_daily.csv"

TARGETS = ["Sales_Amount_Actual", "Cost_Amount_Actual", "Margin"]

HORIZON_LOOKBACKS = {
    2:  [7, 30, 90, 180, 365],
    7:  [15, 30, 90, 180, 365],
    30: [45, 90, 120, 180, 365],
    60: [90, 120, 180, 240, 365],
    90: [120, 180, 240, 300, 365],
}

TOTAL_COMBOS = sum(len(v) for v in HORIZON_LOOKBACKS.values()) * len(TARGETS)

SUMMARY_COLUMNS = [
    "horizon_days", "lookback_days", "target",
    "test_start", "test_end",
    "MAE", "RMSE", "MAPE",
    "trend_note", "seasonality_note", "model",
]
DAILY_COLUMNS = ["horizon_days", "lookback_days", "target", "date", "actual", "predicted"]

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

def load_daily_series() -> pd.DataFrame:
    print("Loading data...", file=sys.stderr)
    df = pd.read_excel(INPUT_FILE)
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
    )
    daily = daily.fillna(0)
    print(f"[OK] Daily series: {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}",
          file=sys.stderr)
    return daily

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

SYSTEM_PROMPT = """You are a forecasting assistant. You will be given a daily time series
for one financial metric, oldest to newest. Reason step by step:
1. State the trend (direction, roughly how much change per day)
2. Note any repeating/seasonal pattern (e.g. weekday effects, spikes)
3. Produce a day-by-day forecast for the next N days requested

Respond with ONLY a JSON object, no other text, no markdown fences:
{"trend": "...", "seasonality_note": "...", "forecast": [<n1>, <n2>, ...], "reasoning": "..."}
The "forecast" list must have EXACTLY the number of values requested, in order.
"""

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
                elif "text" in item:
                    parts.append(item["text"])
        return "\n".join(parts)
    return str(content)

def parse_json_response(text: str) -> dict:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    return json.loads(cleaned, strict=False)

class RateLimitExhausted(Exception):
    pass

def reasoning_forecast(history_series: pd.Series, target: str, horizon_days: int,
                        max_retries: int = 5) -> dict:
    history_json = [{"date": str(d.date()), target: round(float(v), 2)}
                     for d, v in history_series.items()]
    user_msg = (
        f"Metric: {target}\n"
        f"Historical daily values (oldest to newest):\n{json.dumps(history_json)}\n\n"
        f"Forecast the next {horizon_days} days for {target}. "
        f'Return exactly {horizon_days} values in the "forecast" list.'
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]

    rate_limited_last = False
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return parse_json_response(extract_text(response.content))
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                rate_limited_last = True
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                wait = float(match.group(1)) + 2 if match else 30
                print(f"    Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            if "UNAVAILABLE" in msg or "503" in msg:
                rate_limited_last = False
                wait = 15 * (attempt + 1)
                print(f"    Server overloaded (503), waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    if rate_limited_last:
        raise RateLimitExhausted(
            f"Exhausted {max_retries} retries due to rate limiting -- daily quota is likely used up."
        )
    raise RuntimeError(f"Failed after {max_retries} retries due to server overload")

def plot_forecast(horizon_days: int, lookback_days: int, target: str,
                   train: pd.DataFrame, test: pd.DataFrame,
                   predicted_values: np.ndarray):
    wdir = OUTDIR / f"h{horizon_days}d" / f"lb{lookback_days}d"
    wdir.mkdir(parents=True, exist_ok=True)

    plot_start = -(horizon_days * 3)
    plt.figure(figsize=(10, 5))

    plt.plot(train.index[plot_start:], train[target].iloc[plot_start:],
              label="Train", color="#D9D9D9", linewidth=1.5)
    plt.plot(test.index, test[target], label="Test (Actual)",
              color="#666666", linestyle="--", linewidth=2)
    plt.plot(test.index, predicted_values, label="Agent Reasoning Forecast",
              color="#6699CC", linewidth=2)

    plt.axvspan(test.index[0], test.index[-1], alpha=0.15, color="#6699CC")
    plt.title(
        f"{target} | Agent Reasoning\n"
        f"Lookback: {lookback_days}d | Horizon: {horizon_days}d\n"
        f"Test: {test.index[0].date()} \u2192 {test.index[-1].date()}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(wdir / f"{target}_agent.png", dpi=150)
    plt.close()

def build_task_list():
    tasks = []
    for horizon_days, lookbacks in HORIZON_LOOKBACKS.items():
        for lookback_days in lookbacks:
            for target in TARGETS:
                tasks.append((horizon_days, lookback_days, target))
    return tasks

def load_completed_combos() -> set:
    if not SUMMARY_PATH.exists():
        return set()
    existing = pd.read_csv(SUMMARY_PATH)
    if existing.empty:
        return set()
    return set(
        zip(existing["horizon_days"], existing["lookback_days"], existing["target"])
    )

def append_row_csv(path: Path, row: dict, columns: list):
    df_row = pd.DataFrame([row], columns=columns)
    write_header = not path.exists()
    df_row.to_csv(path, mode="a", header=write_header, index=False)

def append_rows_csv(path: Path, rows: list, columns: list):
    if not rows:
        return
    df_rows = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    df_rows.to_csv(path, mode="a", header=write_header, index=False)

def run_backtest(daily: pd.DataFrame):
    all_tasks = build_task_list()
    completed = load_completed_combos()
    remaining = [t for t in all_tasks if t not in completed]

    print(f"\n{len(completed)} of {TOTAL_COMBOS} combos already completed "
          f"(found in {SUMMARY_PATH.name}).")
    print(f"{len(remaining)} combos remaining this run.\n")

    if not remaining:
        print("All combos already completed. Nothing to do.")
        return

    done_this_run = 0
    split_cache = {}

    for horizon_days, lookback_days, target in remaining:
        if horizon_days not in split_cache:
            train_full = daily.iloc[:-horizon_days]
            test_full = daily.iloc[-horizon_days:]
            split_cache[horizon_days] = (train_full, test_full)
        train_full, test = split_cache[horizon_days]

        if len(train_full) < lookback_days:
            print(f"  Skipping horizon={horizon_days}d lookback={lookback_days}d {target} "
                  f"— not enough training history ({len(train_full)}d available)")
            done_this_run += 1
            continue

        history = train_full[target].tail(lookback_days)
        actual_values = test[target].values

        print(f"  [{done_this_run + len(completed) + 1}/{TOTAL_COMBOS}] "
              f"horizon={horizon_days}d lookback={lookback_days}d target={target} "
              f"test={test.index[0].date()}->{test.index[-1].date()}")

        try:
            result = reasoning_forecast(history, target, horizon_days)
            predicted_values = np.array(result.get("forecast", []), dtype=float)
            if len(predicted_values) != horizon_days:
                print(f"    WARNING: got {len(predicted_values)} values, expected {horizon_days} — padding/truncating")
                predicted_values = np.resize(predicted_values, horizon_days)
        except RateLimitExhausted as e:
            print(f"\n  STOPPING RUN: {e}")
            print(f"  Progress saved: {done_this_run} combo(s) completed this run, "
                  f"{len(completed) + done_this_run}/{TOTAL_COMBOS} total.")
            print("  Re-run this script (e.g. tomorrow, once quota resets) to continue "
                  "from exactly this point.")
            return
        except Exception as e:
            print(f"    Error: {e}")
            predicted_values = np.full(horizon_days, np.nan)
            result = {}

        errors = predicted_values - actual_values
        abs_errors = np.abs(errors)
        mae = float(np.nanmean(abs_errors))
        rmse = float(np.sqrt(np.nanmean(errors ** 2)))
        nonzero_actual = np.where(actual_values != 0, actual_values, np.nan)
        mape = float(np.nanmean(np.abs(errors) / np.abs(nonzero_actual)) * 100)

        summary_row = {
            "horizon_days": horizon_days,
            "lookback_days": lookback_days,
            "target": target,
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
            "trend_note": result.get("trend", ""),
            "seasonality_note": result.get("seasonality_note", ""),
            "model": "agent_reasoning",
        }
        append_row_csv(SUMMARY_PATH, summary_row, SUMMARY_COLUMNS)
        print(f"    MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}%")

        daily_rows = []
        for date, actual_val, pred_val in zip(test.index, actual_values, predicted_values):
            daily_rows.append({
                "horizon_days": horizon_days,
                "lookback_days": lookback_days,
                "target": target,
                "date": str(date.date()),
                "actual": round(float(actual_val), 2),
                "predicted": round(float(pred_val), 2) if not np.isnan(pred_val) else None,
            })
        append_rows_csv(DAILY_PATH, daily_rows, DAILY_COLUMNS)

        if not np.all(np.isnan(predicted_values)):
            plot_forecast(horizon_days, lookback_days, target, train_full, test, predicted_values)

        done_this_run += 1

    total_done = len(completed) + done_this_run
    print(f"\nRun complete. {done_this_run} combo(s) completed this run.")
    print(f"Total progress: {total_done}/{TOTAL_COMBOS} combos.")
    if total_done < TOTAL_COMBOS:
        print("Re-run this script to continue with the remaining combos.")

if __name__ == "__main__":
    print("Running FULL agent reasoning forecast backtest "
          f"({TOTAL_COMBOS} horizon x lookback x target combos, resumable)...")
    daily_series = load_daily_series()
    run_backtest(daily_series)

    if SUMMARY_PATH.exists():
        final_summary = pd.read_csv(SUMMARY_PATH)
        print(f"\nCurrent contents of {SUMMARY_PATH.name}: {len(final_summary)} rows")
        print(final_summary.to_string(index=False))
