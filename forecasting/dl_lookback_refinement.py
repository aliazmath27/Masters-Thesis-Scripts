# Input: ILE_Modified.xlsx + DL_fine_tuned_2/winning_configs.csv | Output: DL_fine_tuned_2/dl_lookback_refinement_summary.csv, dl_lookback_refinement_daily.csv, lookback_refinement_report.csv

import gc
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

OUTPUT_ROOT = BASE / "DL_fine_tuned_2"
WINNING_CONFIG_PATH = OUTPUT_ROOT / "winning_configs.csv"
SUMMARY_PATH = OUTPUT_ROOT / "dl_forecast_summary_tuned2.csv"

REFINEMENT_SUMMARY_PATH = OUTPUT_ROOT / "dl_lookback_refinement_summary.csv"
REFINEMENT_DAILY_PATH = OUTPUT_ROOT / "dl_lookback_refinement_daily.csv"
REFINED_OPTIMUM_OUTDIR = OUTPUT_ROOT / "lookback_optimum_refined"
REFINED_OPTIMUM_OUTDIR.mkdir(exist_ok=True, parents=True)
REPORT_PATH = OUTPUT_ROOT / "lookback_refinement_report.csv"

FORECAST_OUTDIR = OUTPUT_ROOT / "dl_forecasting_tuned2_refined"

TARGETS = ["Sales_Amount_Actual", "Cost_Amount_Actual", "Margin"]
TARGET_COLORS = {"Cost_Amount_Actual": "#DC2626", "Margin": "#16A34A", "Sales_Amount_Actual": "#2563EB"}
MODELS = ["MLP", "CNN", "LSTM"]
CALENDAR_COL = "days_to_month_end"

ORIGINAL_HORIZON_LOOKBACKS = {
    2:  [7, 30, 90, 180, 365],
    7:  [15, 30, 90, 180, 365],
    30: [45, 90, 120, 180, 365],
    60: [90, 120, 180, 240, 365],
    90: [120, 180, 240, 300, 365],
}
MIN_SPACING_DAYS = 12


def build_extended_lookbacks(original_candidates):
    lo, hi = min(original_candidates), max(original_candidates)
    raw_new = [int(round(x)) for x in np.linspace(lo, hi, 8)[1:-1]]
    extended = sorted(original_candidates)
    for pt in raw_new:
        if all(abs(pt - existing) >= MIN_SPACING_DAYS for existing in extended):
            extended.append(pt)
    return sorted(extended)


HORIZON_LOOKBACKS = {h: build_extended_lookbacks(c) for h, c in ORIGINAL_HORIZON_LOOKBACKS.items()}

FINE_STEP_DAYS = 2
MIN_LOOKBACK_FLOOR = 5
SAVE_FORECAST_PLOTS = False

DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.0
DEFAULT_UNITS = 128

SUMMARY_COLUMNS = [
    "horizon_days", "lookback_days", "target", "model",
    "tuned_units", "tuned_learning_rate", "tuned_dropout",
    "test_start", "test_end",
    "MAE", "RMSE", "MAPE", "MASE",
    "retrained_after_nan",
]
DAILY_COLUMNS = ["horizon_days", "lookback_days", "target", "model", "date", "actual", "predicted"]
REPORT_COLUMNS = [
    "horizon_days", "target", "model",
    "coarse_best_lookback", "coarse_best_MASE",
    "overall_best_lookback", "overall_best_MASE",
    "lookback_moved_by_days", "mase_improvement",
    "n_fine_points_tested",
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


def load_daily_series() -> pd.DataFrame:
    print("Loading data...")
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
    daily[CALENDAR_COL] = (daily.index.days_in_month - daily.index.day).astype(float)

    print(f"[OK] Daily series: {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}")
    return daily


def create_sequences(data_2col: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data_2col) - window_size):
        X.append(data_2col[i:i + window_size])
        y.append(data_2col[i + window_size, 0])
    return np.array(X), np.array(y)


def make_scaled_2col(df: pd.DataFrame, target: str, fit_on: pd.DataFrame = None):
    cols = [target, CALENDAR_COL]
    scaler = MinMaxScaler()
    scaler.fit((fit_on if fit_on is not None else df)[cols].values)
    scaled = scaler.transform(df[cols].values)
    return scaler, scaled


def scale_calendar_values(calendar_raw: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    dummy = np.zeros((len(calendar_raw), 2))
    dummy[:, 1] = calendar_raw
    return scaler.transform(dummy)[:, 1]


def inverse_transform_target(preds_scaled_1d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    dummy = np.zeros((len(preds_scaled_1d), 2))
    dummy[:, 0] = preds_scaled_1d
    return scaler.inverse_transform(dummy)[:, 0]


def forecast(model, last_window, steps, window_size, mode, future_calendar_scaled):
    preds = []
    current = last_window.copy()
    for step in range(steps):
        inp = current.reshape(1, -1) if mode == "MLP" else current.reshape(1, window_size, 2)
        pred_target = float(model.predict(inp, verbose=0)[0, 0])
        preds.append(pred_target)
        next_row = np.array([pred_target, future_calendar_scaled[step]])
        current = np.vstack([current[1:], next_row])
    return np.array(preds)


def build_mlp(input_dim, units, dropout):
    layers = [Dense(units, activation='relu', input_shape=(input_dim,))]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [Dense(max(units // 2, 4), activation='relu'), Dense(1)]
    return Sequential(layers)


def build_cnn(lookback_days, units, dropout):
    layers = [Conv1D(units, 3, activation='relu', input_shape=(lookback_days, 2)), Flatten()]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [Dense(units * 2, activation='relu'), Dense(1)]
    return Sequential(layers)


def build_lstm(lookback_days, units, dropout):
    layers = [LSTM(units, input_shape=(lookback_days, 2))]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [Dense(max(units // 2, 4), activation='relu'), Dense(1)]
    return Sequential(layers)


EARLY_STOP = lambda: EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
MAX_EPOCHS = 60


def fit_model(model_name, lookback_days, units, learning_rate, dropout, X, y):
    early_stop = EARLY_STOP()
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    if model_name == "MLP":
        X_in = X.reshape((X.shape[0], -1))
        m = build_mlp(X_in.shape[1], units, dropout)
        m.compile(optimizer=optimizer, loss='mse')
        m.fit(X_in, y, epochs=MAX_EPOCHS, batch_size=16, verbose=0, validation_split=0.15, callbacks=[early_stop])
    elif model_name == "CNN":
        m = build_cnn(lookback_days, units, dropout)
        m.compile(optimizer=optimizer, loss='mse')
        m.fit(X, y, epochs=MAX_EPOCHS, batch_size=16, verbose=0, validation_split=0.15, callbacks=[early_stop])
    else:
        m = build_lstm(lookback_days, units, dropout)
        m.compile(optimizer=optimizer, loss='mse')
        m.fit(X, y, epochs=MAX_EPOCHS, batch_size=16, verbose=0, validation_split=0.15, callbacks=[early_stop])
    return m


def release_gpu_memory():
    K.clear_session()
    gc.collect()


def naive_insample_mae(train_target_values: np.ndarray) -> float:
    diffs = np.abs(np.diff(train_target_values))
    if len(diffs) == 0:
        return float('nan')
    return float(np.nanmean(diffs))


def mase_score(actual_values: np.ndarray, pred_values: np.ndarray, insample_naive_mae: float) -> float:
    if insample_naive_mae is None or np.isnan(insample_naive_mae) or insample_naive_mae == 0:
        return float('nan')
    mae = float(np.nanmean(np.abs(pred_values - actual_values)))
    return mae / insample_naive_mae


def mape_score(actual_values: np.ndarray, pred_values: np.ndarray) -> float:
    errors = pred_values - actual_values
    nonzero = np.where(actual_values != 0, actual_values, np.nan)
    return float(np.nanmean(np.abs(errors) / np.abs(nonzero)) * 100)


def train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler, scaled_train,
                             horizon_days, future_calendar_scaled):
    m = fit_model(model_name, lookback_days, cfg["units"], cfg["learning_rate"], cfg["dropout"], X, y)
    last_window = scaled_train[-lookback_days:]
    preds_scaled = forecast(m, last_window, horizon_days, lookback_days, model_name, future_calendar_scaled)
    preds = inverse_transform_target(preds_scaled, scaler)
    del m
    release_gpu_memory()
    return preds


def load_winning_configs() -> dict:
    if not WINNING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{WINNING_CONFIG_PATH} not found. Run dl_forecasting_tuned_2.py first -- "
            "this script reuses its cached hyperparameters and does not tune its own."
        )
    df = pd.read_csv(WINNING_CONFIG_PATH)
    out = {}
    for _, row in df.iterrows():
        key = (int(row["horizon_days"]), row["target"], row["model"])
        out[key] = {"units": int(row["units"]), "learning_rate": float(row["learning_rate"]), "dropout": float(row["dropout"])}
    return out


def fine_candidates_for_cell(coarse_summary: pd.DataFrame, horizon_days, target, model_name, series_len):
    grid = HORIZON_LOOKBACKS[horizon_days]
    sub = coarse_summary[
        (coarse_summary.horizon_days == horizon_days) &
        (coarse_summary.target == target) &
        (coarse_summary.model == model_name)
    ].copy()
    sub["MASE"] = pd.to_numeric(sub["MASE"], errors="coerce")
    sub = sub.dropna(subset=["MASE"])
    if sub.empty:
        print(f"    !! no valid coarse-grid MASE for h={horizon_days}d {target}/{model_name} -- skipping")
        return None, None, []

    best_row = sub.loc[sub["MASE"].idxmin()]
    best_lb = int(best_row["lookback_days"])
    best_mase = float(best_row["MASE"])

    idx = grid.index(best_lb)
    left = grid[idx - 1] if idx > 0 else None
    right = grid[idx + 1] if idx < len(grid) - 1 else None

    if left is not None and right is not None:
        lo, hi = left, right
    elif right is not None:
        lo, hi = max(MIN_LOOKBACK_FLOOR, best_lb - (right - best_lb)), right
    elif left is not None:
        lo, hi = left, best_lb + (best_lb - left)
    else:
        lo, hi = max(MIN_LOOKBACK_FLOOR, best_lb - 20), best_lb + 20

    hi = min(hi, series_len - horizon_days - 2)
    already_tested = set(grid)
    fine = [v for v in range(lo, hi + 1, FINE_STEP_DAYS) if v >= MIN_LOOKBACK_FLOOR and v not in already_tested]
    return best_lb, best_mase, fine


def load_completed_fine_combos() -> set:
    if not REFINEMENT_SUMMARY_PATH.exists():
        return set()
    existing = pd.read_csv(REFINEMENT_SUMMARY_PATH)
    if existing.empty:
        return set()
    return set(zip(existing.horizon_days, existing.lookback_days, existing.target, existing.model))


def append_rows_csv(path: Path, rows: list, columns: list):
    if not rows:
        return
    df_rows = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    df_rows.to_csv(path, mode="a", header=write_header, index=False)


def plot_forecast(horizon_days, lookback_days, model_name, target, train, test, preds_col):
    wdir = FORECAST_OUTDIR / f"h{horizon_days}d" / f"lb{lookback_days}d"
    wdir.mkdir(parents=True, exist_ok=True)
    plot_start = -(horizon_days * 3)
    plt.figure(figsize=(10, 5))
    plt.plot(train.index[plot_start:], train[target].iloc[plot_start:], label="Train")
    plt.plot(test.index, test[target], label="Test (Actual)", linewidth=2)
    plt.plot(test.index, preds_col, label=f"{model_name} Forecast (fine pass)")
    plt.axvspan(test.index[0], test.index[-1], alpha=0.2)
    plt.title(f"{target} | {model_name} (fine lookback pass)\nLookback: {lookback_days}d | Horizon: {horizon_days}d")
    plt.legend()
    plt.tight_layout()
    plt.savefig(wdir / f"{target}_{model_name}.png", dpi=150)
    plt.close()


def run_refinement(daily: pd.DataFrame, winning_config: dict):
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} not found -- run dl_forecasting_tuned_2.py first.")
    coarse_summary = pd.read_csv(SUMMARY_PATH)

    print("\n" + "=" * 70)
    print(f"LOOKBACK REFINEMENT -- every {FINE_STEP_DAYS} days around each cell's current best")
    print("(reuses cached hyperparameters -- no re-tuning)")
    print("=" * 70)

    plan = {}
    total_planned = 0
    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        for target in TARGETS:
            for model_name in MODELS:
                best_lb, best_mase, fine = fine_candidates_for_cell(coarse_summary, horizon_days, target, model_name, len(daily))
                plan[(horizon_days, target, model_name)] = (best_lb, best_mase, fine)
                total_planned += len(fine)

    completed = load_completed_fine_combos()
    already_done = sum(1 for (h, t, m), (_, _, fine) in plan.items() for lb in fine if (h, lb, t, m) in completed)
    print(f"\nPlanned fine-grained evaluations: {total_planned} total across 45 cells.")
    print(f"{already_done} already done in a previous run -- {total_planned - already_done} remaining.\n")

    split_cache, scaler_cache = {}, {}
    done_this_run = 0

    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        if horizon_days not in split_cache:
            split_cache[horizon_days] = (daily.iloc[:-horizon_days], daily.iloc[-horizon_days:])
        train, test = split_cache[horizon_days]

        for target in TARGETS:
            sc_key = (horizon_days, target)
            if sc_key not in scaler_cache:
                scaler, scaled_train = make_scaled_2col(train, target)
                future_calendar_scaled = scale_calendar_values(test[CALENDAR_COL].values, scaler)
                insample_mae = naive_insample_mae(train[target].values)
                scaler_cache[sc_key] = (scaler, scaled_train, future_calendar_scaled, insample_mae)
            scaler, scaled_train, future_calendar_scaled, insample_mae = scaler_cache[sc_key]
            actual_values = test[target].values

            for model_name in MODELS:
                best_lb, best_mase, fine_lookbacks = plan[(horizon_days, target, model_name)]
                if not fine_lookbacks:
                    continue
                cfg = winning_config.get(
                    (horizon_days, target, model_name),
                    {"units": DEFAULT_UNITS, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT},
                )

                for lookback_days in fine_lookbacks:
                    if (horizon_days, lookback_days, target, model_name) in completed:
                        continue

                    X, y = create_sequences(scaled_train, lookback_days)
                    if len(X) == 0:
                        done_this_run += 1
                        continue

                    print(f"  [{done_this_run + already_done + 1}/{total_planned}] "
                          f"h={horizon_days}d lb={lookback_days}d target={target} model={model_name} "
                          f"(current best for this cell: {best_lb}d, MASE={best_mase:.4f})")

                    pred_values = train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler,
                                                            scaled_train, horizon_days, future_calendar_scaled)
                    retrained = False
                    if np.isnan(pred_values).any():
                        print(f"    !! NaN forecast -- retraining once with a fresh init")
                        pred_values = train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler,
                                                                scaled_train, horizon_days, future_calendar_scaled)
                        retrained = True

                    errors = pred_values - actual_values
                    mae = float(np.nanmean(np.abs(errors)))
                    rmse = float(np.sqrt(np.nanmean(errors ** 2)))
                    mape = mape_score(actual_values, pred_values)
                    mase = mase_score(actual_values, pred_values, insample_mae)

                    append_rows_csv(REFINEMENT_SUMMARY_PATH, [{
                        "horizon_days": horizon_days, "lookback_days": lookback_days,
                        "target": target, "model": model_name,
                        "tuned_units": cfg["units"], "tuned_learning_rate": cfg["learning_rate"],
                        "tuned_dropout": cfg["dropout"],
                        "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
                        "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                        "MAPE": round(mape, 2) if not np.isnan(mape) else "",
                        "MASE": round(mase, 4) if not np.isnan(mase) else "",
                        "retrained_after_nan": retrained,
                    }], SUMMARY_COLUMNS)

                    append_rows_csv(REFINEMENT_DAILY_PATH, [
                        {"horizon_days": horizon_days, "lookback_days": lookback_days, "target": target,
                         "model": model_name, "date": str(d.date()), "actual": round(float(a), 2),
                         "predicted": round(float(p), 2) if not np.isnan(p) else ""}
                        for d, a, p in zip(test.index, actual_values, pred_values)
                    ], DAILY_COLUMNS)

                    if SAVE_FORECAST_PLOTS:
                        plot_forecast(horizon_days, lookback_days, model_name, target, train, test, pred_values)

                    done_this_run += 1

    print(f"\nRefinement pass: {done_this_run} new evaluation(s) this run.")
    if done_this_run + already_done < total_planned:
        print("Re-run this script to continue with the remaining fine-grained points.")
    else:
        print("All planned fine-grained points are done.")


def build_report_and_plots():
    coarse = pd.read_csv(SUMMARY_PATH)
    coarse["MASE"] = pd.to_numeric(coarse["MASE"], errors="coerce")
    coarse["pass"] = "coarse"

    if REFINEMENT_SUMMARY_PATH.exists():
        fine = pd.read_csv(REFINEMENT_SUMMARY_PATH)
        fine["MASE"] = pd.to_numeric(fine["MASE"], errors="coerce")
        fine["pass"] = "fine"
    else:
        fine = pd.DataFrame(columns=list(coarse.columns) + ["pass"])

    combined = pd.concat([coarse, fine], ignore_index=True)

    report_rows = []
    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        for target in TARGETS:
            for model_name in MODELS:
                c = coarse[(coarse.horizon_days == horizon_days) & (coarse.target == target) & (coarse.model == model_name)].dropna(subset=["MASE"])
                all_ = combined[(combined.horizon_days == horizon_days) & (combined.target == target) & (combined.model == model_name)].dropna(subset=["MASE"])
                if c.empty or all_.empty:
                    continue
                c_best = c.loc[c["MASE"].idxmin()]
                a_best = all_.loc[all_["MASE"].idxmin()]
                n_fine = len(all_[all_["pass"] == "fine"])
                report_rows.append({
                    "horizon_days": horizon_days, "target": target, "model": model_name,
                    "coarse_best_lookback": int(c_best.lookback_days), "coarse_best_MASE": round(float(c_best.MASE), 4),
                    "overall_best_lookback": int(a_best.lookback_days), "overall_best_MASE": round(float(a_best.MASE), 4),
                    "lookback_moved_by_days": int(a_best.lookback_days) - int(c_best.lookback_days),
                    "mase_improvement": round(float(c_best.MASE) - float(a_best.MASE), 4),
                    "n_fine_points_tested": n_fine,
                })

    report = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
    report.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 70)
    print("BEFORE / AFTER -- coarse-grid best vs. best-of-coarse+fine")
    print("=" * 70)
    moved = report[report.lookback_moved_by_days != 0]
    print(f"{len(moved)} of {len(report)} cells found a better lookback than the original coarse grid.")
    if len(moved):
        print(moved.sort_values("mase_improvement", ascending=False).to_string(index=False))
    print(f"\nFull report saved to {REPORT_PATH}")

    for horizon_days in sorted(HORIZON_LOOKBACKS.keys()):
        for model_name in MODELS:
            sub = combined[(combined.horizon_days == horizon_days) & (combined.model == model_name)]
            if sub.empty:
                continue
            wdir = REFINED_OPTIMUM_OUTDIR / f"h{horizon_days}d"
            wdir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(9, 5.5))
            for target in TARGETS:
                tsub = sub[(sub.target == target)].dropna(subset=["MASE"]).sort_values("lookback_days")
                if tsub.empty:
                    continue
                coarse_pts = tsub[tsub["pass"] == "coarse"]
                fine_pts = tsub[tsub["pass"] == "fine"]
                color = TARGET_COLORS[target]
                label = target.replace("_Amount_Actual", "")

                plt.plot(coarse_pts.lookback_days, coarse_pts.MASE, marker='o', markersize=6,
                         color=color, linewidth=1.6, label=f"{label} (coarse grid)")
                if not fine_pts.empty:
                    plt.scatter(fine_pts.lookback_days, fine_pts.MASE, marker='.', s=14,
                                color=color, alpha=0.55, label=f"{label} (fine pass)")

                best_idx = tsub["MASE"].idxmin()
                plt.scatter([tsub.loc[best_idx, "lookback_days"]], [tsub.loc[best_idx, "MASE"]],
                            color=color, s=170, zorder=5, edgecolors='black', linewidths=1.4)

            plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            plt.xlabel('Lookback (days)')
            plt.ylabel('MASE (1.0 = as good as naive)')
            plt.title(f'{model_name} — refined optimum lookback per target\n'
                      f'Horizon: {horizon_days}d (large circles = original coarse grid, dots = fine pass, '
                      f'ringed point = overall best)')
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(wdir / f"{model_name}_optimum_curve_refined.png", dpi=150)
            plt.close()

    print(f"Refined optimum-curve plots saved to {REFINED_OPTIMUM_OUTDIR}")


if __name__ == "__main__":
    daily_series = load_daily_series()
    winning_config = load_winning_configs()

    total_planned = sum(
        len(fine_candidates_for_cell(pd.read_csv(SUMMARY_PATH), h, t, m, len(daily_series))[2])
        for h in ORIGINAL_HORIZON_LOOKBACKS for t in TARGETS for m in MODELS
    )
    print(f"\nThis run will attempt up to {total_planned} new (horizon, lookback, target, model) fits "
          f"across all 45 cells, {FINE_STEP_DAYS} days apart, around each cell's current best lookback.")
    print("Safe to stop (Ctrl+C) and re-run at any time -- already-completed points are skipped.\n")

    run_refinement(daily_series, winning_config)
    build_report_and_plots()

    print(f"\n✅ REFINEMENT PASS DONE — results saved under {OUTPUT_ROOT}")
    print(f"  New summary/daily:     {REFINEMENT_SUMMARY_PATH.name} / {REFINEMENT_DAILY_PATH.name}")
    print(f"  Before/after report:   {REPORT_PATH.name}")
    print(f"  Refined optimum plots: {REFINED_OPTIMUM_OUTDIR}\\h<horizon>d\\<model>_optimum_curve_refined.png")

