# Input: ILE_Modified.xlsx | Output: DL_fine_tuned_2/ + dl_forecast_summary_tuned2.csv / dl_forecast_daily_tuned2.csv / winning_configs.csv

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
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

TUNING_OUTDIR = OUTPUT_ROOT / "hyperparameter_tuning"
TUNING_OUTDIR.mkdir(exist_ok=True, parents=True)
TUNING_SUMMARY_PATH = OUTPUT_ROOT / "hyperparameter_tuning_summary.csv"
WINNING_CONFIG_PATH = OUTPUT_ROOT / "winning_configs.csv"

OPTIMUM_OUTDIR = OUTPUT_ROOT / "lookback_optimum"
OPTIMUM_OUTDIR.mkdir(exist_ok=True, parents=True)

FORECAST_OUTDIR = OUTPUT_ROOT / "dl_forecasting_tuned2"
FORECAST_OUTDIR.mkdir(exist_ok=True, parents=True)
SUMMARY_PATH = OUTPUT_ROOT / "dl_forecast_summary_tuned2.csv"
DAILY_PATH = OUTPUT_ROOT / "dl_forecast_daily_tuned2.csv"

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
REPRESENTATIVE_LOOKBACK = {h: c[len(c) // 2] for h, c in ORIGINAL_HORIZON_LOOKBACKS.items()}

UNIT_CANDIDATES = [32, 64, 128, 256]
LR_CANDIDATES = [0.01, 0.001, 0.0001]
DROPOUT_CANDIDATES = [0.0, 0.2]
REPEATS = 2

DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.0
DEFAULT_UNITS = 128

TOTAL_LB_COMBOS = sum(len(v) for v in HORIZON_LOOKBACKS.values())
TOTAL_FINAL_COMBOS = TOTAL_LB_COMBOS * len(TARGETS)

SUMMARY_COLUMNS = [
    "horizon_days", "lookback_days", "target", "model",
    "tuned_units", "tuned_learning_rate", "tuned_dropout",
    "test_start", "test_end",
    "MAE", "RMSE", "MAPE", "MASE",
    "retrained_after_nan",
]
DAILY_COLUMNS = ["horizon_days", "lookback_days", "target", "model", "date", "actual", "predicted"]
TUNING_SUMMARY_COLUMNS = [
    "horizon_days", "target", "model", "stage", "parameter", "value",
    "validation_MASE", "is_winner",
]
WINNING_CONFIG_COLUMNS = ["horizon_days", "target", "model", "units", "learning_rate", "dropout", "val_MASE"]


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
    if fit_on is not None:
        scaler = MinMaxScaler()
        scaler.fit(fit_on[cols].values)
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[cols].values)
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
        if mode == "MLP":
            inp = current.reshape(1, -1)
        else:
            inp = current.reshape(1, window_size, 2)
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
        m.fit(X_in, y, epochs=MAX_EPOCHS, batch_size=16, verbose=0,
              validation_split=0.15, callbacks=[early_stop])
    elif model_name == "CNN":
        m = build_cnn(lookback_days, units, dropout)
        m.compile(optimizer=optimizer, loss='mse')
        m.fit(X, y, epochs=MAX_EPOCHS, batch_size=16, verbose=0,
              validation_split=0.15, callbacks=[early_stop])
    else:
        m = build_lstm(lookback_days, units, dropout)
        m.compile(optimizer=optimizer, loss='mse')
        m.fit(X, y, epochs=MAX_EPOCHS, batch_size=16, verbose=0,
              validation_split=0.15, callbacks=[early_stop])
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


def safe_min(scores, fallback_label=""):
    valid = [(v, s) for v, s in scores if not (isinstance(s, float) and np.isnan(s))]
    if not valid:
        mid_value = scores[len(scores) // 2][0]
        print(f"    !! WARNING: every candidate failed (all NaN) for {fallback_label} "
              f"-- falling back to {mid_value} as a safe default, not a validated winner")
        return mid_value, float('nan')
    return min(valid, key=lambda t: t[1])


def evaluate_config(model_name, lookback_days, units, learning_rate, dropout,
                     X, y, scaler, last_window, horizon_days,
                     tune_val_actual, future_calendar_scaled, insample_naive_mae,
                     repeats=REPEATS):
    run_scores = []
    for _ in range(repeats):
        m = fit_model(model_name, lookback_days, units, learning_rate, dropout, X, y)
        preds_scaled = forecast(m, last_window, horizon_days, lookback_days, model_name, future_calendar_scaled)
        preds = inverse_transform_target(preds_scaled, scaler)
        run_scores.append(mase_score(tune_val_actual, preds, insample_naive_mae))
        del m
        release_gpu_memory()
    valid = [s for s in run_scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else float('nan')


def load_winning_configs() -> dict:
    if not WINNING_CONFIG_PATH.exists():
        return {}
    df = pd.read_csv(WINNING_CONFIG_PATH)
    out = {}
    for _, row in df.iterrows():
        key = (int(row["horizon_days"]), row["target"], row["model"])
        out[key] = {
            "units": int(row["units"]),
            "learning_rate": float(row["learning_rate"]),
            "dropout": float(row["dropout"]),
        }
    return out


def append_winning_config(horizon_days, target, model_name, cfg, val_mase):
    row = pd.DataFrame([{
        "horizon_days": horizon_days, "target": target, "model": model_name,
        "units": cfg["units"], "learning_rate": cfg["learning_rate"], "dropout": cfg["dropout"],
        "val_MASE": round(val_mase, 4) if not np.isnan(val_mase) else "",
    }], columns=WINNING_CONFIG_COLUMNS)
    write_header = not WINNING_CONFIG_PATH.exists()
    row.to_csv(WINNING_CONFIG_PATH, mode="a", header=write_header, index=False)


def record_stage(tuning_rows, horizon_days, target, model_name, stage, parameter, scores, winner_value):
    for value, score in scores:
        tuning_rows.append({
            "horizon_days": horizon_days, "target": target, "model": model_name, "stage": stage,
            "parameter": parameter, "value": value,
            "validation_MASE": round(score, 4) if not np.isnan(score) else "",
            "is_winner": (value == winner_value),
        })


def append_tuning_rows(tuning_rows):
    if not tuning_rows:
        return
    df_rows = pd.DataFrame(tuning_rows, columns=TUNING_SUMMARY_COLUMNS)
    write_header = not TUNING_SUMMARY_PATH.exists()
    df_rows.to_csv(TUNING_SUMMARY_PATH, mode="a", header=write_header, index=False)


def plot_search_curve(horizon_days, target, model_name, parameter, scores, best_value, log_x=False, categorical=False):
    wdir = TUNING_OUTDIR / f"h{horizon_days}d"
    wdir.mkdir(parents=True, exist_ok=True)

    values = [s[0] for s in scores]
    mases = [s[1] for s in scores]

    plt.figure(figsize=(7, 4.5))
    if categorical:
        x_pos = range(len(values))
        colors = ['#DC2626' if v == best_value else '#6699CC' for v in values]
        plt.bar(x_pos, mases, color=colors)
        plt.xticks(x_pos, [str(v) for v in values])
    else:
        plt.plot(values, mases, marker='o', color='#6699CC', linewidth=2)
        best_idx = values.index(best_value)
        plt.scatter([best_value], [mases[best_idx]], color='#DC2626', s=110, zorder=5,
                    label=f'Winner: {best_value}')
        plt.legend()
        if log_x:
            plt.xscale('log')
        plt.xticks(values)

    plt.xlabel(parameter)
    plt.ylabel('Validation MASE')
    target_label = target.replace("_Amount_Actual", "")
    plt.title(f'{model_name} — {parameter} search — horizon {horizon_days}d — target: {target_label}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(wdir / f"{target}_{model_name}_{parameter}_search.png", dpi=150)
    plt.close()


def run_tuning_search(daily: pd.DataFrame):
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING -- sequential search per target: units -> learning_rate -> dropout")
    print("(scored by validation MASE, not averaged-MAPE-across-3-targets)")
    print("=" * 70)

    winning_config = load_winning_configs()
    already_done = len(winning_config)
    total_pairs = len(ORIGINAL_HORIZON_LOOKBACKS) * len(TARGETS) * len(MODELS)
    print(f"\n{already_done} of {total_pairs} (horizon, target, model) triples already tuned "
          f"-- resuming the rest.\n")

    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        lookback_days = REPRESENTATIVE_LOOKBACK[horizon_days]

        train_full = daily.iloc[:-horizon_days]
        tune_train = train_full.iloc[:-horizon_days]
        tune_val = train_full.iloc[-horizon_days:]

        enough_history = len(tune_train) >= lookback_days + horizon_days

        for target in TARGETS:
            for model_name in MODELS:
                key = (horizon_days, target, model_name)
                if key in winning_config:
                    continue

                print(f"\n  == {model_name} @ horizon={horizon_days}d, target={target} "
                      f"(representative lookback={lookback_days}d) ==")

                if not enough_history:
                    print(f"    Not enough history for a tuning validation window -- using safe defaults")
                    cfg = {"units": DEFAULT_UNITS, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT}
                    winning_config[key] = cfg
                    append_winning_config(horizon_days, target, model_name, cfg, float('nan'))
                    continue

                scaler, scaled_tune_train = make_scaled_2col(tune_train, target)
                X, y = create_sequences(scaled_tune_train, lookback_days)

                if len(X) == 0:
                    print(f"    No sequences available for tuning -- using safe defaults")
                    cfg = {"units": DEFAULT_UNITS, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT}
                    winning_config[key] = cfg
                    append_winning_config(horizon_days, target, model_name, cfg, float('nan'))
                    continue

                last_window = scaled_tune_train[-lookback_days:]
                tune_val_actual = tune_val[target].values
                future_calendar_scaled = scale_calendar_values(tune_val[CALENDAR_COL].values, scaler)
                insample_mae = naive_insample_mae(tune_train[target].values)

                tuning_rows = []

                stage_scores = []
                for units in UNIT_CANDIDATES:
                    score = evaluate_config(model_name, lookback_days, units, DEFAULT_LR, DEFAULT_DROPOUT,
                                             X, y, scaler, last_window, horizon_days,
                                             tune_val_actual, future_calendar_scaled, insample_mae)
                    stage_scores.append((units, score))
                    print(f"    [stage 1: units]   units={units:>4}  MASE={score:.4f}  (avg of {REPEATS} runs)")
                best_units, _ = safe_min(stage_scores, fallback_label=f"{model_name}@h{horizon_days}d/{target} stage1")
                record_stage(tuning_rows, horizon_days, target, model_name, 1, "units", stage_scores, best_units)
                plot_search_curve(horizon_days, target, model_name, "units", stage_scores, best_units)

                stage_scores = []
                for lr in LR_CANDIDATES:
                    score = evaluate_config(model_name, lookback_days, best_units, lr, DEFAULT_DROPOUT,
                                             X, y, scaler, last_window, horizon_days,
                                             tune_val_actual, future_calendar_scaled, insample_mae)
                    stage_scores.append((lr, score))
                    print(f"    [stage 2: lr]      lr={lr:<8}  MASE={score:.4f}  (avg of {REPEATS} runs)")
                best_lr, _ = safe_min(stage_scores, fallback_label=f"{model_name}@h{horizon_days}d/{target} stage2")
                record_stage(tuning_rows, horizon_days, target, model_name, 2, "learning_rate", stage_scores, best_lr)
                plot_search_curve(horizon_days, target, model_name, "learning_rate", stage_scores, best_lr, log_x=True)

                stage_scores = []
                for dropout in DROPOUT_CANDIDATES:
                    score = evaluate_config(model_name, lookback_days, best_units, best_lr, dropout,
                                             X, y, scaler, last_window, horizon_days,
                                             tune_val_actual, future_calendar_scaled, insample_mae)
                    stage_scores.append((dropout, score))
                    print(f"    [stage 3: dropout] dropout={dropout:<5}  MASE={score:.4f}  (avg of {REPEATS} runs)")
                best_dropout, best_final_score = safe_min(stage_scores, fallback_label=f"{model_name}@h{horizon_days}d/{target} stage3")
                record_stage(tuning_rows, horizon_days, target, model_name, 3, "dropout", stage_scores, best_dropout)
                plot_search_curve(horizon_days, target, model_name, "dropout", stage_scores, best_dropout, categorical=True)

                append_tuning_rows(tuning_rows)

                cfg = {"units": best_units, "learning_rate": best_lr, "dropout": best_dropout}
                winning_config[key] = cfg
                append_winning_config(horizon_days, target, model_name, cfg, best_final_score)

                print(f"    -> FINAL CONFIG: units={best_units}, lr={best_lr}, dropout={best_dropout} "
                      f"(validation MASE={best_final_score:.4f})")

    return winning_config


def append_rows_csv(path: Path, rows: list, columns: list):
    if not rows:
        return
    df_rows = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    df_rows.to_csv(path, mode="a", header=write_header, index=False)


def load_completed_final_combos() -> set:
    if not SUMMARY_PATH.exists():
        return set()
    existing = pd.read_csv(SUMMARY_PATH)
    if existing.empty:
        return set()
    counts = existing.groupby(["horizon_days", "lookback_days", "target"]).size()
    return set(counts[counts >= len(MODELS)].index)


def plot_forecast(horizon_days, lookback_days, model_name, target, train, test, preds_col):
    wdir = FORECAST_OUTDIR / f"h{horizon_days}d" / f"lb{lookback_days}d"
    wdir.mkdir(parents=True, exist_ok=True)

    plot_start = -(horizon_days * 3)
    plt.figure(figsize=(10, 5))
    plt.plot(train.index[plot_start:], train[target].iloc[plot_start:], label="Train")
    plt.plot(test.index, test[target], label="Test (Actual)", linewidth=2)
    plt.plot(test.index, preds_col, label=f"{model_name} Forecast")
    plt.axvspan(test.index[0], test.index[-1], alpha=0.2)
    plt.title(
        f"{target} | {model_name} (per-target tuned)\n"
        f"Lookback: {lookback_days}d | Horizon: {horizon_days}d\n"
        f"Test: {test.index[0].date()} \u2192 {test.index[-1].date()}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(wdir / f"{target}_{model_name}.png", dpi=150)
    plt.close()


def train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler, scaled_train,
                             horizon_days, future_calendar_scaled):
    m = fit_model(model_name, lookback_days, cfg["units"], cfg["learning_rate"], cfg["dropout"], X, y)
    last_window = scaled_train[-lookback_days:]
    preds_scaled = forecast(m, last_window, horizon_days, lookback_days, model_name, future_calendar_scaled)
    preds = inverse_transform_target(preds_scaled, scaler)
    del m
    release_gpu_memory()
    return preds


def run_final_training(daily: pd.DataFrame, winning_config: dict):
    print("\n" + "=" * 70)
    print("FINAL TRAINING -- fine-grained lookback sweep, per-target tuned configs")
    print("=" * 70)

    completed = load_completed_final_combos()
    all_combos = [(h, lb, t) for h, lbs in HORIZON_LOOKBACKS.items() for lb in lbs for t in TARGETS]
    remaining = [c for c in all_combos if c not in completed]

    print(f"\n{len(completed)} of {TOTAL_FINAL_COMBOS} (horizon, lookback, target) combos already completed.")
    print(f"{len(remaining)} combos remaining this run.\n")

    split_cache = {}
    scaler_cache = {}
    done_this_run = 0

    for horizon_days, lookback_days, target in remaining:
        if horizon_days not in split_cache:
            train = daily.iloc[:-horizon_days]
            test = daily.iloc[-horizon_days:]
            split_cache[horizon_days] = (train, test)
        train, test = split_cache[horizon_days]

        sc_key = (horizon_days, target)
        if sc_key not in scaler_cache:
            scaler, scaled_train = make_scaled_2col(train, target)
            future_calendar_scaled = scale_calendar_values(test[CALENDAR_COL].values, scaler)
            insample_mae = naive_insample_mae(train[target].values)
            scaler_cache[sc_key] = (scaler, scaled_train, future_calendar_scaled, insample_mae)
        scaler, scaled_train, future_calendar_scaled, insample_mae = scaler_cache[sc_key]

        print(f"  [{len(completed) + done_this_run + 1}/{TOTAL_FINAL_COMBOS}] "
              f"horizon={horizon_days}d lookback={lookback_days}d target={target} "
              f"test={test.index[0].date()}->{test.index[-1].date()}")

        X, y = create_sequences(scaled_train, lookback_days)
        if len(X) == 0:
            print(f"    Skipping — not enough training history for lookback={lookback_days}d")
            done_this_run += 1
            continue

        actual_values = test[target].values
        summary_rows = []
        daily_rows = []

        for model_name in MODELS:
            cfg = winning_config.get(
                (horizon_days, target, model_name),
                {"units": DEFAULT_UNITS, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT},
            )

            pred_values = train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler,
                                                    scaled_train, horizon_days, future_calendar_scaled)
            retrained = False
            if np.isnan(pred_values).any():
                print(f"    !! {model_name}/{target}: NaN forecast detected -- retraining once with a fresh init")
                pred_values = train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler,
                                                        scaled_train, horizon_days, future_calendar_scaled)
                retrained = True
                if np.isnan(pred_values).any():
                    print(f"    !! {model_name}/{target}: still NaN after retry -- recording as-is (check this row)")

            errors = pred_values - actual_values
            mae = float(np.nanmean(np.abs(errors)))
            rmse = float(np.sqrt(np.nanmean(errors ** 2)))
            mape = mape_score(actual_values, pred_values)
            mase = mase_score(actual_values, pred_values, insample_mae)

            summary_rows.append({
                "horizon_days": horizon_days, "lookback_days": lookback_days,
                "target": target, "model": model_name,
                "tuned_units": cfg["units"], "tuned_learning_rate": cfg["learning_rate"],
                "tuned_dropout": cfg["dropout"],
                "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
                "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                "MAPE": round(mape, 2) if not np.isnan(mape) else "",
                "MASE": round(mase, 4) if not np.isnan(mase) else "",
                "retrained_after_nan": retrained,
            })

            for date, actual_val, pred_val in zip(test.index, actual_values, pred_values):
                daily_rows.append({
                    "horizon_days": horizon_days, "lookback_days": lookback_days,
                    "target": target, "model": model_name,
                    "date": str(date.date()),
                    "actual": round(float(actual_val), 2),
                    "predicted": round(float(pred_val), 2) if not np.isnan(pred_val) else "",
                })

            plot_forecast(horizon_days, lookback_days, model_name, target, train, test, pred_values)

        append_rows_csv(SUMMARY_PATH, summary_rows, SUMMARY_COLUMNS)
        append_rows_csv(DAILY_PATH, daily_rows, DAILY_COLUMNS)
        print(f"    Done — {len(summary_rows)} rows written (3 models)")
        done_this_run += 1

    total_done = len(completed) + done_this_run
    print(f"\nFinal training complete. {done_this_run} combo(s) trained this run.")
    print(f"Total progress: {total_done}/{TOTAL_FINAL_COMBOS} combos.")
    if total_done < TOTAL_FINAL_COMBOS:
        print("Re-run this script to continue with the remaining combos.")


def plot_optimum_curves():
    if not SUMMARY_PATH.exists():
        print("No summary file yet -- run final training first.")
        return

    summary = pd.read_csv(SUMMARY_PATH)

    for horizon_days in sorted(HORIZON_LOOKBACKS.keys()):
        for model_name in MODELS:
            sub = summary[(summary.horizon_days == horizon_days) & (summary.model == model_name)]
            if sub.empty:
                continue

            wdir = OPTIMUM_OUTDIR / f"h{horizon_days}d"
            wdir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 5))
            for target in TARGETS:
                tsub = sub[sub.target == target].sort_values("lookback_days")
                tsub = tsub[pd.to_numeric(tsub["MASE"], errors="coerce").notna()]
                if tsub.empty:
                    continue
                lookbacks = tsub["lookback_days"].values
                mases = pd.to_numeric(tsub["MASE"]).values
                plt.plot(lookbacks, mases, marker='o', markersize=4,
                         color=TARGET_COLORS[target], linewidth=2, label=target.replace("_Amount_Actual", ""))
                best_idx = int(np.argmin(mases))
                plt.scatter([lookbacks[best_idx]], [mases[best_idx]],
                            color=TARGET_COLORS[target], s=140, zorder=5,
                            edgecolors='black', linewidths=1.2)

            plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            plt.xlabel('Lookback (days)')
            plt.ylabel('MASE (1.0 = as good as naive)')
            plt.title(f'{model_name} — optimum lookback per target (per-target tuned)\n'
                      f'Horizon: {horizon_days}d (marked points = lowest MASE per target)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(wdir / f"{model_name}_optimum_curve.png", dpi=150)
            plt.close()

    print(f"Saved optimum-point plots to {OPTIMUM_OUTDIR}")


if __name__ == "__main__":
    daily_series = load_daily_series()

    print("\nExtended lookback grid (original 5 candidates + fine-grained intermediate points):")
    for h, lbs in HORIZON_LOOKBACKS.items():
        print(f"  horizon={h}d: {len(lbs)} points -> {lbs}")

    winning_config = run_tuning_search(daily_series)

    print("\nWinning hyperparameter configs per (horizon, target, model):")
    for (h, t, m), cfg in sorted(winning_config.items()):
        print(f"  horizon={h}d  {t}  {m}: {cfg}")

    run_final_training(daily_series, winning_config)
    plot_optimum_curves()

    print(f"\n✅ ALL DONE — results saved under {OUTPUT_ROOT}")
    print(f"  Tuning search plots:  {TUNING_OUTDIR}\\h<horizon>d\\<target>_<model>_<parameter>_search.png")
    print(f"  Optimum-point plots:  {OPTIMUM_OUTDIR}\\h<horizon>d\\<model>_optimum_curve.png")
    print(f"  Final summary/daily:  {SUMMARY_PATH.name} / {DAILY_PATH.name}")
