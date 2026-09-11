# Input: ILE_Modified.xlsx | Output: dl_forecasting_tuned/ + dl_forecast_summary_tuned.csv / dl_forecast_daily_tuned.csv

import os
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
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


BASE = Path(r"C:\Users\aliaz\IU\Thesis")
INPUT_FILE = BASE / "ILE_Modified.xlsx"

TUNING_OUTDIR = BASE / "hyperparameter_tuning"
TUNING_OUTDIR.mkdir(exist_ok=True)
TUNING_SUMMARY_PATH = BASE / "hyperparameter_tuning_summary.csv"

OPTIMUM_OUTDIR = BASE / "lookback_optimum"
OPTIMUM_OUTDIR.mkdir(exist_ok=True)

OUTDIR = BASE / "dl_forecasting_tuned"
OUTDIR.mkdir(exist_ok=True)
SUMMARY_PATH = BASE / "dl_forecast_summary_tuned.csv"
DAILY_PATH = BASE / "dl_forecast_daily_tuned.csv"

TARGETS = ["Sales_Amount_Actual", "Cost_Amount_Actual", "Margin"]
TARGET_ORDER_FOR_PLOTS = ["Cost_Amount_Actual", "Margin", "Sales_Amount_Actual"]
TARGET_COLORS = {"Cost_Amount_Actual": "#DC2626", "Margin": "#16A34A", "Sales_Amount_Actual": "#2563EB"}
MODELS = ["MLP", "CNN", "LSTM"]

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

TOTAL_COMBOS = sum(len(v) for v in HORIZON_LOOKBACKS.values())

SUMMARY_COLUMNS = [
    "horizon_days", "lookback_days", "target", "model",
    "tuned_units", "tuned_learning_rate", "tuned_dropout",
    "test_start", "test_end",
    "MAE", "RMSE", "MAPE",
]
DAILY_COLUMNS = ["horizon_days", "lookback_days", "target", "model", "date", "actual", "predicted"]
TUNING_SUMMARY_COLUMNS = ["horizon_days", "model", "stage", "parameter", "value", "validation_MAPE", "is_winner"]


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
    print(f"[OK] Daily series: {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}")
    return daily


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def forecast(model, last_window, steps, window_size, mode):
    preds = []
    current = last_window.copy()
    for _ in range(steps):
        if mode == "MLP":
            inp = current.reshape(1, -1)
        else:
            inp = current.reshape(1, window_size, 3)
        pred = model.predict(inp, verbose=0)[0]
        preds.append(pred)
        current = np.vstack([current[1:], pred])
    return np.array(preds)


def build_mlp(input_dim, units, dropout):
    layers = [Dense(units, activation='relu', input_shape=(input_dim,))]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [Dense(max(units // 2, 4), activation='relu'), Dense(3)]
    return Sequential(layers)

def build_cnn(lookback_days, units, dropout):
    layers = [Conv1D(units, 3, activation='relu', input_shape=(lookback_days, 3)), Flatten()]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [Dense(units * 2, activation='relu'), Dense(3)]
    return Sequential(layers)

def build_lstm(lookback_days, units, dropout):
    layers = [LSTM(units, activation='relu', input_shape=(lookback_days, 3))]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [Dense(max(units // 2, 4), activation='relu'), Dense(3)]
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


def mape_score(actual_values, pred_values):
    errors = pred_values - actual_values
    nonzero = np.where(actual_values != 0, actual_values, np.nan)
    return float(np.nanmean(np.abs(errors) / np.abs(nonzero)) * 100)


def evaluate_config(model_name, lookback_days, units, learning_rate, dropout,
                     X, y, scaler, last_window, horizon_days, tune_val_actual_df,
                     repeats=REPEATS):
    run_scores = []
    for _ in range(repeats):
        m = fit_model(model_name, lookback_days, units, learning_rate, dropout, X, y)
        preds_scaled = forecast(m, last_window, horizon_days, lookback_days, model_name)
        preds = scaler.inverse_transform(preds_scaled)
        combo_mapes = [
            mape_score(tune_val_actual_df[target].values, preds[:, i])
            for i, target in enumerate(TARGETS)
        ]
        run_scores.append(float(np.nanmean(combo_mapes)))
    return float(np.nanmean(run_scores)) if not all(np.isnan(run_scores)) else float('nan')


def safe_min(scores, fallback_label=""):
    valid = [(v, s) for v, s in scores if not (isinstance(s, float) and np.isnan(s))]
    if not valid:
        mid_value = scores[len(scores) // 2][0]
        print(f"    !! WARNING: every candidate failed (all NaN) for {fallback_label} "
              f"-- falling back to {mid_value} as a safe default, not a validated winner")
        return mid_value, float('nan')
    return min(valid, key=lambda t: t[1])


def run_tuning_search(daily: pd.DataFrame):
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING -- sequential search: units -> learning_rate -> dropout")
    print("=" * 70)

    winning_config = {}
    tuning_rows = []

    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        lookback_days = REPRESENTATIVE_LOOKBACK[horizon_days]
        print(f"\n--- Horizon {horizon_days}d (search uses representative lookback {lookback_days}d) ---")

        train_full = daily.iloc[:-horizon_days]
        tune_train = train_full.iloc[:-horizon_days]
        tune_val_actual_df = train_full.iloc[-horizon_days:]

        if len(tune_train) < lookback_days + horizon_days:
            print(f"  Skipping horizon={horizon_days}d — not enough history for a tuning validation window")
            for model_name in MODELS:
                winning_config[(horizon_days, model_name)] = {
                    "units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT
                }
            continue

        scaler = MinMaxScaler()
        scaled_tune_train = scaler.fit_transform(tune_train)
        X, y = create_sequences(scaled_tune_train, lookback_days)

        if len(X) == 0:
            print(f"  Skipping horizon={horizon_days}d — no sequences available for tuning")
            for model_name in MODELS:
                winning_config[(horizon_days, model_name)] = {
                    "units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT
                }
            continue

        last_window = scaled_tune_train[-lookback_days:]

        for model_name in MODELS:
            print(f"\n  == {model_name} @ horizon={horizon_days}d ==")

            stage_scores = []
            for units in UNIT_CANDIDATES:
                mape = evaluate_config(model_name, lookback_days, units, DEFAULT_LR, DEFAULT_DROPOUT,
                                        X, y, scaler, last_window, horizon_days, tune_val_actual_df)
                stage_scores.append((units, mape))
                print(f"    [stage 1: units]   units={units:>4}  MAPE={mape:.2f}%  (avg of {REPEATS} runs)")
            best_units, _ = safe_min(stage_scores, fallback_label=f"{model_name}@h{horizon_days}d stage1-units")
            record_stage(tuning_rows, horizon_days, model_name, 1, "units", stage_scores, best_units)
            plot_search_curve(horizon_days, model_name, "units", stage_scores, best_units)

            stage_scores = []
            for lr in LR_CANDIDATES:
                mape = evaluate_config(model_name, lookback_days, best_units, lr, DEFAULT_DROPOUT,
                                        X, y, scaler, last_window, horizon_days, tune_val_actual_df)
                stage_scores.append((lr, mape))
                print(f"    [stage 2: lr]      lr={lr:<8}  MAPE={mape:.2f}%  (avg of {REPEATS} runs)")
            best_lr, _ = safe_min(stage_scores, fallback_label=f"{model_name}@h{horizon_days}d stage2-lr")
            record_stage(tuning_rows, horizon_days, model_name, 2, "learning_rate", stage_scores, best_lr)
            plot_search_curve(horizon_days, model_name, "learning_rate", stage_scores, best_lr, log_x=True)

            stage_scores = []
            for dropout in DROPOUT_CANDIDATES:
                mape = evaluate_config(model_name, lookback_days, best_units, best_lr, dropout,
                                        X, y, scaler, last_window, horizon_days, tune_val_actual_df)
                stage_scores.append((dropout, mape))
                print(f"    [stage 3: dropout] dropout={dropout:<5}  MAPE={mape:.2f}%  (avg of {REPEATS} runs)")
            best_dropout, best_mape_final = safe_min(stage_scores, fallback_label=f"{model_name}@h{horizon_days}d stage3-dropout")
            record_stage(tuning_rows, horizon_days, model_name, 3, "dropout", stage_scores, best_dropout)
            plot_search_curve(horizon_days, model_name, "dropout", stage_scores, best_dropout, categorical=True)

            winning_config[(horizon_days, model_name)] = {
                "units": best_units, "learning_rate": best_lr, "dropout": best_dropout
            }
            print(f"    -> FINAL CONFIG for {model_name} @ horizon={horizon_days}d: "
                  f"units={best_units}, lr={best_lr}, dropout={best_dropout} "
                  f"(final validation MAPE={best_mape_final:.2f}%)")

    tuning_df = pd.DataFrame(tuning_rows, columns=TUNING_SUMMARY_COLUMNS)
    tuning_df.to_csv(TUNING_SUMMARY_PATH, index=False)
    print(f"\nSaved: {TUNING_SUMMARY_PATH.name}")

    return winning_config


def record_stage(tuning_rows, horizon_days, model_name, stage, parameter, scores, winner_value):
    for value, mape in scores:
        tuning_rows.append({
            "horizon_days": horizon_days, "model": model_name, "stage": stage,
            "parameter": parameter, "value": value, "validation_MAPE": round(mape, 2),
            "is_winner": (value == winner_value),
        })


def plot_search_curve(horizon_days, model_name, parameter, scores, best_value, log_x=False, categorical=False):
    wdir = TUNING_OUTDIR / f"h{horizon_days}d"
    wdir.mkdir(parents=True, exist_ok=True)

    values = [s[0] for s in scores]
    mapes = [s[1] for s in scores]

    plt.figure(figsize=(7, 4.5))
    if categorical:
        x_pos = range(len(values))
        colors = ['#DC2626' if v == best_value else '#6699CC' for v in values]
        plt.bar(x_pos, mapes, color=colors)
        plt.xticks(x_pos, [str(v) for v in values])
    else:
        plt.plot(values, mapes, marker='o', color='#6699CC', linewidth=2)
        best_idx = values.index(best_value)
        plt.scatter([best_value], [mapes[best_idx]], color='#DC2626', s=110, zorder=5,
                    label=f'Winner: {best_value}')
        plt.legend()
        if log_x:
            plt.xscale('log')
        plt.xticks(values)

    plt.xlabel(parameter)
    plt.ylabel('Validation MAPE (%) -- avg across 3 targets')
    plt.title(f'{model_name} — {parameter} search — horizon {horizon_days}d')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(wdir / f"{model_name}_{parameter}_search.png", dpi=150)
    plt.close()


def append_rows_csv(path: Path, rows: list, columns: list):
    if not rows:
        return
    df_rows = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    df_rows.to_csv(path, mode="a", header=write_header, index=False)


def load_completed_combos() -> set:
    if not SUMMARY_PATH.exists():
        return set()
    existing = pd.read_csv(SUMMARY_PATH)
    if existing.empty:
        return set()
    counts = existing.groupby(["horizon_days", "lookback_days"]).size()
    return set(counts[counts >= len(MODELS) * len(TARGETS)].index)


def plot_forecast(horizon_days, lookback_days, model_name, target, train, test, preds_col):
    wdir = OUTDIR / f"h{horizon_days}d" / f"lb{lookback_days}d"
    wdir.mkdir(parents=True, exist_ok=True)

    plot_start = -(horizon_days * 3)
    plt.figure(figsize=(10, 5))
    plt.plot(train.index[plot_start:], train[target].iloc[plot_start:], label="Train")
    plt.plot(test.index, test[target], label="Test (Actual)", linewidth=2)
    plt.plot(test.index, preds_col, label=f"{model_name} Forecast")
    plt.axvspan(test.index[0], test.index[-1], alpha=0.2)
    plt.title(
        f"{target} | {model_name} (tuned)\n"
        f"Lookback: {lookback_days}d | Horizon: {horizon_days}d\n"
        f"Test: {test.index[0].date()} \u2192 {test.index[-1].date()}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(wdir / f"{target}_{model_name}.png", dpi=150)
    plt.close()


def run_final_training(daily: pd.DataFrame, winning_config: dict):
    print("\n" + "=" * 70)
    print("FINAL TRAINING -- fine-grained lookback sweep, tuned (units/lr/dropout) per horizon")
    print("=" * 70)

    completed = load_completed_combos()
    all_combos = [(h, lb) for h, lbs in HORIZON_LOOKBACKS.items() for lb in lbs]
    remaining = [c for c in all_combos if c not in completed]

    print(f"\n{len(completed)} of {TOTAL_COMBOS} combos already completed.")
    print(f"{len(remaining)} combos remaining this run.\n")

    split_cache = {}
    done_this_run = 0

    for horizon_days, lookback_days in remaining:
        if horizon_days not in split_cache:
            train = daily.iloc[:-horizon_days]
            test = daily.iloc[-horizon_days:]
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train)
            split_cache[horizon_days] = (train, test, scaler, scaled_train)
        train, test, scaler, scaled_train = split_cache[horizon_days]

        print(f"  [{len(completed) + done_this_run + 1}/{TOTAL_COMBOS}] "
              f"horizon={horizon_days}d lookback={lookback_days}d "
              f"test={test.index[0].date()}->{test.index[-1].date()}")

        X, y = create_sequences(scaled_train, lookback_days)
        if len(X) == 0:
            print(f"    Skipping — not enough training history for lookback={lookback_days}d")
            done_this_run += 1
            continue

        preds_dict = {}
        config_used = {}
        for model_name in MODELS:
            cfg = winning_config.get((horizon_days, model_name),
                                      {"units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT})
            config_used[model_name] = cfg
            m = fit_model(model_name, lookback_days, cfg["units"], cfg["learning_rate"], cfg["dropout"], X, y)
            last_window = scaled_train[-lookback_days:]
            preds_scaled = forecast(m, last_window, horizon_days, lookback_days, model_name)
            preds_dict[model_name] = scaler.inverse_transform(preds_scaled)

        summary_rows = []
        daily_rows = []

        for model_name, preds in preds_dict.items():
            cfg = config_used[model_name]
            for i, target in enumerate(TARGETS):
                actual_values = test[target].values
                pred_values = preds[:, i]

                errors = pred_values - actual_values
                abs_errors = np.abs(errors)
                mae = float(np.nanmean(abs_errors))
                rmse = float(np.sqrt(np.nanmean(errors ** 2)))
                mape = mape_score(actual_values, pred_values)

                summary_rows.append({
                    "horizon_days": horizon_days, "lookback_days": lookback_days,
                    "target": target, "model": model_name,
                    "tuned_units": cfg["units"], "tuned_learning_rate": cfg["learning_rate"],
                    "tuned_dropout": cfg["dropout"],
                    "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
                    "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2),
                })

                for date, actual_val, pred_val in zip(test.index, actual_values, pred_values):
                    daily_rows.append({
                        "horizon_days": horizon_days, "lookback_days": lookback_days,
                        "target": target, "model": model_name,
                        "date": str(date.date()), "actual": round(float(actual_val), 2),
                        "predicted": round(float(pred_val), 2),
                    })

                plot_forecast(horizon_days, lookback_days, model_name, target, train, test, pred_values)

        append_rows_csv(SUMMARY_PATH, summary_rows, SUMMARY_COLUMNS)
        append_rows_csv(DAILY_PATH, daily_rows, DAILY_COLUMNS)
        print(f"    Done — configs used: {config_used}")
        done_this_run += 1

    total_done = len(completed) + done_this_run
    print(f"\nFinal training complete. {done_this_run} combo(s) trained this run.")
    print(f"Total progress: {total_done}/{TOTAL_COMBOS} combos.")


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
            for target in TARGET_ORDER_FOR_PLOTS:
                tsub = sub[sub.target == target].sort_values("lookback_days")
                if tsub.empty:
                    continue
                lookbacks = tsub["lookback_days"].values
                mapes = tsub["MAPE"].values
                plt.plot(lookbacks, mapes, marker='o', markersize=4,
                         color=TARGET_COLORS[target], linewidth=2, label=target.replace("_Amount_Actual", ""))
                best_idx = np.argmin(mapes)
                plt.scatter([lookbacks[best_idx]], [mapes[best_idx]],
                            color=TARGET_COLORS[target], s=140, zorder=5,
                            edgecolors='black', linewidths=1.2)

            plt.xlabel('Lookback (days)')
            plt.ylabel('MAPE (%)')
            plt.title(f'{model_name} — optimum lookback per target\nHorizon: {horizon_days}d '
                      f'(marked points = lowest MAPE per target)')
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

    print("\nWinning hyperparameter configs per (horizon, model):")
    for (h, m), cfg in sorted(winning_config.items()):
        print(f"  horizon={h}d  {m}: {cfg}")

    run_final_training(daily_series, winning_config)
    plot_optimum_curves()

    print("\n✅ ALL DONE — tuned + fine-grained DL results saved to "
          "dl_forecast_summary_tuned.csv / dl_forecast_daily_tuned.csv")
    print("Tuning search plots saved to hyperparameter_tuning/h<horizon>d/<model>_<parameter>_search.png")
    print("Optimum-point curves saved to lookback_optimum/h<horizon>d/<model>_optimum_curve.png")
