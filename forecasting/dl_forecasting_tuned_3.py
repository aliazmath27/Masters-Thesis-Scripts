# Input: ILE_Modified.xlsx | Output: DL_fine_tuned_3/ + dl_forecast_summary_tuned3.csv / dl_forecast_daily_tuned3.csv / winning_configs_v3.csv

import gc
import itertools
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

V2_ROOT = BASE / "DL_fine_tuned_2"
V2_TUNING_SUMMARY_PATH = V2_ROOT / "hyperparameter_tuning_summary.csv"
V2_WINNING_CONFIG_PATH = V2_ROOT / "winning_configs.csv"
V2_SUMMARY_PATH = V2_ROOT / "dl_forecast_summary_tuned2.csv"

OUTPUT_ROOT = BASE / "DL_fine_tuned_3"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
GRID_LOG_PATH = OUTPUT_ROOT / "hyperparameter_grid_v3.csv"
WINNING_CONFIG_PATH = OUTPUT_ROOT / "winning_configs_v3.csv"
HP_COMPARISON_PATH = OUTPUT_ROOT / "hyperparameter_method_comparison.csv"
SUMMARY_PATH = OUTPUT_ROOT / "dl_forecast_summary_tuned3.csv"
DAILY_PATH = OUTPUT_ROOT / "dl_forecast_daily_tuned3.csv"
LOOKBACK_COMPARISON_PATH = OUTPUT_ROOT / "lookback_v3_vs_v2_comparison.csv"
OPTIMUM_OUTDIR = OUTPUT_ROOT / "lookback_optimum_v3"
OPTIMUM_OUTDIR.mkdir(exist_ok=True, parents=True)
FORECAST_OUTDIR = OUTPUT_ROOT / "dl_forecasting_tuned3"

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


V2_HORIZON_LOOKBACKS = {h: build_extended_lookbacks(c) for h, c in ORIGINAL_HORIZON_LOOKBACKS.items()}
REPRESENTATIVE_LOOKBACK = {h: c[len(c) // 2] for h, c in ORIGINAL_HORIZON_LOOKBACKS.items()}

UNIT_CANDIDATES = [32, 64, 128, 256]
LR_CANDIDATES = [0.01, 0.001, 0.0001]
DROPOUT_CANDIDATES = [0.0, 0.2]
FULL_GRID = list(itertools.product(UNIT_CANDIDATES, LR_CANDIDATES, DROPOUT_CANDIDATES))
REPEATS = 2

DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.0

FINE_STEP_DAYS = 2
MIN_LOOKBACK_FLOOR = 5
SAVE_FORECAST_PLOTS = False

GRID_LOG_COLUMNS = ["horizon_days", "target", "model", "units", "learning_rate", "dropout", "validation_MASE", "source"]
WINNING_CONFIG_COLUMNS = ["horizon_days", "target", "model", "units", "learning_rate", "dropout", "val_MASE"]
HP_COMPARISON_COLUMNS = [
    "horizon_days", "target", "model",
    "ofat_units", "ofat_lr", "ofat_dropout", "ofat_val_MASE",
    "fullgrid_units", "fullgrid_lr", "fullgrid_dropout", "fullgrid_val_MASE",
    "config_changed", "val_MASE_improvement",
]
SUMMARY_COLUMNS = [
    "horizon_days", "lookback_days", "target", "model",
    "tuned_units", "tuned_learning_rate", "tuned_dropout",
    "test_start", "test_end",
    "MAE", "RMSE", "MAPE", "MASE",
    "retrained_after_nan", "source",
]
DAILY_COLUMNS = ["horizon_days", "lookback_days", "target", "model", "date", "actual", "predicted"]
LOOKBACK_COMPARISON_COLUMNS = [
    "horizon_days", "target", "model",
    "v2_best_lookback", "v2_best_MASE",
    "v3_best_lookback", "v3_best_MASE",
    "lookback_moved_by_days", "mase_improvement", "hyperparameters_changed",
]


def de_number_to_float(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    mask = s.str.contains(",", na=False)
    s2 = s.copy()
    s2.loc[mask] = s2.loc[mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
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
    daily = df.groupby("Posting_Date").agg(
        {"Sales_Amount_Actual": "sum", "Cost_Amount_Actual": "sum", "Margin": "sum"}
    ).sort_index()
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


def mase_score(actual_values, pred_values, insample_naive_mae) -> float:
    if insample_naive_mae is None or np.isnan(insample_naive_mae) or insample_naive_mae == 0:
        return float('nan')
    mae = float(np.nanmean(np.abs(pred_values - actual_values)))
    return mae / insample_naive_mae


def mape_score(actual_values, pred_values) -> float:
    errors = pred_values - actual_values
    nonzero = np.where(actual_values != 0, actual_values, np.nan)
    return float(np.nanmean(np.abs(errors) / np.abs(nonzero)) * 100)


def evaluate_config(model_name, lookback_days, units, learning_rate, dropout,
                     X, y, scaler, last_window, horizon_days,
                     tune_val_actual, future_calendar_scaled, insample_naive_mae, repeats=REPEATS):
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


def train_and_forecast_once(model_name, lookback_days, cfg, X, y, scaler, scaled_train, horizon_days, future_calendar_scaled):
    m = fit_model(model_name, lookback_days, cfg["units"], cfg["learning_rate"], cfg["dropout"], X, y)
    last_window = scaled_train[-lookback_days:]
    preds_scaled = forecast(m, last_window, horizon_days, lookback_days, model_name, future_calendar_scaled)
    preds = inverse_transform_target(preds_scaled, scaler)
    del m
    release_gpu_memory()
    return preds


def append_rows_csv(path: Path, rows: list, columns: list):
    if not rows:
        return
    df_rows = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    df_rows.to_csv(path, mode="a", header=write_header, index=False)


def reconstruct_v2_ofat_triples(cell_tuning_log: pd.DataFrame) -> dict:
    s1 = cell_tuning_log[cell_tuning_log.parameter == "units"]
    s2 = cell_tuning_log[cell_tuning_log.parameter == "learning_rate"]
    s3 = cell_tuning_log[cell_tuning_log.parameter == "dropout"]
    if s1.empty or s2.empty or s3.empty:
        return {}
    best_units = s1.loc[s1.is_winner, "value"].iloc[0]
    best_lr = s2.loc[s2.is_winner, "value"].iloc[0]

    triples = {}
    for _, r in s1.iterrows():
        triples[(r.value, DEFAULT_LR, DEFAULT_DROPOUT)] = r.validation_MASE
    for _, r in s2.iterrows():
        triples[(best_units, r.value, DEFAULT_DROPOUT)] = r.validation_MASE
    for _, r in s3.iterrows():
        triples[(best_units, best_lr, r.value)] = r.validation_MASE
    return triples


def load_v2_ofat_log() -> pd.DataFrame:
    if not V2_TUNING_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{V2_TUNING_SUMMARY_PATH} not found -- run dl_forecasting_tuned_2.py first.")
    return pd.read_csv(V2_TUNING_SUMMARY_PATH)


def load_v2_winning_configs() -> dict:
    if not V2_WINNING_CONFIG_PATH.exists():
        return {}
    df = pd.read_csv(V2_WINNING_CONFIG_PATH)
    return {
        (int(r.horizon_days), r.target, r.model): {"units": int(r.units), "learning_rate": float(r.learning_rate), "dropout": float(r.dropout)}
        for _, r in df.iterrows()
    }


def load_grid_log() -> pd.DataFrame:
    if not GRID_LOG_PATH.exists():
        return pd.DataFrame(columns=GRID_LOG_COLUMNS)
    return pd.read_csv(GRID_LOG_PATH)


def load_winning_configs_v3() -> dict:
    if not WINNING_CONFIG_PATH.exists():
        return {}
    df = pd.read_csv(WINNING_CONFIG_PATH)
    return {
        (int(r.horizon_days), r.target, r.model): {"units": int(r.units), "learning_rate": float(r.learning_rate), "dropout": float(r.dropout)}
        for _, r in df.iterrows()
    }


def run_full_grid_tuning(daily: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("PHASE 1 -- FULL-GRID HYPERPARAMETER SEARCH (24 combos/cell)")
    print("Reusing v2's 7 already-tested combos/cell; only new combos are trained.")
    print("=" * 70)

    v2_ofat_log = load_v2_ofat_log()
    v2_winning = load_v2_winning_configs()
    winning_config = load_winning_configs_v3()
    grid_log = load_grid_log()

    total_cells = len(ORIGINAL_HORIZON_LOOKBACKS) * len(TARGETS) * len(MODELS)
    cells_done = len(winning_config)
    print(f"\n{cells_done} of {total_cells} cells already fully gridded in a previous run.\n")

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

                cell_log = grid_log[(grid_log.horizon_days == horizon_days) & (grid_log.target == target) & (grid_log.model == model_name)]
                already_logged = set(zip(cell_log.units, cell_log.learning_rate, cell_log.dropout)) if not cell_log.empty else set()

                print(f"\n  == {model_name} @ horizon={horizon_days}d, target={target} (grid) ==")

                if not enough_history:
                    print("    Not enough history -- safe defaults, matching v2's fallback")
                    cfg = {"units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT}
                    winning_config[key] = cfg
                    append_rows_csv(WINNING_CONFIG_PATH, [{"horizon_days": horizon_days, "target": target, "model": model_name,
                                                            "units": cfg["units"], "learning_rate": cfg["learning_rate"],
                                                            "dropout": cfg["dropout"], "val_MASE": ""}], WINNING_CONFIG_COLUMNS)
                    continue

                cell_ofat_log = v2_ofat_log[(v2_ofat_log.horizon_days == horizon_days) & (v2_ofat_log.target == target) & (v2_ofat_log.model == model_name)]
                reused_triples = reconstruct_v2_ofat_triples(cell_ofat_log)

                new_log_rows = []
                for (u, lr, d), mase in reused_triples.items():
                    if (u, lr, d) not in already_logged:
                        new_log_rows.append({"horizon_days": horizon_days, "target": target, "model": model_name,
                                              "units": u, "learning_rate": lr, "dropout": d,
                                              "validation_MASE": mase, "source": "reused_from_v2_ofat"})
                if new_log_rows:
                    append_rows_csv(GRID_LOG_PATH, new_log_rows, GRID_LOG_COLUMNS)
                    already_logged |= {(r["units"], r["learning_rate"], r["dropout"]) for r in new_log_rows}

                scaler, scaled_tune_train = make_scaled_2col(tune_train, target)
                X, y = create_sequences(scaled_tune_train, lookback_days)
                if len(X) == 0:
                    cfg = {"units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT}
                    winning_config[key] = cfg
                    append_rows_csv(WINNING_CONFIG_PATH, [{"horizon_days": horizon_days, "target": target, "model": model_name,
                                                            "units": cfg["units"], "learning_rate": cfg["learning_rate"],
                                                            "dropout": cfg["dropout"], "val_MASE": ""}], WINNING_CONFIG_COLUMNS)
                    continue
                last_window = scaled_tune_train[-lookback_days:]
                tune_val_actual = tune_val[target].values
                future_calendar_scaled = scale_calendar_values(tune_val[CALENDAR_COL].values, scaler)
                insample_mae = naive_insample_mae(tune_train[target].values)

                missing = [c for c in FULL_GRID if c not in already_logged]
                print(f"    {len(reused_triples)} combos reused from v2's OFAT log, {len(missing)} new combos to train")

                for (units, lr, dropout) in missing:
                    score = evaluate_config(model_name, lookback_days, units, lr, dropout, X, y, scaler, last_window,
                                             horizon_days, tune_val_actual, future_calendar_scaled, insample_mae)
                    print(f"    [new grid point] units={units:>4} lr={lr:<8} dropout={dropout:<4} MASE={score:.4f}")
                    append_rows_csv(GRID_LOG_PATH, [{"horizon_days": horizon_days, "target": target, "model": model_name,
                                                      "units": units, "learning_rate": lr, "dropout": dropout,
                                                      "validation_MASE": score, "source": "new_v3_grid"}], GRID_LOG_COLUMNS)

                full_cell_log = load_grid_log()
                full_cell_log = full_cell_log[(full_cell_log.horizon_days == horizon_days) & (full_cell_log.target == target) & (full_cell_log.model == model_name)]
                full_cell_log = full_cell_log.dropna(subset=["validation_MASE"])
                if full_cell_log.empty:
                    cfg = {"units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT}
                    best_mase = float('nan')
                else:
                    best_row = full_cell_log.loc[full_cell_log.validation_MASE.idxmin()]
                    cfg = {"units": int(best_row.units), "learning_rate": float(best_row.learning_rate), "dropout": float(best_row.dropout)}
                    best_mase = float(best_row.validation_MASE)

                winning_config[key] = cfg
                append_rows_csv(WINNING_CONFIG_PATH, [{"horizon_days": horizon_days, "target": target, "model": model_name,
                                                        "units": cfg["units"], "learning_rate": cfg["learning_rate"],
                                                        "dropout": cfg["dropout"], "val_MASE": round(best_mase, 4) if not np.isnan(best_mase) else ""}],
                                WINNING_CONFIG_COLUMNS)
                print(f"    -> FULL-GRID WINNER: {cfg} (validation MASE={best_mase:.4f})")

                ofat_cfg = v2_winning.get(key)
                if ofat_cfg is not None:
                    ofat_mase = reused_triples.get((ofat_cfg["units"], ofat_cfg["learning_rate"], ofat_cfg["dropout"]))
                    changed = ofat_cfg != cfg
                    improvement = (ofat_mase - best_mase) if (ofat_mase is not None and not np.isnan(best_mase)) else None
                    append_rows_csv(HP_COMPARISON_PATH, [{
                        "horizon_days": horizon_days, "target": target, "model": model_name,
                        "ofat_units": ofat_cfg["units"], "ofat_lr": ofat_cfg["learning_rate"], "ofat_dropout": ofat_cfg["dropout"],
                        "ofat_val_MASE": ofat_mase,
                        "fullgrid_units": cfg["units"], "fullgrid_lr": cfg["learning_rate"], "fullgrid_dropout": cfg["dropout"],
                        "fullgrid_val_MASE": round(best_mase, 4) if not np.isnan(best_mase) else "",
                        "config_changed": changed,
                        "val_MASE_improvement": round(improvement, 4) if improvement is not None else "",
                    }], HP_COMPARISON_COLUMNS)

    print(f"\nPhase 1 complete: {len(winning_config)}/{total_cells} cells have a full-grid winner.")
    return winning_config


def load_completed_v3_points() -> set:
    if not SUMMARY_PATH.exists():
        return set()
    existing = pd.read_csv(SUMMARY_PATH)
    if existing.empty:
        return set()
    return set(zip(existing.horizon_days, existing.lookback_days, existing.target, existing.model))


def plot_forecast(horizon_days, lookback_days, model_name, target, train, test, preds_col):
    wdir = FORECAST_OUTDIR / f"h{horizon_days}d" / f"lb{lookback_days}d"
    wdir.mkdir(parents=True, exist_ok=True)
    plot_start = -(horizon_days * 3)
    plt.figure(figsize=(10, 5))
    plt.plot(train.index[plot_start:], train[target].iloc[plot_start:], label="Train")
    plt.plot(test.index, test[target], label="Test (Actual)", linewidth=2)
    plt.plot(test.index, preds_col, label=f"{model_name} Forecast (v3)")
    plt.axvspan(test.index[0], test.index[-1], alpha=0.2)
    plt.title(f"{target} | {model_name} (v3 full-grid + full-range)\nLookback: {lookback_days}d | Horizon: {horizon_days}d")
    plt.legend()
    plt.tight_layout()
    plt.savefig(wdir / f"{target}_{model_name}.png", dpi=150)
    plt.close()


def run_full_range_lookback_sweep(daily: pd.DataFrame, winning_config: dict):
    print("\n" + "=" * 70)
    print("PHASE 2 -- FULL-RANGE LOOKBACK SWEEP, every " + str(FINE_STEP_DAYS) + " days")
    print("=" * 70)

    v2_winning = load_v2_winning_configs()
    v2_summary = pd.read_csv(V2_SUMMARY_PATH) if V2_SUMMARY_PATH.exists() else pd.DataFrame()

    series_len = len(daily)
    full_range = {}
    for horizon_days, candidates in ORIGINAL_HORIZON_LOOKBACKS.items():
        lo, hi = min(candidates), max(candidates)
        hi = min(hi, series_len - horizon_days - 2)
        full_range[horizon_days] = [v for v in range(lo, hi + 1, FINE_STEP_DAYS) if v >= MIN_LOOKBACK_FLOOR]

    total_planned = sum(len(full_range[h]) for h in ORIGINAL_HORIZON_LOOKBACKS) * len(TARGETS) * len(MODELS)
    completed = load_completed_v3_points()
    print(f"\nPlanned points across all 45 cells: {total_planned}. Already done: {len(completed)}.\n")

    split_cache, scaler_cache = {}, {}
    done_this_run = 0

    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        if horizon_days not in split_cache:
            split_cache[horizon_days] = (daily.iloc[:-horizon_days], daily.iloc[-horizon_days:])
        train, test = split_cache[horizon_days]
        v2_coarse_points = set(V2_HORIZON_LOOKBACKS[horizon_days])

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
                key = (horizon_days, target, model_name)
                v3_cfg = winning_config.get(key, {"units": 128, "learning_rate": DEFAULT_LR, "dropout": DEFAULT_DROPOUT})
                v2_cfg = v2_winning.get(key)
                config_unchanged = (v2_cfg is not None and v2_cfg == v3_cfg)

                for lookback_days in full_range[horizon_days]:
                    if (horizon_days, lookback_days, target, model_name) in completed:
                        continue

                    if config_unchanged and lookback_days in v2_coarse_points and not v2_summary.empty:
                        v2_row = v2_summary[(v2_summary.horizon_days == horizon_days) & (v2_summary.lookback_days == lookback_days) &
                                             (v2_summary.target == target) & (v2_summary.model == model_name)]
                        if not v2_row.empty:
                            r = v2_row.iloc[0]
                            append_rows_csv(SUMMARY_PATH, [{
                                "horizon_days": horizon_days, "lookback_days": lookback_days, "target": target, "model": model_name,
                                "tuned_units": v3_cfg["units"], "tuned_learning_rate": v3_cfg["learning_rate"], "tuned_dropout": v3_cfg["dropout"],
                                "test_start": r.test_start, "test_end": r.test_end,
                                "MAE": r.MAE, "RMSE": r.RMSE, "MAPE": r.MAPE, "MASE": r.MASE,
                                "retrained_after_nan": r.retrained_after_nan, "source": "reused_from_v2",
                            }], SUMMARY_COLUMNS)
                            done_this_run += 1
                            continue

                    X, y = create_sequences(scaled_train, lookback_days)
                    if len(X) == 0:
                        done_this_run += 1
                        continue

                    print(f"  [{len(completed) + done_this_run + 1}/{total_planned}] h={horizon_days}d lb={lookback_days}d "
                          f"target={target} model={model_name}")

                    pred_values = train_and_forecast_once(model_name, lookback_days, v3_cfg, X, y, scaler, scaled_train, horizon_days, future_calendar_scaled)
                    retrained = False
                    if np.isnan(pred_values).any():
                        print("    !! NaN forecast -- retraining once")
                        pred_values = train_and_forecast_once(model_name, lookback_days, v3_cfg, X, y, scaler, scaled_train, horizon_days, future_calendar_scaled)
                        retrained = True

                    errors = pred_values - actual_values
                    mae = float(np.nanmean(np.abs(errors)))
                    rmse = float(np.sqrt(np.nanmean(errors ** 2)))
                    mape = mape_score(actual_values, pred_values)
                    mase = mase_score(actual_values, pred_values, insample_mae)

                    append_rows_csv(SUMMARY_PATH, [{
                        "horizon_days": horizon_days, "lookback_days": lookback_days, "target": target, "model": model_name,
                        "tuned_units": v3_cfg["units"], "tuned_learning_rate": v3_cfg["learning_rate"], "tuned_dropout": v3_cfg["dropout"],
                        "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
                        "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                        "MAPE": round(mape, 2) if not np.isnan(mape) else "",
                        "MASE": round(mase, 4) if not np.isnan(mase) else "",
                        "retrained_after_nan": retrained, "source": "new_v3",
                    }], SUMMARY_COLUMNS)
                    append_rows_csv(DAILY_PATH, [
                        {"horizon_days": horizon_days, "lookback_days": lookback_days, "target": target, "model": model_name,
                         "date": str(d.date()), "actual": round(float(a), 2), "predicted": round(float(p), 2) if not np.isnan(p) else ""}
                        for d, a, p in zip(test.index, actual_values, pred_values)
                    ], DAILY_COLUMNS)
                    if SAVE_FORECAST_PLOTS:
                        plot_forecast(horizon_days, lookback_days, model_name, target, train, test, pred_values)
                    done_this_run += 1

    print(f"\nPhase 2: {done_this_run} point(s) this run. Total so far: {len(completed) + done_this_run}/{total_planned}.")
    if len(completed) + done_this_run < total_planned:
        print("Re-run this script to continue with the remaining points.")


def build_lookback_comparison_and_plots():
    if not SUMMARY_PATH.exists():
        print("No v3 summary yet.")
        return
    v3 = pd.read_csv(SUMMARY_PATH)
    v3["MASE"] = pd.to_numeric(v3["MASE"], errors="coerce")
    v2 = pd.read_csv(V2_SUMMARY_PATH) if V2_SUMMARY_PATH.exists() else pd.DataFrame()
    if not v2.empty:
        v2["MASE"] = pd.to_numeric(v2["MASE"], errors="coerce")
    hp_compare = pd.read_csv(HP_COMPARISON_PATH) if HP_COMPARISON_PATH.exists() else pd.DataFrame()

    rows = []
    for horizon_days in ORIGINAL_HORIZON_LOOKBACKS:
        for target in TARGETS:
            for model_name in MODELS:
                v3_sub = v3[(v3.horizon_days == horizon_days) & (v3.target == target) & (v3.model == model_name)].dropna(subset=["MASE"])
                if v3_sub.empty:
                    continue
                v3_best = v3_sub.loc[v3_sub.MASE.idxmin()]
                hp_row = hp_compare[(hp_compare.horizon_days == horizon_days) & (hp_compare.target == target) & (hp_compare.model == model_name)] if not hp_compare.empty else pd.DataFrame()
                hp_changed = bool(hp_row.iloc[0].config_changed) if not hp_row.empty else None

                if not v2.empty:
                    v2_sub = v2[(v2.horizon_days == horizon_days) & (v2.target == target) & (v2.model == model_name)].dropna(subset=["MASE"])
                else:
                    v2_sub = pd.DataFrame()
                if v2_sub.empty:
                    continue
                v2_best = v2_sub.loc[v2_sub.MASE.idxmin()]

                rows.append({
                    "horizon_days": horizon_days, "target": target, "model": model_name,
                    "v2_best_lookback": int(v2_best.lookback_days), "v2_best_MASE": round(float(v2_best.MASE), 4),
                    "v3_best_lookback": int(v3_best.lookback_days), "v3_best_MASE": round(float(v3_best.MASE), 4),
                    "lookback_moved_by_days": int(v3_best.lookback_days) - int(v2_best.lookback_days),
                    "mase_improvement": round(float(v2_best.MASE) - float(v3_best.MASE), 4),
                    "hyperparameters_changed": hp_changed,
                })

    report = pd.DataFrame(rows, columns=LOOKBACK_COMPARISON_COLUMNS)
    report.to_csv(LOOKBACK_COMPARISON_PATH, index=False)
    print("\n" + "=" * 70)
    print("v2 (coarse grid + OFAT) vs v3 (full grid + full-range lookback)")
    print("=" * 70)
    moved = report[report.lookback_moved_by_days != 0]
    print(f"{len(moved)} of {len(report)} cells found a better lookback than v2's coarse grid.")
    if len(moved):
        print(moved.sort_values("mase_improvement", ascending=False).to_string(index=False))
    print(f"\nFull comparison saved to {LOOKBACK_COMPARISON_PATH}")

    for horizon_days in sorted(ORIGINAL_HORIZON_LOOKBACKS.keys()):
        for model_name in MODELS:
            sub = v3[(v3.horizon_days == horizon_days) & (v3.model == model_name)].dropna(subset=["MASE"])
            if sub.empty:
                continue
            wdir = OPTIMUM_OUTDIR / f"h{horizon_days}d"
            wdir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(9, 5.5))
            for target in TARGETS:
                tsub = sub[sub.target == target].sort_values("lookback_days")
                if tsub.empty:
                    continue
                color = TARGET_COLORS[target]
                label = target.replace("_Amount_Actual", "")
                plt.plot(tsub.lookback_days, tsub.MASE, color=color, linewidth=1.3, alpha=0.85, label=label)
                best_idx = tsub.MASE.idxmin()
                plt.scatter([tsub.loc[best_idx, "lookback_days"]], [tsub.loc[best_idx, "MASE"]],
                            color=color, s=150, zorder=5, edgecolors='black', linewidths=1.3)
            plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            plt.xlabel('Lookback (days)')
            plt.ylabel('MASE (1.0 = as good as naive)')
            plt.title(f'{model_name} — full-range optimum lookback (v3)\nHorizon: {horizon_days}d '
                      f'(every {FINE_STEP_DAYS} days tested, ringed point = best)')
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(wdir / f"{model_name}_optimum_curve_v3.png", dpi=150)
            plt.close()
    print(f"Full-range optimum-curve plots saved to {OPTIMUM_OUTDIR}")


if __name__ == "__main__":
    daily_series = load_daily_series()

    n_grid_cells = len(ORIGINAL_HORIZON_LOOKBACKS) * len(TARGETS) * len(MODELS)
    print(f"\nPhase 1 will grid up to {n_grid_cells * len(FULL_GRID)} (cell, hyperparameter-combo) points "
          f"total, minus whatever v2's OFAT log already covers per cell.")
    n_range_points = sum(
        len([v for v in range(min(c), min(max(c), len(daily_series) - h - 2) + 1, FINE_STEP_DAYS) if v >= MIN_LOOKBACK_FLOOR])
        for h, c in ORIGINAL_HORIZON_LOOKBACKS.items() for c in [ORIGINAL_HORIZON_LOOKBACKS[h]]
    ) * len(TARGETS) * len(MODELS)
    print(f"Phase 2 will sweep up to {n_range_points} (horizon, lookback, target, model) points "
          f"total, minus whatever matches v2's coarse grid AND unchanged hyperparameters.")
    print("Both phases are resumable -- safe to Ctrl+C and re-run at any time.\n")

    winning_config_v3 = run_full_grid_tuning(daily_series)
    run_full_range_lookback_sweep(daily_series, winning_config_v3)
    build_lookback_comparison_and_plots()

    print(f"\n✅ v3 PIPELINE DONE — results saved under {OUTPUT_ROOT}")
    print(f"  Hyperparameter grid log:     {GRID_LOG_PATH.name}")
    print(f"  OFAT-vs-full-grid comparison: {HP_COMPARISON_PATH.name}")
    print(f"  Final summary/daily:         {SUMMARY_PATH.name} / {DAILY_PATH.name}")
    print(f"  v2-vs-v3 lookback comparison: {LOOKBACK_COMPARISON_PATH.name}")
    print(f"  Full-range optimum plots:     {OPTIMUM_OUTDIR}\\h<horizon>d\\<model>_optimum_curve_v3.png")

