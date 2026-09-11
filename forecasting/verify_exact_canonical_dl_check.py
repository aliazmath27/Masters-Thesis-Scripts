# Input: ILE_Modified.xlsx + DL_fine_tuned_3/winning_configs_v3.csv (run from the Thesis folder) | Output: dl_check/dl_forecast_summary_exact_canonical.csv, dl_forecast_daily_exact_canonical.csv

import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf

SEED = 42

RAW_FILE = Path("ILE_Modified.xlsx")
WINNING_CONFIG_PATH = Path("DL_fine_tuned_3/winning_configs_v3.csv")
OUT_DIR = Path("dl_check")
OUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_OUT = OUT_DIR / "dl_forecast_summary_exact_canonical.csv"
DAILY_OUT = OUT_DIR / "dl_forecast_daily_exact_canonical.csv"

TARGETS = ["Sales_Amount_Actual", "Cost_Amount_Actual", "Margin"]
MODELS = ["MLP", "CNN", "LSTM"]
CALENDAR_COL = "days_to_month_end"

MISSING_PAIRS = [(2, 30), (2, 90), (2, 180), (7, 30), (7, 90), (7, 180),
                  (30, 90), (30, 120), (30, 180), (60, 365), (90, 365)]

MAX_CELLS_THIS_RUN = int(os.environ["MAX_CELLS"]) if os.environ.get("MAX_CELLS") else None

SUMMARY_COLUMNS = ["horizon_days", "lookback_days", "target", "model",
                    "tuned_units", "tuned_learning_rate", "tuned_dropout",
                    "test_start", "test_end", "MAE", "RMSE", "MAPE", "MASE",
                    "retrained_after_nan", "source"]
DAILY_COLUMNS = ["horizon_days", "lookback_days", "target", "model", "date", "actual", "predicted"]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def de_number_to_float(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    mask = s.str.contains(",", na=False)
    s2 = s.copy()
    s2.loc[mask] = s2.loc[mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def load_daily_series() -> pd.DataFrame:
    df = pd.read_excel(RAW_FILE)
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


def load_done_cells() -> set:
    if not SUMMARY_OUT.exists():
        return set()
    d = pd.read_csv(SUMMARY_OUT)
    if d.empty:
        return set()
    return set(zip(d.horizon_days, d.lookback_days, d.target, d.model))


def main():
    print("Loading winning configs...")
    wc = pd.read_csv(WINNING_CONFIG_PATH)
    wc_map = {(int(r.horizon_days), r.target, r.model): {"units": int(r.units), "learning_rate": float(r.learning_rate), "dropout": float(r.dropout)}
              for _, r in wc.iterrows()}

    print("Loading raw daily series...")
    daily = load_daily_series()
    print(f"Daily series: {len(daily)} rows, {daily.index.min().date()} to {daily.index.max().date()}")

    done = load_done_cells()
    print(f"{len(done)} cells already done (resuming).")

    all_cells = [(h, lb, t, m) for (h, lb) in MISSING_PAIRS for t in TARGETS for m in MODELS]
    todo = [c for c in all_cells if c not in done]
    print(f"{len(todo)} of {len(all_cells)} cells remaining.")

    if MAX_CELLS_THIS_RUN is not None:
        todo = todo[:MAX_CELLS_THIS_RUN]
        print(f"[TEST MODE] limiting this run to {len(todo)} cells.")

    prep_cache = {}
    t_start = time.time()
    for i, (horizon_days, lookback_days, target, model_name) in enumerate(todo, 1):
        prep_key = (horizon_days, lookback_days, target)
        if prep_key not in prep_cache:
            train = daily.iloc[:-horizon_days]
            test = daily.iloc[-horizon_days:]
            scaler, scaled_train = make_scaled_2col(train, target)
            future_calendar_scaled = scale_calendar_values(test[CALENDAR_COL].values, scaler)
            insample_mae = naive_insample_mae(train[target].values)
            actual_values = test[target].values
            X, y = create_sequences(scaled_train, lookback_days)
            prep_cache[prep_key] = dict(train=train, test=test, scaler=scaler, scaled_train=scaled_train,
                                         future_calendar_scaled=future_calendar_scaled, insample_mae=insample_mae,
                                         actual_values=actual_values, X=X, y=y)
        P = prep_cache[prep_key]
        if len(P["X"]) == 0:
            print(f"  [WARN] no sequences for h={horizon_days} lb={lookback_days} target={target}, skipping {model_name}")
            continue
        cfg = wc_map.get((horizon_days, target, model_name))
        if cfg is None:
            print(f"  [WARN] no winning config for {(horizon_days, target, model_name)}, skipping")
            continue

        set_seed()
        t0 = time.time()
        pred_values = train_and_forecast_once(model_name, lookback_days, cfg, P["X"], P["y"], P["scaler"],
                                               P["scaled_train"], horizon_days, P["future_calendar_scaled"])
        retrained = False
        if np.isnan(pred_values).any():
            set_seed()
            pred_values = train_and_forecast_once(model_name, lookback_days, cfg, P["X"], P["y"], P["scaler"],
                                                   P["scaled_train"], horizon_days, P["future_calendar_scaled"])
            retrained = True

        actual_values = P["actual_values"]
        errors = pred_values - actual_values
        mae = float(np.nanmean(np.abs(errors)))
        rmse = float(np.sqrt(np.nanmean(errors ** 2)))
        mape = mape_score(actual_values, pred_values)
        mase = mase_score(actual_values, pred_values, P["insample_mae"])
        dt = time.time() - t0
        print(f"  [{i}/{len(todo)}] h={horizon_days} lb={lookback_days} target={target} model={model_name} "
              f"MASE={mase:.4f} ({dt:.1f}s)")

        test = P["test"]
        append_rows_csv(SUMMARY_OUT, [{
            "horizon_days": horizon_days, "lookback_days": lookback_days, "target": target, "model": model_name,
            "tuned_units": cfg["units"], "tuned_learning_rate": cfg["learning_rate"], "tuned_dropout": cfg["dropout"],
            "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
            "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2) if not np.isnan(mape) else "",
            "MASE": round(mase, 4) if not np.isnan(mase) else "",
            "retrained_after_nan": retrained, "source": "dl_check_independent_rerun",
        }], SUMMARY_COLUMNS)
        append_rows_csv(DAILY_OUT, [
            {"horizon_days": horizon_days, "lookback_days": lookback_days, "target": target, "model": model_name,
             "date": str(d.date()), "actual": round(float(a), 2), "predicted": round(float(p), 2) if not np.isnan(p) else ""}
            for d, a, p in zip(test.index, actual_values, pred_values)
        ], DAILY_COLUMNS)

    print(f"\n{time.time() - t_start:.1f}s total this run. Outputs in {OUT_DIR}/")
    print("Send dl_check/dl_forecast_summary_exact_canonical.csv (and the daily file, if you want the")
    print("day-by-day comparison too) back and Claude will diff it against DL_fine_tuned_3's original fill.")


if __name__ == "__main__":
    main()

