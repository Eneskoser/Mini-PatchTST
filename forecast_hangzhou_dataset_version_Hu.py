"""
Hangzhou Metro forecasting with PatchTST (version Hu).
Functionized pipeline; per-config training for station 9/15 inbound/outbound.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model.mini_patchtst import MiniPatchTST

from utilsforecast.preprocessing import fill_gaps

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
INPUT_LEN = 216
FORECAST_HORIZON = 1
VAL_WINDOW = 144
TEST_LAST = 144
BATCH_SIZE = 32
FEATURE_COLS = ["y_scaled", "hour_scaled", "week_day_scaled"]

# Training configs: (station_suffix, direction, epochs, num_layers)
# 9 inbound: 130 epochs, 3 layers; 9 outbound: 110, 4; 15 inbound: 110, 2; 15 outbound: 130, 3
TRAIN_CONFIGS = [
    ("9", "inbound", 130, 3),
    ("9", "outbound", 110, 4),
    ("15", "inbound", 110, 2),
    ("15", "outbound", 130, 3),
]


# -----------------------------------------------------------------------------
# Data loading and aggregation
# -----------------------------------------------------------------------------
def load_raw_and_aggregate(folder_path: str) -> pd.DataFrame:
    """Load all CSVs, parse time, aggregate to 10-min buckets by lineID, stationID, status."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["time"] = pd.to_datetime(combined_df["time"])
    combined_df["time_10min"] = combined_df["time"].dt.floor("10T")

    agg_df = combined_df.groupby(["lineID", "stationID", "status", "time_10min"]).size().reset_index(name="y")
    agg_df = agg_df[agg_df["time_10min"] < "2019-01-21"]
    agg_df = agg_df[agg_df["time_10min"] >= "2019-01-02"]
    return agg_df


def build_direction_dataset(agg_df: pd.DataFrame, status: int) -> pd.DataFrame:
    """Build inbound (status=0) or outbound (status=1) dataset with unique_id, fill gaps, scale."""
    direction_df = agg_df[agg_df["status"] == status].copy()
    direction_df["unique_id"] = direction_df["lineID"].astype(str) + "_" + direction_df["stationID"].astype(str)
    direction_df = direction_df.drop(columns=["lineID", "stationID", "status"])
    direction_df = direction_df.rename(columns={"time_10min": "ds"})
    direction_df = direction_df[direction_df["ds"] >= "2019-01-02"]

    df = fill_gaps(direction_df, freq="10min", start="2019-01-02")
    df["y"] = df["y"].fillna(0)

    df["ds"] = pd.to_datetime(df["ds"])
    df["hour"] = df["ds"].dt.hour
    df["week_day"] = df["ds"].dt.weekday
    station_encoder = LabelEncoder()
    df["station_idx"] = station_encoder.fit_transform(df["unique_id"])

    y_scaler = StandardScaler()
    hour_scaler = StandardScaler()
    week_day_scaler = StandardScaler()
    station_scaler = StandardScaler()

    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["y_scaled"] = y_scaler.fit_transform(df[["y"]])
    df["hour_scaled"] = hour_scaler.fit_transform(df[["hour"]])
    df["week_day_scaled"] = week_day_scaler.fit_transform(df[["week_day"]])
    df["station_scaled"] = station_scaler.fit_transform(df[["station_idx"]])

    return df, y_scaler


def build_sequences(
    station_df: pd.DataFrame,
    input_len: int,
    forecast_horizon: int,
    val_window: int,
    test_last: int,
):
    """
    Build X_train, y_train, X_val, y_val, X_test, y_test for the given station subset.
    Train: from input_len up to (total_len - test_last - val_window - forecast_horizon) for inbound,
           or (total_len - test_last - val_window - forecast_horizon) for outbound-style.
    Val: last val_window points before the test window.
    Test: last test_last points (aligned with Hu script: total_len - 144 - forecast_horizon + 1 backwards).
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    total_len = len(station_df)

    # Train: same as Hu (inbound: total_len - test_last - val_window - forecast_horizon + 1)
    train_end = total_len - test_last - val_window - forecast_horizon + 1
    for i in range(input_len, train_end):
        x = station_df.iloc[i - input_len : i][FEATURE_COLS].values.astype(np.float32)
        y = station_df.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
        X_train.append(x)
        y_train.append(y)

    # Validation window
    val_start = total_len - test_last - val_window - forecast_horizon + 1
    val_end = total_len - test_last - forecast_horizon + 1
    for i in range(val_start, val_end):
        x = station_df.iloc[i - input_len : i][FEATURE_COLS].values.astype(np.float32)
        y = station_df.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
        X_val.append(x)
        y_val.append(y)

    # Test: last test_last steps before end
    for i in range(
        total_len - test_last - forecast_horizon + 1,
        total_len - forecast_horizon + 1,
    ):
        x = station_df.iloc[i - input_len : i][FEATURE_COLS].values.astype(np.float32)
        y = station_df.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
        X_test.append(x)
        y_test.append(y)

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_val),
        np.array(y_val),
        np.array(X_test),
        np.array(y_test),
    )


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute MSE, RMSE, MAE, MAPE (excluding zero actuals)."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    nonzero = y_true_np != 0
    mape = (
        np.mean(np.abs((y_true_np[nonzero] - y_pred_np[nonzero]) / y_true_np[nonzero])) * 100 if np.any(nonzero) else np.nan
    )
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def run_config(
    df: pd.DataFrame,
    y_scaler: StandardScaler,
    epochs: int,
    num_layers: int,
    device: torch.device,
):
    """Run one config: filter stations, build sequences, train, evaluate on val and print metrics."""
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = build_sequences(
        df,
        INPUT_LEN,
        FORECAST_HORIZON,
        VAL_WINDOW,
        TEST_LAST,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MiniPatchTST(
        input_length=INPUT_LEN,
        forecast_horizon=FORECAST_HORIZON,
        patch_len=18,
        num_layers=num_layers,
        f_number=3,
    ).to(device)

    model = train_model(model, train_loader, epochs, device)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()

    pred_inv = y_scaler.inverse_transform(all_preds.flatten().reshape(-1, 1)).clip(0, np.inf)
    y_true_inv = y_scaler.inverse_transform(y_test.flatten().reshape(-1, 1))
    pred_inv = pred_inv.flatten()
    y_true_inv = y_true_inv.flatten()

    pred_inv = np.round(pred_inv).clip(0, None)
    y_true_inv = np.round(y_true_inv)

    metrics = evaluate_predictions(y_true_inv, pred_inv)
    print(f"MSE: {metrics['mse']:.4f} RMSE: {metrics['rmse']:.4f} MAE: {metrics['mae']:.4f} MAPE: {metrics['mape']:.2f}%")
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "data", "hangzhou_metro_dataset")
    if not os.path.isdir(folder_path):
        folder_path = os.path.join(script_dir, "..", "data", "hangzhou_metro_dataset")
    print("Data folder:", folder_path)

    agg_df = load_raw_and_aggregate(folder_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inbound: station 9 and 15
    df_in, y_scaler_in = build_direction_dataset(agg_df, status=0)
    df_09 = df_in[df_in.unique_id == "B_9"]
    df_15 = df_in[df_in.unique_id == "B_15"]
    print(f"Training for station Longxiang Bridge Station - inbound")
    run_config(df_09, y_scaler_in, epochs=130, num_layers=3, device=device)
    print(f"Training for station East Railway Station - inbound")
    run_config(df_15, y_scaler_in, epochs=110, num_layers=2, device=device)

    # Outbound: station 9 and 15
    df_out, y_scaler_out = build_direction_dataset(agg_df, status=1)
    df_09_out = df_out[df_out.unique_id == "B_9"]
    df_15_out = df_out[df_out.unique_id == "B_15"]
    print(f"Training for station Longxiang Bridge Station - outbound")
    run_config(df_09_out, y_scaler_out, epochs=110, num_layers=4, device=device)
    print(f"Training for station East Railway Station - outbound")
    run_config(df_15_out, y_scaler_out, epochs=130, num_layers=3, device=device)


if __name__ == "__main__":
    main()
