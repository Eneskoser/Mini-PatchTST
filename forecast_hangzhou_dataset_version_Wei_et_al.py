"""
Hangzhou Metro forecasting with PatchTST.
Runs A_80 (epochs=110, num_layers=4) and B_5 (epochs=110, num_layers=2).
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

from utilsforecast.preprocessing import fill_gaps

warnings.filterwarnings("ignore")
from model.mini_patchtst import MiniPatchTST

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
INPUT_LEN = 216
FORECAST_HORIZON = 1
TEST_LEN = 288
VAL_LEN = 144
BATCH_SIZE = 32
FEATURE_COLS = ["y_scaled", "hour_scaled", "week_day_scaled"]


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
def load_and_preprocess_metro_data(folder_path: str) -> pd.DataFrame:
    """Load CSVs, aggregate to 10-min buckets, add unique_id, fill gaps."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    print("Combined shape:", combined_df.shape)

    combined_df["time"] = pd.to_datetime(combined_df["time"])
    combined_df["time_10min"] = combined_df["time"].dt.floor("10T")

    agg_df = combined_df.groupby(["lineID", "stationID", "status", "time_10min"]).size().reset_index(name="y")
    final_df = agg_df.groupby(["lineID", "stationID", "time_10min"]).y.sum().reset_index()
    final_df["unique_id"] = final_df["lineID"].astype(str) + "_" + final_df["stationID"].astype(str)
    final_df = final_df.drop(columns=["lineID", "stationID"]).rename(columns={"time_10min": "ds"})

    df = fill_gaps(final_df, freq="10min", start="2019-01-01")
    df["y"] = df["y"].fillna(0)
    return df


def prepare_station_data(df: pd.DataFrame, unique_id: str):
    """
    Filter by unique_id, add hour/week_day, scale features.
    Returns (station_df, y_scaler).
    """
    station_df = df[df["unique_id"] == unique_id].reset_index(drop=True).copy()
    station_df["ds"] = pd.to_datetime(station_df["ds"])
    station_df["hour"] = station_df["ds"].dt.hour
    station_df["week_day"] = station_df["ds"].dt.weekday

    station_encoder = LabelEncoder()
    station_df["station_idx"] = station_encoder.fit_transform(station_df["unique_id"])

    y_scaler = StandardScaler()
    hour_scaler = StandardScaler()
    week_day_scaler = StandardScaler()
    station_scaler = StandardScaler()

    station_df = station_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    station_df["y_scaled"] = y_scaler.fit_transform(station_df[["y"]])
    station_df["hour_scaled"] = hour_scaler.fit_transform(station_df[["hour"]])
    station_df["week_day_scaled"] = week_day_scaler.fit_transform(station_df[["week_day"]])
    station_df["station_scaled"] = station_scaler.fit_transform(station_df[["station_idx"]])
    station_df["y_inverse_scaled"] = y_scaler.inverse_transform(station_df[["y_scaled"]])
    return station_df, y_scaler


def build_sequences(
    station_df: pd.DataFrame,
    input_len: int = INPUT_LEN,
    forecast_horizon: int = FORECAST_HORIZON,
    test_len: int = TEST_LEN,
    val_len: int = VAL_LEN,
):
    """
    Build train/val/test sequences for a single station (one unique_id).
    Returns (X_train, y_train, X_test, y_test, X_val, y_val) as numpy arrays.
    """
    station_ids = list(station_df["unique_id"].unique())
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for sid in station_ids:
        sub = station_df[station_df["unique_id"] == sid].dropna().reset_index(drop=True)
        total_len = len(sub)

        # Train: from input_len until (before last test_len + val_len)
        train_end = total_len - test_len - val_len - forecast_horizon + 1
        for i in range(input_len, train_end):
            x = sub.iloc[i - input_len : i][FEATURE_COLS].values.astype(np.float32)
            y = sub.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
            X_train.append(x)
            y_train.append(y)

        # Test: last test_len points
        test_start = total_len - test_len - forecast_horizon + 1
        test_end = total_len - forecast_horizon + 1
        for i in range(test_start, test_end):
            x = sub.iloc[i - input_len : i][FEATURE_COLS].values.astype(np.float32)
            y = sub.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
            X_test.append(x)
            y_test.append(y)

        # Validation:
        val_start = total_len - test_len - val_len - forecast_horizon + 1
        val_end = total_len - test_len - forecast_horizon + 1
        for i in range(val_start, val_end):
            x = sub.iloc[i - input_len : i][FEATURE_COLS].values.astype(np.float32)
            y = sub.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
            X_val.append(x)
            y_val.append(y)
    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_val),
        np.array(y_val),
        np.array(X_test),
        np.array(y_test),
    )


def get_dataloaders(X_train, y_train, X_test, y_test, X_val, y_val, batch_size: int = BATCH_SIZE):
    """Build PyTorch DataLoaders."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, test_loader, val_loader


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class PatchTST(nn.Module):
    """PatchTST: patch-based transformer for time series (3 features per timestep)."""

    def __init__(
        self,
        input_length: int,
        forecast_horizon: int,
        patch_len: int = 18,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.n_patches = input_length // patch_len
        self.d_model = d_model
        n_features = 3
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * self.n_patches, d_model)
        self.connect_res = nn.Linear(
            d_model + forecast_horizon * d_model,
            d_model + forecast_horizon * d_model,
        )
        self.final = nn.Linear(d_model, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        F = 3
        x = x.view(B, self.n_patches, self.patch_len, F)
        x = x.view(B, self.n_patches, self.patch_len * F)
        x = self.patch_embed(x) + self.pos_embed
        x = self.transformer_encoder(x)
        x = x.flatten(1)
        x = self.head(x)
        return self.final(x)


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
) -> list:
    """Train PatchTST; returns list of train losses per epoch."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss)
        scheduler.step()
    return train_losses


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    y_scaler: StandardScaler,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Run model on test set, inverse-scale, clip >= 0.
    Returns (metrics_dict, y_pred, y_true).
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()

    y_pred = y_scaler.inverse_transform(all_preds.flatten().reshape(-1, 1)).flatten()
    y_pred = np.clip(y_pred, 0, np.inf)
    y_true = y_scaler.inverse_transform(y_test.flatten().reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    metrics = {"mse": mse, "rmse": rmse, "mae": mae}
    return metrics


def run_station_forecast(
    df: pd.DataFrame,
    unique_id: str,
    epochs: int,
    num_layers: int,
    input_len: int = INPUT_LEN,
    forecast_horizon: int = FORECAST_HORIZON,
    device: torch.device | None = None,
) -> dict:
    """
    Full pipeline for one station: prepare data -> build sequences -> train -> evaluate.
    Returns test metrics dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    station_df, y_scaler = prepare_station_data(df, unique_id)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = build_sequences(
        station_df,
        input_len=input_len,
        forecast_horizon=forecast_horizon,
    )
    train_loader, test_loader, val_loader = get_dataloaders(X_train, y_train, X_test, y_test, X_val, y_val)

    model = MiniPatchTST(
        input_length=input_len,
        forecast_horizon=forecast_horizon,
        patch_len=18,
        num_layers=num_layers,
        f_number=3,
    ).to(device)
    if unique_id == "A_80":
        station_name = "Xinfeng"
    elif unique_id == "B_5":
        station_name = "Jinjiang"
    print(f"Training {station_name} | Epochs: {epochs}, Num layers: {num_layers}")
    train_model(model, train_loader, device, epochs)

    metrics = evaluate_model(model, test_loader, y_scaler, y_test, device)
    print(f"{station_name} Test — MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(20 * "=")
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    folder_path = "data/hangzhou_metro_dataset"
    df = load_and_preprocess_metro_data(folder_path)

    # A_80: epochs 110, encoder layers 4
    run_station_forecast(df, "A_80", epochs=110, num_layers=4)

    # B_5: epochs 110, encoder layers 2
    run_station_forecast(df, "B_5", epochs=110, num_layers=2)


if __name__ == "__main__":
    main()
