"""
Run Mini-PatchTST on Istanbul metro data.
Structure: (1) Data reading and preparation, (2) Preprocessing for model, (3) Training.
"""

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.mini_patchtst import MiniPatchTST

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = "data/istanbul_data.csv"  # or "../data/final_metro_data.csv"
INPUT_LEN = 168
FORECAST_HORIZON = 3
BATCH_SIZE = 32
T_MAX = 200
LEARNING_RATE = 1e-3

# =============================================================================
# 1. Data reading and preparation
# =============================================================================


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load raw CSV and standardize columns for forecasting."""
    df = pd.read_csv(data_path)

    df["ds"] = pd.to_datetime(df["ds"])

    df["unique_id"] = df["line_name"].astype(str) + "_" + df["station_name"].astype(str)
    df = df.rename(columns={"number_of_passengers": "y"})
    df = df.drop(columns=["line_name", "station_name"])
    df = df[["unique_id", "ds", "y"]]
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


# =============================================================================
# 2. Preprocessing for model
# =============================================================================


def add_features_and_scale(df: pd.DataFrame) -> dict:
    """Add time/station features and fit scalers. Modifies df in place; returns scalers."""
    df["hour"] = df["ds"].dt.hour
    df["week_day"] = df["ds"].dt.weekday  # Monday=0, Sunday=6

    station_encoder = LabelEncoder()
    df["station_idx"] = station_encoder.fit_transform(df["unique_id"])

    y_scaler = StandardScaler()
    hour_scaler = StandardScaler()
    week_day_scaler = StandardScaler()
    station_scaler = StandardScaler()

    df["y_scaled"] = y_scaler.fit_transform(df[["y"]])
    df["hour_scaled"] = hour_scaler.fit_transform(df[["hour"]])
    df["week_day_scaled"] = week_day_scaler.fit_transform(df[["week_day"]])
    df["station_scaled"] = station_scaler.fit_transform(df[["station_idx"]])

    return {
        "y": y_scaler,
        "hour": hour_scaler,
        "week_day": week_day_scaler,
        "station": station_scaler,
    }


def build_sequences(df: pd.DataFrame, input_len: int, forecast_horizon: int, train: bool) -> tuple:
    """
    Build (X, y) sequences per station. X includes y_scaled, hour_scaled, week_day_scaled, station_scaled.
    train=True: all valid windows except the last one per station.
    train=False: only the last valid window per station (test set).
    Returns (X, y) as numpy arrays.
    """
    station_ids = list(df["unique_id"].unique())
    X_list, y_list = [], []

    for station_id in station_ids:
        station_df = df[df["unique_id"] == station_id].dropna().reset_index(drop=True)
        total_len = len(station_df)

        if train:
            indices = range(input_len, total_len - input_len - forecast_horizon + 1)
        else:
            start = total_len - input_len - forecast_horizon + 1
            end = total_len - forecast_horizon + 1
            indices = range(start, end)

        for i in indices:
            x = station_df.iloc[i - input_len : i][
                ["y_scaled", "hour_scaled", "week_day_scaled", "station_scaled"]
            ].values.astype(np.float32)
            y = station_df.iloc[i : i + forecast_horizon]["y_scaled"].values.astype(np.float32)
            X_list.append(x)
            y_list.append(y)

    return np.array(X_list), np.array(y_list)


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size: int):
    """Convert numpy arrays to PyTorch tensors and create DataLoaders."""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


# =============================================================================
# 3. Training model
# =============================================================================


def train_model(
    train_loader: DataLoader,
    device: torch.device,
    T_MAX: int = T_MAX,
    lr: float = LEARNING_RATE,
):
    """Train Mini-PatchTST and return list of per-epoch train losses."""
    NUM_LAYERS = 2
    model = MiniPatchTST(
        input_length=INPUT_LEN,
        forecast_horizon=FORECAST_HORIZON,
        num_layers=NUM_LAYERS,
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)

    train_losses = []
    epochs = 111
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss)
        scheduler.step()
    return model


# =============================================================================
# 4. Evaluation model
# =============================================================================


def evaluate_model(model, test_loader, device, y_scaler):
    criterion = torch.nn.MSELoss()
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            epoch_loss += loss.item()
            all_preds.append(preds.cpu())
        # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    prediction_inversed = y_scaler.inverse_transform(all_preds.flatten().reshape(-1, 1))
    y_true_inversed = y_scaler.inverse_transform(y_test.flatten().reshape(-1, 1))
    prediction_inversed_clipped = prediction_inversed.clip(0, np.inf)
    # Convert tensors to NumPy arrays
    y_true = y_true_inversed.flatten()
    y_pred = prediction_inversed_clipped.flatten()

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return mse, mae


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Data reading and preparation
    df = load_and_prepare_data(DATA_PATH)

    # 2. Preprocessing for model
    scalers = add_features_and_scale(df)

    X_train, y_train = build_sequences(df, INPUT_LEN, FORECAST_HORIZON, train=True)
    X_test, y_test = build_sequences(df, INPUT_LEN, FORECAST_HORIZON, train=False)

    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE)

    # 3. Training model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_model(train_loader, device, T_MAX=T_MAX, lr=LEARNING_RATE)

    # add an evaluation to evaluate the model
    mse, mae = evaluate_model(model, test_loader, device, scalers["y"])
    print(f"MSE: {mse}, MAE: {mae}")
