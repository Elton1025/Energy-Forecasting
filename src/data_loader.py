import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class PowerConsumptionDataset(Dataset):
    """PyTorch Dataset for multi-step time-series forecasting."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_windows(data_features: np.ndarray, target_values: np.ndarray, input_len: int = 168, output_len: int = 24):
    """
    Creates rolling window sequences.
    
    Args:
        data_features: array of shape (T, num_features)
        target_values: array of shape (T, 1) or (T,) - target column to predict
        input_len: number of past timesteps (default: 168 hours = 7 days)
        output_len: number of future timesteps to forecast (default: 24 hours)
    
    Returns:
        X: array of shape (N, input_len, num_features)
        y: array of shape (N, output_len)
    """
    X_list, y_list = [], []
    total_len = len(data_features)
    for i in range(total_len - input_len - output_len + 1):
        X_list.append(data_features[i : i + input_len])
        y_list.append(target_values[i + input_len : i + input_len + output_len].flatten())
    
    return np.array(X_list), np.array(y_list)

def load_and_split_data(
    csv_path: str = None,
    input_len: int = 168,
    output_len: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64
):
    """
    Loads hourly power consumption data, performs chronological split (no leakage),
    fits MinMaxScaler ONLY on the training split, creates sliding windows,
    and returns PyTorch DataLoaders along with the target scaler.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "household_power_hourly.csv"
    
    df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)

    feature_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend"
    ]
    target_col = "Global_active_power"

    data = df[feature_cols].values
    target = df[[target_col]].values

    # Chronological Split (Strictly preserve time order)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_target = target[:train_end]
    val_target = target[train_end:val_end]
    test_target = target[val_end:]

    # Normalization (Fit scaler on train set ONLY to prevent data leakage)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    train_data_scaled = feature_scaler.fit_transform(train_data)
    val_data_scaled = feature_scaler.transform(val_data)
    test_data_scaled = feature_scaler.transform(test_data)

    train_target_scaled = target_scaler.fit_transform(train_target)
    val_target_scaled = target_scaler.transform(val_target)
    test_target_scaled = target_scaler.transform(test_target)

    # Sliding Windows
    X_train, y_train = create_sliding_windows(train_data_scaled, train_target_scaled, input_len, output_len)
    X_val, y_val = create_sliding_windows(val_data_scaled, val_target_scaled, input_len, output_len)
    X_test, y_test = create_sliding_windows(test_data_scaled, test_target_scaled, input_len, output_len)

    # PyTorch DataLoaders
    train_dataset = PowerConsumptionDataset(X_train, y_train)
    val_dataset = PowerConsumptionDataset(X_val, y_val)
    test_dataset = PowerConsumptionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "input_len": input_len,
        "output_len": output_len,
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape,
        "X_test_shape": X_test.shape,
    }

    return train_loader, val_loader, test_loader, target_scaler, metadata
