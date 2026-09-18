import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes standard regression metrics across the 24-hour horizon.
    Assumes y_true and y_pred are in original kW units.
    """
    # Flatten across batch and time for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)

    # Avoid division by zero in MAPE (epsilon = 1e-4)
    epsilon = 1e-4
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.clip(y_true_flat, epsilon, None))) * 100

    return {
        "MAE (kW)": float(mae),
        "RMSE (kW)": float(rmse),
        "MAPE (%)": float(mape),
        "R2 Score": float(r2)
    }

def calculate_hourly_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates MAE for each horizon step (+1h to +24h)."""
    return np.mean(np.abs(y_true - y_pred), axis=0)

def plot_sample_forecast(
    y_true: np.ndarray,
    predictions_dict: dict,
    sample_indices: list = [0, 50, 100],
    save_path: str = None
):
    """
    Plots multi-step 24-hour forecast overlays comparing models against ground truth.
    
    predictions_dict: {"LSTM": y_pred_lstm, "1D-CNN": y_pred_cnn, "MLP": y_pred_mlp}
    """
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples), sharex=True)
    if num_samples == 1:
        axes = [axes]

    hours = np.arange(1, y_true.shape[1] + 1)

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        ax.plot(hours, y_true[idx], label="Ground Truth (Actual)", color="black", linewidth=2.5, marker="o")
        
        for model_name, preds in predictions_dict.items():
            ax.plot(hours, preds[idx], label=f"Predicted ({model_name})", linestyle="--", linewidth=1.8)

        ax.set_title(f"24-Hour Ahead Forecast Sample #{idx}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Active Power (kW)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved forecast comparison plot to: {save_path}")
    plt.close()

def plot_horizon_error(hourly_mae_dict: dict, save_path: str = None):
    """
    Plots MAE degradation across forecast horizon (+1h to +24h) for all models.
    """
    plt.figure(figsize=(10, 5))
    for model_name, errors in hourly_mae_dict.items():
        hours = np.arange(1, len(errors) + 1)
        plt.plot(hours, errors, marker="o", label=model_name, linewidth=2)

    plt.title("Forecast Error (MAE) Across 24-Hour Horizon", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Step (Hours Ahead)", fontsize=11)
    plt.ylabel("Mean Absolute Error (kW)", fontsize=11)
    plt.xticks(np.arange(1, 25))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved horizon error plot to: {save_path}")
    plt.close()

def generate_comparison_table(results_dict: dict) -> pd.DataFrame:
    """
    Creates a summary comparison DataFrame for all models.
    results_dict format: {"Model Name": {"MAE (kW)": 0.35, "RMSE (kW)": 0.50, ...}}
    """
    df = pd.DataFrame(results_dict).T
    return df
