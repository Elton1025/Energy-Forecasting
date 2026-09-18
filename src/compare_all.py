import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import calculate_metrics, calculate_hourly_mae, plot_sample_forecast, plot_horizon_error, generate_comparison_table

def main():
    root = Path(__file__).resolve().parent.parent
    saved_dir = root / "saved_models"
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    trues_path = saved_dir / "trues_test.npy"
    if not trues_path.exists():
        print(f"Ground truth file not found at {trues_path}. Please train at least one model first.")
        return

    y_true = np.load(trues_path)
    models = ["mlp", "cnn1d", "lstm"]
    model_labels = {"mlp": "MLP (Baseline)", "cnn1d": "1D-CNN", "lstm": "LSTM"}

    results = {}
    preds_dict = {}
    hourly_mae_dict = {}

    for m in models:
        pred_path = saved_dir / f"preds_{m}.npy"
        if pred_path.exists():
            y_pred = np.load(pred_path)
            label = model_labels[m]
            preds_dict[label] = y_pred
            results[label] = calculate_metrics(y_true, y_pred)
            hourly_mae_dict[label] = calculate_hourly_mae(y_true, y_pred)
            print(f"Loaded predictions for: {label}")
        else:
            print(f"Note: Predictions for {m.upper()} not found yet (run `python src/train.py --model {m}`).")

    if not results:
        print("No model predictions found to compare.")
        return

    # Comparison Table
    df_results = generate_comparison_table(results)
    print("\n" + "=" * 65)
    print("                OVERALL MODEL COMPARISON (TEST SET)")
    print("=" * 65)
    print(df_results.to_markdown())
    print("=" * 65)

    # Save to CSV
    table_csv = plots_dir / "model_comparison_metrics.csv"
    df_results.to_csv(table_csv)
    print(f"\nSaved metrics summary to: {table_csv}")

    # Generate multi-sample 24h forecast overlay
    forecast_plot_path = plots_dir / "forecast_comparison_24h.png"
    plot_sample_forecast(y_true, preds_dict, sample_indices=[12, 120, 300], save_path=forecast_plot_path)

    # Generate horizon error plot
    horizon_plot_path = plots_dir / "horizon_mae_error.png"
    plot_horizon_error(hourly_mae_dict, save_path=horizon_plot_path)

if __name__ == "__main__":
    main()
