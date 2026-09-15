import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_split_data
from src.models.mlp import MLPForecaster
from src.models.cnn1d import CNN1DForecaster
from src.models.lstm import LSTMForecaster
from src.evaluate import calculate_metrics

def get_model(model_name: str, num_features: int, input_len: int, output_len: int):
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLPForecaster(seq_len=input_len, num_features=num_features, output_len=output_len)
    elif model_name in ["cnn", "cnn1d", "1d-cnn"]:
        return CNN1DForecaster(num_features=num_features, output_len=output_len)
    elif model_name == "lstm":
        return LSTMForecaster(num_features=num_features, hidden_dim=64, num_layers=2, output_len=output_len)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: mlp, cnn1d, lstm")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

def predict_all(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(y_batch.numpy())
    return np.vstack(all_preds), np.vstack(all_trues)

def main():
    parser = argparse.ArgumentParser(description="Train time-series forecasting models.")
    parser.add_argument("--model", type=str, default="lstm", choices=["mlp", "cnn1d", "lstm"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Model: {args.model.upper()}")

    # 1. Load Data
    print("\nLoading dataset and creating sliding windows...")
    train_loader, val_loader, test_loader, target_scaler, meta = load_and_split_data(
        input_len=168, output_len=24, batch_size=args.batch_size
    )

    # 2. Initialize Model
    model = get_model(args.model, meta["num_features"], meta["input_len"], meta["output_len"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop with Early Stopping
    save_dir = Path(__file__).resolve().parent.parent / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"best_{args.model}.pt"

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            improved = " [SAVED BEST]"
        else:
            patience_counter += 1

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {elapsed:.1f}s{improved}")

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds.")

    # 4. Evaluation on Test Set
    print("\nLoading best model for test set evaluation...")
    model.load_state_dict(torch.load(best_model_path))

    scaled_preds, scaled_trues = predict_all(model, test_loader, device)

    # Inverse transform to original kW scale
    # target_scaler was fitted on 2D shape (N, 1), so reshape before inverse transform
    unscaled_preds = target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).reshape(scaled_preds.shape)
    unscaled_trues = target_scaler.inverse_transform(scaled_trues.reshape(-1, 1)).reshape(scaled_trues.shape)

    metrics = calculate_metrics(unscaled_trues, unscaled_preds)
    print("\n" + "=" * 45)
    print(f"     TEST SET EVALUATION RESULTS ({args.model.upper()})")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"  {k:15s} : {v:.4f}")
    print("=" * 45)

    # Save predictions array for comparison
    np.save(save_dir / f"preds_{args.model}.npy", unscaled_preds)
    np.save(save_dir / "trues_test.npy", unscaled_trues)

    # Plot & Save Loss Curve
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_losses, label="Train Loss (MSE)", linewidth=2)
    plt.plot(val_losses, label="Validation Loss (MSE)", linewidth=2)
    plt.title(f"{args.model.upper()} - Training & Validation Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Scaled)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = plots_dir / f"{args.model}_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curve plot to: {loss_plot_path}")

if __name__ == "__main__":
    main()
