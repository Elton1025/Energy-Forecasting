# Short-Term Household Electric Power Consumption Forecasting

Multivariate 24-hour ahead time-series load forecasting using Deep Learning architectures:
- **Baseline**: Multi-Layer Perceptron (MLP)
- **Convolutional**: 1D-Convolutional Neural Network (1D-CNN)
- **Recurrent**: Long Short-Term Memory (LSTM)
- **Evaluation & Benchmarking**: Common regression metrics & comparative visualizations

---

## 1. Project Directory Structure

```text
DeepLeaarning-Project/
├── data/
│   ├── raw/                 # Original UCI dataset (zip & extracted txt)
│   └── processed/           # Resampled hourly dataset (household_power_hourly.csv)
├── saved_models/            # Saved PyTorch checkpoints (.pt) and predictions (.npy)
├── plots/                   # Loss curves, forecast comparisons, horizon errors
├── src/
│   ├── data_loader.py       # Resampling, sliding windows, chronological split
│   ├── download_and_preprocess.py # Automated raw data pipeline
│   ├── models/
│   │   ├── mlp.py           # Multi-Layer Perceptron baseline
│   │   ├── cnn1d.py         # 1D-CNN temporal feature extractor
│   │   └── lstm.py          # 2-layer LSTM recurrent architecture
│   ├── train.py             # Unified training script with EarlyStopping
│   ├── evaluate.py          # Metrics (MAE, RMSE, MAPE, R2) and plotting tools
│   └── compare_all.py       # Generates comparison tables and benchmark plots
├── requirements.txt
└── project.md
```

---

## 2. Environment Setup

Install the required packages:
```powershell
pip install -r requirements.txt
```

---

## 3. Data Preprocessing

If `data/processed/household_power_hourly.csv` is not yet generated, run:
```powershell
python src/download_and_preprocess.py
```
This will:
1. Download and extract the 2+ million minute-level records from UCI.
2. Impute missing values.
3. Resample the data into 1-hour averages (34,589 timestamps).
4. Engineer cyclical time encodings ($\sin/\cos$ for hour, day of week, and month).

---

## 4. Model Training

Each model can be trained independently using the unified training pipeline:

### Train the LSTM
```powershell
python src/train.py --model lstm --epochs 15 --batch_size 64 --lr 0.001
```

### Train the MLP Baseline
```powershell
python src/train.py --model mlp --epochs 15 --batch_size 64 --lr 0.001
```

### Train the 1D-CNN
```powershell
python src/train.py --model cnn1d --epochs 15 --batch_size 64 --lr 0.001
```

*Notes on Training:*
- Chronological train/validation/test split (70% / 15% / 15%) is enforced.
- Early stopping monitors validation loss to prevent overfitting.
- Metrics are calculated after inverse-scaling back to real kilowatt (kW) units.

---

## 5. Model Comparison & Final Evaluation

After training the models, run the common evaluation module:
```powershell
python src/compare_all.py
```
This produces:
1. **Summary Metrics Table**: Formatted side-by-side comparison of MAE, RMSE, MAPE, and $R^2$.
2. **`plots/forecast_comparison_24h.png`**: Overlaid 24-hour predictions against ground truth.
3. **`plots/horizon_mae_error.png`**: Plot showing error degradation from step +1 to +24.
