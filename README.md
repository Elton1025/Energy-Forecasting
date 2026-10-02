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

## 4. Official Phase 2 Reproducibility Pipeline

The authoritative workflow for generating the final Phase 2 interim report results is the unified `phase2.py` script. 

To run the complete experiment (data split, training all 3 models, baseline calculations, metrics generation, and plotting):
```powershell
python src/phase2.py --epochs 15 --batch-size 64
```

This will automatically output all artifacts to the `submission/phase2/` directory, including:
- `results/model_comparison.csv`: The complete metrics table (MAE, RMSE, MAPE, R2) for MLP, CNN, LSTM, and baselines.
- `results/horizon_mae.csv`: Error degradation over the 24h horizon.
- `images/`: The final benchmark plots and learning curves.

**Note:** For convenience, the final generated metrics and plots have been copied to the tracked `plots/` directory in this repository. 

---

## 5. (Optional) Modular Training & Evaluation

If you wish to train models individually or debug them, you can still use the original modular pipeline.

### Train Models Individually
```powershell
python src/train.py --model lstm --epochs 15 --batch_size 64 --lr 0.001
python src/train.py --model mlp --epochs 15 --batch_size 64 --lr 0.001
python src/train.py --model cnn1d --epochs 15 --batch_size 64 --lr 0.001
```

### Manual Comparison
After individual training, you can generate basic comparisons using:
```powershell
python src/compare_all.py
```
