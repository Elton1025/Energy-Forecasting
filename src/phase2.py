"""Reproducible Phase 2 experiment; preserves original saved artifacts."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import get_model, train_epoch, evaluate_epoch, predict_all
from src.evaluate import calculate_metrics, calculate_hourly_mae, plot_sample_forecast, plot_horizon_error
FEATURES = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend']

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

def prepare(out):
    raw = ROOT / 'data/raw/household_power_consumption.txt'
    if not raw.exists():
        from src.download_and_preprocess import download_and_extract
        download_and_extract()
    with raw.open('rb') as handle:
        raw_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    df = pd.read_csv(raw, sep=';', na_values=['?', ''], low_memory=False)
    idx = pd.to_datetime(df.pop('Date') + ' ' + df.pop('Time'), format='%d/%m/%Y %H:%M:%S')
    df.index = pd.DatetimeIndex(idx, name='datetime')
    df = df.apply(pd.to_numeric, errors='coerce').sort_index()
    if not df.index.is_unique:
        raise ValueError('Duplicate raw timestamps')
    observed = df['Global_active_power'].notna().resample('1h').sum()
    hourly = df.resample('1h').mean().ffill().dropna()
    hourly['target_observed'] = observed.reindex(hourly.index).ge(60)
    for prefix, values, period in [('hour', hourly.index.hour, 24), ('day', hourly.index.dayofweek, 7), ('month', hourly.index.month-1, 12)]:
        hourly[prefix+'_sin'] = np.sin(2*np.pi*values/period)
        hourly[prefix+'_cos'] = np.cos(2*np.pi*values/period)
    hourly['is_weekend'] = (hourly.index.dayofweek >= 5).astype(int)
    if not hourly.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Irregular hourly timestamps')
    audit = {'raw_sha256': raw_hash, 'raw_rows': len(df), 'hourly_rows': len(hourly),
             'start': str(hourly.index[0]), 'end': str(hourly.index[-1]),
             'missing_raw_by_column': df.isna().sum().to_dict(),
             'fully_observed_target_hours': int(hourly.target_observed.sum()),
             'excluded_incomplete_target_hours': int((~hourly.target_observed).sum()),
             'preprocessing': 'Observed hourly means, causal forward fill; labels require 60 observed minutes.', 'features': FEATURES}
    hourly.to_csv(out/'hourly_causal.csv')
    write_json(out/'data_audit.json', audit)
    return hourly, audit

class Windows(Dataset):
    def __init__(self, x, y, observed, input_len=168, output_len=24):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.input_len, self.output_len = input_len, output_len
        self.starts = np.array([i for i in range(len(x)-input_len-output_len+1)
                                if observed[i+input_len:i+input_len+output_len].all()], dtype=int)
        if not len(self.starts):
            raise ValueError('No valid windows')
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(self.x[s:s+self.input_len]), torch.from_numpy(self.y[s+self.input_len:s+self.input_len+self.output_len])

def split_data(df, batch_size=128, seed=42):
    n = len(df)
    bounds = [0, int(n*.70), int(n*.85), n]
    fs, ts = MinMaxScaler(), MinMaxScaler()
    fs.fit(df.iloc[:bounds[1]][FEATURES]); ts.fit(df.iloc[:bounds[1]][['Global_active_power']])
    datasets, loaders, info = [], [], {}
    for i, name in enumerate(['train', 'validation', 'test']):
        part = df.iloc[bounds[i]:bounds[i+1]]
        ds = Windows(fs.transform(part[FEATURES]), ts.transform(part[['Global_active_power']]), part.target_observed.to_numpy(dtype=bool))
        datasets.append(ds)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=i==0, generator=torch.Generator().manual_seed(seed)))
        info[name] = {'rows':len(part), 'windows':len(ds), 'start':str(part.index[0]), 'end':str(part.index[-1]),
                      'first_forecast':str(part.index[ds.starts[0]+168]), 'last_forecast_end':str(part.index[ds.starts[-1]+191])}
    info['boundary_indices'] = bounds
    info['scalers'] = {'feature_min':fs.data_min_.tolist(), 'feature_max':fs.data_max_.tolist(),
                       'target_min':ts.data_min_.tolist(), 'target_max':ts.data_max_.tolist()}
    return datasets, loaders, fs, ts, info

def inverse(a, scaler):
    return scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)

def train_model(name, loaders, scaler, args, models, images):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    loaders[0].generator.manual_seed(args.seed)
    model = get_model(name, 14, 168, 24)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = torch.nn.MSELoss()
    best, stale, records = float('inf'), 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        t = time.perf_counter()
        tr = train_epoch(model, loaders[0], loss, opt, 'cpu')
        va = evaluate_epoch(model, loaders[1], loss, 'cpu')
        records.append({'epoch':epoch, 'train_mse_scaled':tr, 'validation_mse_scaled':va, 'seconds':time.perf_counter()-t})
        print(f'{name} epoch {epoch}: train={tr:.6f} val={va:.6f} time={records[-1]["seconds"]:.1f}s', flush=True)
        if va < best:
            best, stale = va, 0
            torch.save(model.state_dict(), models/f'best_{name}.pt')
        else:
            stale += 1
        pd.DataFrame(records).to_csv(models/f'history_{name}.csv', index=False)
        if stale >= args.patience: break
    elapsed = time.perf_counter()-start
    model.load_state_dict(torch.load(models/f'best_{name}.pt', weights_only=True, map_location='cpu'))
    p, y = predict_all(model, loaders[2], 'cpu')
    p, y = inverse(p, scaler), inverse(y, scaler)
    np.save(models/f'preds_{name}.npy', p); np.save(models/'trues_test.npy', y)
    plt.figure(figsize=(7, 3.3))
    plt.plot([r['epoch'] for r in records], [r['train_mse_scaled'] for r in records], label='Training')
    plt.plot([r['epoch'] for r in records], [r['validation_mse_scaled'] for r in records], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('MSE (scaled)'); plt.title(name.upper()+' learning curve'); plt.legend(); plt.tight_layout()
    plt.savefig(images/f'{name}_loss_curve.png', dpi=220); plt.close()
    result = {'model':name, **calculate_metrics(y,p), 'parameters':sum(p.numel() for p in model.parameters()),
              'epochs_run':len(records), 'best_epoch':int(np.argmin([r['validation_mse_scaled'] for r in records])+1),
              'best_validation_mse':best, 'training_seconds':elapsed}
    write_json(models/f'metrics_{name}.json', result)
    return p, y, result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--output', type=Path, default=ROOT/'submission/phase2')
    args = ap.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1: ap.error('Positive counts required')
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    out = args.output.resolve()
    results, images, models = [out/p for p in ['results','images','models']]
    for p in [results,images,models]: p.mkdir(parents=True, exist_ok=True)
    write_json(results/'run_config.json', {**vars(args), 'torch_version':torch.__version__, 'numpy_version':np.__version__,
               'pandas_version':pd.__version__, 'python':sys.version, 'device':'cpu', 'input_hours':168,
               'forecast_hours':24, 'created_at':pd.Timestamp.now().isoformat()})
    df, audit = prepare(results)
    ds, loaders, fs, ts, info = split_data(df, args.batch_size, args.seed)
    write_json(results/'split_manifest.json', info)
    predictions, rows = {}, []
    for name in ['mlp','cnn1d','lstm']:
        p, y, row = train_model(name, loaders, ts, args, models, images)
        predictions[name.upper()] = p; rows.append(row)
        pd.DataFrame(rows).to_csv(results/'model_comparison.csv', index=False)
    for name, lag in [('Persistence',1),('Seasonal 24h',24),('Seasonal 168h',168)]:
        base = np.stack([np.repeat(ds[2].y[s+167],24) if lag==1 else ds[2].y[s+168-lag:s+192-lag] for s in ds[2].starts])
        base = inverse(base,ts); predictions[name] = base
        np.save(models/('preds_'+name.lower().replace(' ','_')+'.npy'),base)
        rows.append({'model':name, **calculate_metrics(y,base), 'parameters':0, 'epochs_run':0, 'best_epoch':0, 'training_seconds':0})
    pd.DataFrame(rows).to_csv(results/'model_comparison.csv', index=False)
    h = {name:calculate_hourly_mae(y,p) for name,p in predictions.items()}
    pd.DataFrame(h, index=pd.Index(range(1,25),name='hour_ahead')).to_csv(results/'horizon_mae.csv')
    plot_horizon_error(h,images/'horizon_mae_error.png')
    plot_sample_forecast(y,{k:v for k,v in predictions.items() if k in ['MLP','CNN1D','LSTM','Seasonal 24h']},[0,len(y)//2,len(y)-1],images/'forecast_comparison_24h.png')
    origins = df.index[info['boundary_indices'][2]+ds[2].starts+168]
    pd.DataFrame({'forecast_start':origins}).to_csv(results/'test_forecast_origins.csv',index=False)
    long = {'forecast_start':np.repeat(origins.astype(str),24), 'hour_ahead':np.tile(np.arange(1,25),len(y)), 'actual_kw':y.ravel()}
    for name,p in predictions.items(): long[name+'_kw'] = p.ravel()
    pd.DataFrame(long).to_csv(results/'test_predictions.csv.gz',index=False,compression='gzip')
    best_name = min(['MLP','CNN1D','LSTM'],key=lambda k:calculate_metrics(y,predictions[k])['MAE (kW)'])
    errors = np.abs(y-predictions[best_name]).mean(axis=1); worst = int(errors.argmax())
    plot_sample_forecast(y,{best_name:predictions[best_name]},[worst],images/'worst_forecast.png')
    threshold = float(df.iloc[:info['boundary_indices'][1]].Global_active_power.quantile(.9)); peak = y>=threshold
    write_json(results/'error_analysis.json', {'best_neural_by_test_mae':best_name,'worst_sample_index':worst,
               'worst_sample_start':str(origins[worst]),'worst_sample_mae_kw':float(errors[worst]),
               'train_90th_percentile_kw':threshold,'peak_target_count':int(peak.sum()),
               'peak_mae_kw':float(np.abs(y-predictions[best_name])[peak].mean()),
               'nonpeak_mae_kw':float(np.abs(y-predictions[best_name])[~peak].mean()),
               'note':'Overlapping forecast windows are not independent. Single-seed preliminary results.'})
    write_json(results/'completion.json',{'complete':True,'models':list(predictions),'test_windows':len(y)})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__ == '__main__': main()
