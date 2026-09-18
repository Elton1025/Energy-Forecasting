"""Forecast the next 24 hours from a prepared hourly CSV and a Phase 2 checkpoint."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase2 import FEATURES
from src.train import get_model

def forecast(csv, model_name, artifacts):
    df = pd.read_csv(csv, index_col='datetime', parse_dates=True)
    if len(df) < 168 or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError('Input must contain at least 168 ordered, unique hourly rows')
    tail = df.iloc[-168:]
    if not tail.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Last 168 hours must be contiguous')
    x = tail[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('Features must all be finite')
    scaling = json.loads((artifacts/'results/split_manifest.json').read_text())['scalers']
    low, high = np.array(scaling['feature_min']), np.array(scaling['feature_max'])
    x = (x-low)/np.where(high==low, 1, high-low)
    model = get_model(model_name,14,168,24)
    model.load_state_dict(torch.load(artifacts/'models'/f'best_{model_name}.pt',map_location='cpu',weights_only=True))
    model.eval()
    with torch.no_grad(): pred = model(torch.tensor(x[None],dtype=torch.float32)).numpy().ravel()
    tmin, tmax = scaling['target_min'][0], scaling['target_max'][0]
    pred = pred*(tmax-tmin if tmax!=tmin else 1)+tmin
    return pd.DataFrame({'datetime':pd.date_range(tail.index[-1]+pd.Timedelta(hours=1),periods=24,freq='h'),
                         'predicted_active_power_kw':pred})

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['mlp','cnn1d','lstm'],default='mlp')
    p.add_argument('--artifacts',type=Path,default=ROOT/'submission/phase2')
    p.add_argument('--csv',type=Path)
    p.add_argument('--output',type=Path)
    a = p.parse_args(); torch.set_num_threads(2)
    result = forecast(a.csv or a.artifacts/'results/hourly_causal.csv',a.model,a.artifacts)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(a.output,index=False)
    print(result.to_string(index=False))
