"""Scientific protocol checks, runnable with unittest (no network required)."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase2 import FEATURES, Windows, split_data
from src.train import get_model
from src.evaluate import calculate_metrics

class ProtocolTests(unittest.TestCase):
    def test_windows_contain_past_inputs_and_future_labels(self):
        x = np.arange(220, dtype=np.float32)
        ds = Windows(x[:, None], x, np.ones(220, dtype=bool))
        self.assertEqual(len(ds), 29)
        features, target = ds[0]
        np.testing.assert_array_equal(features[:, 0], np.arange(168))
        np.testing.assert_array_equal(target, np.arange(168, 192))

    def test_incomplete_labels_are_excluded(self):
        x = np.arange(220, dtype=np.float32)
        observed = np.ones(220, dtype=bool)
        observed[180] = False
        ds = Windows(x[:, None], x, observed)
        self.assertTrue(all(not (s+168 <= 180 < s+192) for s in ds.starts))

    def test_scaler_fits_training_only_and_splits_are_disjoint(self):
        df = pd.DataFrame(np.arange(2000)[:, None] * np.ones((1, 14)),
                          index=pd.date_range('2020-01-01', periods=2000, freq='h'), columns=FEATURES)
        df['target_observed'] = True
        datasets, _, fs, _, info = split_data(df)
        self.assertEqual(fs.data_max_[0], 1399)
        self.assertGreater(datasets[-1].x.max(), 1)
        self.assertLess(info['train']['end'], info['validation']['start'])
        self.assertLess(info['validation']['end'], info['test']['start'])

    def test_models_produce_24_outputs_and_receive_gradients(self):
        torch.set_num_threads(2)
        for name in ['mlp', 'cnn1d', 'lstm']:
            model = get_model(name, 14, 168, 24)
            pred = model(torch.randn(2, 168, 14))
            self.assertEqual(tuple(pred.shape), (2, 24))
            pred.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_metrics_known_answer(self):
        y = np.array([[1., 2.], [3., 4.]])
        perfect = calculate_metrics(y, y)
        self.assertEqual(perfect['MAE (kW)'], 0)
        self.assertEqual(perfect['R2 Score'], 1)
        shifted = calculate_metrics(y, y+1)
        self.assertEqual(shifted['MAE (kW)'], 1)
        self.assertEqual(shifted['RMSE (kW)'], 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
