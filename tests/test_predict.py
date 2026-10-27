"""Check that saved scaling and checkpoints reproduce inference in kW."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.phase2 import FEATURES, Windows
from src.predict import forecast
from src.train import get_model


class InferenceTests(unittest.TestCase):
    def test_mlp_accepts_noncontiguous_inputs(self):
        model = get_model('mlp', 14, 168, 24).eval()
        x = torch.randn(2, 14, 168).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x.contiguous()))

    def test_checkpoint_and_scaling_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'models').mkdir()
            (root / 'results').mkdir()
            model = get_model('mlp', 14, 168, 24)
            for param in model.parameters():
                torch.nn.init.zeros_(param)
            torch.save(model.state_dict(), root / 'models/best_mlp.pt')
            scale = {'feature_min': [0.] * 14, 'feature_max': [1.] * 14,
                     'target_min': [2.], 'target_max': [6.]}
            (root / 'results/split_manifest.json').write_text(json.dumps({'scalers': scale}))
            df = pd.DataFrame(np.ones((168, 14)), columns=FEATURES,
                              index=pd.date_range('2020-01-01', periods=168, freq='h', name='datetime'))
            csv = root / 'input.csv'
            df.to_csv(csv)
            result = forecast(csv, 'mlp', root)
            self.assertEqual(len(result), 24)
            self.assertEqual(result.datetime.iloc[0], df.index[-1] + pd.Timedelta(hours=1))
            np.testing.assert_allclose(result.predicted_active_power_kw, 2.)
            df.index = df.index.where(df.index != df.index[-1], df.index[-1] + pd.Timedelta(hours=1))
            df.to_csv(csv)
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                forecast(csv, 'mlp', root)

    def test_reject_misaligned_and_nonfinite_windows(self):
        with self.assertRaisesRegex(ValueError, 'matching rows'):
            Windows(np.zeros((200, 2)), np.zeros(199), np.ones(200))
        with self.assertRaisesRegex(ValueError, 'finite'):
            Windows(np.full((200, 2), np.nan), np.zeros(200), np.ones(200))


if __name__ == '__main__':
    unittest.main()
