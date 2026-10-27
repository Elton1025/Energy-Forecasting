"""Build a standalone Kaggle notebook from the versioned experiment sources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    paths = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'tests').glob('test*.py'))
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): p.read_text(encoding='utf-8') for p in paths}
    cells = []
    def cell(kind, source):
        item = {'cell_type': kind, 'id': f'forecast-{len(cells):02d}', 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code': item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '# Household power forecasting\n\nElton Lewis and Vignesh Rao\n\n'
         'We compare MLP, 1D-CNN and LSTM using identical chronological splits and three random seeds. '
         'All results below are computed on Kaggle. Code and documentation preparation used OpenAI Codex assistance. '
         'See the repository report for scope and limitations.\n')
    cell('code', 'from pathlib import Path\nimport json, os, subprocess, sys\n'
         'ROOT = Path("/kaggle/working/energy_forecasting")\nROOT.mkdir(exist_ok=True)\n'
         'os.chdir(ROOT)\nos.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n'
         'sources = ' + repr(sources) + '\n'
         'for name, content in sources.items():\n'
         '    p = ROOT / name\n    p.parent.mkdir(parents=True, exist_ok=True)\n    p.write_text(content, encoding="utf-8")\n'
         'subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)\n')
    cell('markdown', '## Training and evaluation\n'
         'A 168-hour history predicts 24 future hourly means. Scalers use training rows only; '
         'all target hours must have 60 observed readings. Validation MSE selects checkpoints. '
         'Test scores are reported without test-based tuning.\n')
    cell('code', 'for seed in [42, 43, 44]:\n'
         '    subprocess.run([sys.executable, "src/phase2.py", "--device", "cuda", '
         '"--epochs", "30", "--patience", "6", "--batch-size", "128", '
         '"--seed", str(seed), "--output", f"artifacts/seed_{seed}"], check=True)\n')
    cell('code', 'import pandas as pd\n'
         'frames = []\nfor seed in [42, 43, 44]:\n'
         '    frame = pd.read_csv(f"artifacts/seed_{seed}/results/model_comparison.csv")\n'
         '    frame["seed"] = seed\n    frames.append(frame)\n'
         'scores = pd.concat(frames, ignore_index=True)\n'
         'scores.to_csv("artifacts/all_runs.csv", index=False)\n'
         'summary = scores.groupby("model")[["MAE (kW)", "RMSE (kW)", "MAPE (%)", "R2 Score"]].agg(["mean", "std"])\n'
         'summary.to_csv("artifacts/summary.csv")\ndisplay(summary)\n'
         'subprocess.run([sys.executable, "src/predict.py", "--artifacts", "artifacts/seed_42", '
         '"--output", "artifacts/example_forecast.csv"], check=True)\n'
         'import shutil\nshutil.make_archive("/kaggle/working/forecast_results", "zip", ROOT / "artifacts")\n'
         'print("Artifacts saved to /kaggle/working/forecast_results.zip")\n')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    path = ROOT / 'notebooks/02_final_kaggle.ipynb'
    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
