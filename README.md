# NILMvPaper (REEFLEX NILM)

Code + configs for energy disaggregation (NILM) using a NILMFormer-style model, plus scripts to fetch daily data from SEL API and run inference per-house.

## What is tracked in git (and what is not)

- Tracked:
  - source code, configs, scripts
  - `models/` pretrained model bundle (for inference)
- Not tracked:
  - `DATA/` datasets (large / environment-specific)
  - `results/` outputs (plots/logs/predictions)
  - credentials (see `.env.example`)

## Local setup (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install PyTorch (choose CPU or CUDA build)
# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick inference / evaluation (uses your split CSVs)

1. Update the CSV paths in `config_release_eval.yaml` (`paths.train_data/val_data/test_data`), or just override via the daily scripts.
2. Run:

```bash
python run.py --config config_release_eval.yaml
```

Outputs are written under `results/` (plots, metrics JSON, predictions CSV).

## Daily inference (SEL API)

See `SEL_API.md`.

High-level:

1. Fetch a day:
   - `python scripts/fetch_sel_daily.py --date YYYY-MM-DD --participants ...`
2. Run per-house inference:
   - `python scripts/run_daily_eval.py --date YYYY-MM-DD --split-data-csv <daily_merged.csv> --per-house --run`

## Docker (CPU-friendly)

See `DOCKER.md` for build/run instructions.

