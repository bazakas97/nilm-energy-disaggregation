# REEFLEX-NILM

Inference and training code for the REEFLEX NILM pipeline.

This repository is now organized around a small number of official entrypoints:

- `configs/active/release_eval.yaml`: main inference / evaluation config
- `configs/active/train_mains5_all10.yaml`: main training config
- `scripts/fetch_sel_daily.py`: fetch one day from SEL API
- `scripts/run_daily_eval.py`: generate per-house daily configs and run inference

Old experimental configs are still available under `configs/archive/`, but they are not the recommended starting point.

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you use CUDA, install the matching PyTorch build instead of the CPU wheel.

## Run inference / evaluation

Main command:

```bash
python run.py --config configs/active/release_eval.yaml
```

Before running, check these paths inside `configs/active/release_eval.yaml`:

- `paths.train_data`
- `paths.val_data`
- `paths.test_data`

The model bundle used for inference is already tracked in:

- `models/nilmformer_paper_mains5_all10_60_20_20/`

Outputs are written under `results/`:

- predictions CSV
- metrics JSON
- Plotly HTML plots

## Run training

Main training command:

```bash
python run.py --config configs/active/train_mains5_all10.yaml
```

This trains the current “official” NILMFormer-style configuration and saves outputs under `results/models/` and `results/plots/`.

## Daily SEL API inference

Set credentials:

```bash
export SEL_API_EMAIL="you@example.com"
export SEL_API_PASSWORD="your-password"
```

Fetch one day:

```bash
python scripts/fetch_sel_daily.py \
  --date 2026-03-15 \
  --participants certhr5fwl7p,certhckoz1h4 \
  --output-dir DATA/daily_sel_api
```

Run per-house inference on the fetched merged CSV:

```bash
python scripts/run_daily_eval.py \
  --base-config configs/active/release_eval.yaml \
  --date 2026-03-15 \
  --split-data-csv DATA/daily_sel_api/20260315/daily_20260315_merged.csv \
  --per-house \
  --house-overrides configs/active/house_overrides_daily.example.yaml \
  --run
```

If the daily CSV has no appliance labels, metrics are not meaningful. In that case, inspect the predictions CSVs and the HTML plots.

## Repository layout

- `configs/active/`: supported configs
- `configs/archive/`: legacy / experimental configs kept for reference
- `models/`: tracked inference model bundle
- `scripts/`: SEL fetch + daily run helpers
- `DATA/`: local datasets only, not tracked
- `results/`: generated outputs only, not tracked

## Docker

Docker support is available, but it is optional. See `DOCKER.md` if you want a containerized run.

