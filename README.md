# REEFLEX-NILM

Inference and training code for the REEFLEX NILM pipeline.

This repository is now organized around a small number of official entrypoints:

- `configs/active/train_mains5_all10.yaml`: main training config
- `configs/active/release_eval.yaml`: main inference / evaluation config
- `scripts/fetch_training_corpus.py`: fetch all participant/day data required by the training split CSVs
- `scripts/fetch_sel_daily.py`: fetch one day from SEL API
- `scripts/run_daily_eval.py`: generate per-house daily configs and run inference
- `scripts/run_daily_pipeline.py`: one-command fetch + inference wrapper for daily SEL runs

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

## 1) Fetch all data required for training

If you want to rebuild the training corpus from SEL API, use:

```bash
export SEL_API_EMAIL="you@example.com"
export SEL_API_PASSWORD="your-password"

python scripts/fetch_training_corpus.py \
  --split-dir DATA/splits_nilm_mains5_rebuilt_60_20_20 \
  --output-dir DATA/daily_sel_api_training_corpus \
  --estimate-only
```

This scans the configured `train.csv`, `val.csv`, and `test.csv`, extracts the required `(participant, date)` pairs, and prints:

- number of unique days
- number of participant-days
- rough ETA for the full download

For the current `DATA/splits_nilm_mains5_rebuilt_60_20_20` setup, the estimate is roughly:

- `670` unique days
- `2953` participant-days
- around `2.5 hours` with the default heuristic

To actually fetch the corpus:

```bash
python scripts/fetch_training_corpus.py \
  --split-dir DATA/splits_nilm_mains5_rebuilt_60_20_20 \
  --output-dir DATA/daily_sel_api_training_corpus
```

Notes:

- the script groups requests by day and participant list
- it forwards the same normalization as `fetch_sel_daily.py`
- it inserts a small sleep between the two SEL API calls made per participant to reduce 502 / overload issues
- the ETA is heuristic only; actual duration depends heavily on SEL server latency and availability

## 2) Run training

Main training command:

```bash
python run.py --config configs/active/train_mains5_all10.yaml
```

This trains the current “official” NILMFormer-style configuration and saves outputs under `results/models/` and `results/plots/`.

## 3) Run inference / evaluation

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

## 4) Daily SEL API inference

One-command daily pipeline:

```bash
python scripts/run_daily_pipeline.py \
  --date 2026-03-15 \
  --participants certhr5fwl7p,certhckoz1h4
```

This wrapper does:

1. fetch one day from SEL API
2. build the merged daily CSV
3. run per-house inference with `configs/active/release_eval.yaml`
4. automatically restrict reported outputs to the devices detected for each house from the fetched SEL sensors

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

This daily inference path is sensor-aware:

- it reads the fetched `*_sensors.json` for each participant
- keeps only the devices that actually belong to that house
- writes plots / prediction columns / reported metrics only for that subset

Daily outputs are written to:

- `DATA/daily_sel_api/YYYYMMDD/`: fetched and normalized daily data
- `results/generated_configs/daily_eval_<split>_YYYYMMDD/`: generated per-house configs
- `results/csv/`: predictions CSV and metrics JSON
- `results/plots_*/YYYYMMDD/<participant>/`: Plotly HTML plots per house

If the daily CSV has no appliance labels, metrics are not meaningful. In that case, inspect the predictions CSVs and the HTML plots.

## 5) Output semantics and current resolution

Why do daily raw data look 1-minute, but predictions / results look 4-6 minutes apart?

- fetched SEL daily data are regularized to a fixed 1-minute grid
- the current model is evaluated with `evaluate.stride: 6`
- this means we emit one prediction every 6 minutes, not every raw minute

Why do predictions start later than `00:00` and stop before `23:59`?

- the current configuration uses `window_size: 128`
- the dataset needs left/right context around each prediction center
- with a 128-step window on 1-minute data, the first valid center appears after the left context and the last valid center appears before the right context
- so edge timestamps are dropped because a full context window cannot be formed there

What do the prediction CSV columns mean?

- `*_true`: ground-truth power for that device, when available
- `*_pred`: model prediction for that device
- `*_known`: label-availability mask; `1` means the target label for that device is considered known/usable at that timestamp, `0` means missing / masked / not trusted
- `*_onoff_prob`: raw ON/OFF probability head output for that device before hard thresholding / hysteresis gating

## What preprocessing is currently implemented

Fetch-time preprocessing in `scripts/fetch_sel_daily.py`:

- converts SEL energy values to power
- reindexes each participant/day to a fixed 1-minute grid
- interpolates only short gaps
- drops house/day when mains missing ratio is too high

Inference-time preprocessing in the active configs:

- `participant_data_filter`
- EV `label_gap_fill`
- `unattributed_mains_mask`
- participant/device gating
- postprocessing thresholds and denoising from the active config / house overrides

This means the pipeline is operational, but house-specific postprocessing is still configurable and not “final for every house forever”.

## Repository layout

- `configs/active/`: supported configs
- `configs/archive/`: legacy / experimental configs kept for reference
- `models/`: tracked inference model bundle
- `scripts/`: SEL fetch + daily run helpers
- `DATA/`: local datasets only, not tracked
- `results/`: generated outputs only, not tracked

## Docker

Docker support is available, but it is optional. See `DOCKER.md` if you want a containerized run.
