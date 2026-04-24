# REEFLEX-NILM

Inference and training code for the REEFLEX NILM pipeline.

This repository is now organized around a small number of official entrypoints:

- `configs/active/train_mains5_all10.yaml`: main training config
- `configs/active/train_sel_full_rebuilt_60_20_20.yaml`: full SEL-corpus rebuild training config
- `configs/active/release_eval.yaml`: main inference / evaluation config
- `configs/active/release_eval_sel_full_rebuilt.yaml`: inference config for the rebuilt model bundle
- `scripts/fetch_training_corpus.py`: fetch all participant/day data required by the training split CSVs
- `scripts/fetch_sel_full_corpus.py`: fetch all known SEL participants over a date range
- `scripts/build_splits_from_daily_corpus.py`: rebuild train/val/test splits from fetched daily SEL data
- `scripts/run_full_rebuild_pipeline.sh`: full fetch -> split build -> train -> bundle -> evaluate wrapper
- `scripts/fetch_sel_daily.py`: fetch one day from SEL API
- `scripts/run_daily_eval.py`: generate per-house daily configs and run inference
- `scripts/run_daily_pipeline.py`: one-command fetch + inference wrapper for daily SEL runs
- `scripts/run_daily_production.sh`: non-interactive wrapper for once-per-day production runs
- `scripts/sel_doctor.py`: preflight checker for env, configs, model bundle, splits, and SEL API availability
- `scripts/build_portfolio_dashboard.py`: build a static multi-day demo site from generated Plotly outputs

Old experimental configs are still available under `configs/archive/`, but they are not the recommended starting point.
Archived one-off utilities now live under `scripts/archive/` and are not part of the supported pipeline.

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

There are two supported fetch modes.

### Full SEL rebuild

Use this when you want to rescan all known SEL participants with the current parser and rebuild the model from scratch:

```bash
cp .env.example .env
# edit .env with SEL_API_EMAIL and SEL_API_PASSWORD

START_DATE=2024-02-01 \
END_DATE=2026-04-15 \
CUDA_VISIBLE_DEVICES=1 \
RUN_DAILY_TEST=1 \
DAILY_TEST_DATE=2026-04-15 \
bash scripts/run_full_rebuild_pipeline.sh
```

The wrapper performs:

1. full SEL fetch into `DATA/daily_sel_api_full_corpus/`
2. split rebuild into `DATA/splits_sel_full_rebuilt_60_20_20/`
3. GPU Docker build if needed
4. training with `configs/active/train_sel_full_rebuilt_60_20_20.yaml`
5. model bundle copy into `models/nilmformer_sel_full_rebuilt_60_20_20/`
6. evaluation with `configs/active/release_eval_sel_full_rebuilt.yaml`
7. optional daily inference test when `RUN_DAILY_TEST=1`

For a detached long run:

```bash
mkdir -p logs
screen -dmS nilm_full_rebuild bash -lc '
  cd /path/to/REEFLEX-NILM &&
  START_DATE=2024-02-01 \
  END_DATE=2026-04-15 \
  CUDA_VISIBLE_DEVICES=1 \
  RUN_DAILY_TEST=1 \
  DAILY_TEST_DATE=2026-04-15 \
  bash scripts/run_full_rebuild_pipeline.sh >> logs/full_rebuild.log 2>&1
'
tail -f logs/full_rebuild.log
```

### Fetch only the currently configured split days

Use this when you only want to refetch the participant/date pairs already present in an existing split directory:

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

Preflight / doctor check:

```bash
python scripts/sel_doctor.py \
  --date 2026-03-15 \
  --participant certhr5fwl7p
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

`scripts/run_daily_production.sh` now runs `scripts/sel_doctor.py` automatically by default before the daily pipeline. Set `DAILY_RUN_DOCTOR=0` if you want to skip that preflight.

## 5) Static Portfolio Demo

You can build a shareable static dashboard from already-generated daily evaluation plots.

Recommended demo window for the current full SEL corpus:

- `2024-08-02`
- `2024-08-03`
- `2024-08-04`

Generate the daily per-house plots first:

```bash
python scripts/run_daily_eval.py \
  --base-config configs/active/release_eval_sel_full_rebuilt.yaml \
  --date 2024-08-02 \
  --split-data-csv DATA/daily_sel_api_full_corpus/20240802/daily_20240802_merged.csv \
  --per-house \
  --run \
  --keep-going
```

Repeat for the other demo dates, then build the static site:

```bash
python scripts/build_portfolio_dashboard.py \
  --dates 20240802,20240803,20240804 \
  --clean
```

The generated site is written to:

- `demo/portfolio_dashboard/`

For a quick local preview:

```bash
cd demo/portfolio_dashboard
python -m http.server 8000
```

Then open:

- `http://localhost:8000`

### Containerized daily production

If you want the pipeline to run once per day on another machine, the supported path is:

1. put credentials and participant ids in `.env`
2. run the one-shot Docker service `reeflex-nilm-daily`
3. let the host scheduler (`systemd timer` or `cron`) invoke that service every day

Manual dry-run:

```bash
cp .env.example .env
# edit .env first

docker compose build reeflex-nilm-daily
DAILY_PRINT_ONLY=1 docker compose run --rm reeflex-nilm-daily 2026-03-15
```

Real run:

```bash
docker compose run --rm reeflex-nilm-daily
```

By default this daily service fetches the previous day and writes outputs back to the mounted repo. See `DOCKER.md` and `deploy/systemd/` for the production deploy files.

## 5) Output semantics and current resolution

Why do fetched raw data use 1-minute resolution, while the model config still says `evaluate.stride: 6`?

- fetched SEL daily data are regularized to a fixed 1-minute grid
- the current model still runs sliding windows with `evaluate.stride: 6`
- however, the active config also enables dense sequence reconstruction (`dense_sequence_reconstruction: true`)
- this means the seq2seq outputs from overlapping windows are merged back onto the original timeline
- so the exported predictions / plots now return to 1-minute resolution

Why do predictions now cover the full `00:00` to `23:59` daily range?

- the model itself is still window-based (`window_size: 128`)
- dense reconstruction merges all overlapping sequence predictions back to the underlying per-minute timeline
- the evaluation path also adds edge coverage for the first/last valid windows
- so a daily run with a complete 1440-row input day now exports a full 1440-row prediction timeline

Note:

- older artifacts generated before dense reconstruction was enabled may still look sparse or truncated
- the current `release_eval.yaml` behavior is full-day, 1-minute export

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

Docker support is available, but it is optional. `DOCKER.md` now covers:

- CPU/GPU one-shot runs
- the daily production container service
- `.env` configuration
- `systemd timer` deployment for once-per-day execution
- `cron` as a fallback scheduler
