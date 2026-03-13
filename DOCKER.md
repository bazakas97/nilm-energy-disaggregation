# Docker Run Guide

This project is Docker-ready for both CPU and GPU runs.

## 1) Build images

From the project root (`NILMvPaper`):

```bash
docker compose build nilm-cpu
docker compose build nilm-gpu
```

`nilm-gpu` uses CUDA PyTorch wheels (`cu121`) and requires NVIDIA Container Toolkit on the host.

## 2) Run training

CPU:

```bash
docker compose run --rm nilm-cpu --config config_all_houses_all6_evgapfill.yaml
```

GPU:

```bash
docker compose run --rm nilm-gpu --config config_all_houses_all6_evgapfill.yaml
```

## 3) Run evaluation

CPU:

```bash
docker compose run --rm nilm-cpu --config config_all_houses_all6_evgapfill_eval.yaml
```

GPU:

```bash
docker compose run --rm nilm-gpu --config config_all_houses_all6_evgapfill_eval.yaml
```

## 4) Custom config

Use any config by changing the final argument:

```bash
docker compose run --rm nilm-gpu --config config_single_house_certhr5fwl7p_all6_evgapfill_eval.yaml
```

## 5) Notes

- Project folder is mounted into the container as `/workspace`.
- Outputs are written back to host in `results/`.
- If GPU is not available on server, use `nilm-cpu`.

## 6) Daily Evaluation (One Day, All Houses)

Generate one config for one day (all houses together) and run:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --run
```

Only specific houses:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --participants certhr5fwl7p,certhckoz1h4 \
  --run
```

## 7) Daily Evaluation Per House (Recommended)

Generate one config per participant (house), so each house can have its own overrides:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --per-house \
  --run
```

Note: daily script disables `data.preprocessing.participant_data_filter` by default (because a single day often has fewer rows than `min_rows`). Use `--keep-participant-filter` only if you explicitly want strict filtering.

With per-house preprocessing/postprocessing overrides:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --per-house \
  --house-overrides house_overrides_daily.example.yaml \
  --run
```

Only selected houses:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --per-house \
  --participants certhr5fwl7p,certhckoz1h4 \
  --run
```

## 8) Daily run on CPU server

CPU-only server is supported. Use `nilm-cpu` service:

```bash
docker compose run --rm --entrypoint python nilm-cpu \
  scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date yesterday \
  --per-house \
  --run
```

## 9) Optional cron scheduling

Example cron (runs every day at 02:10):

```cron
10 2 * * * cd /path/to/NILMvPaper && docker compose run --rm --entrypoint python nilm-cpu scripts/run_daily_eval.py --base-config config_all_houses_all6_evgapfill_eval.yaml --date yesterday --per-house --run >> results/logs/daily_eval.log 2>&1
```

Inside Docker (CPU):

```bash
docker compose run --rm --entrypoint python nilm-cpu \
  scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --run
```

## 10) Optional SEL API Daily Pull

Fetch one day of data per house from SEL API (stored under `DATA/daily_sel_api/YYYYMMDD/`):

```bash
SEL_API_EMAIL='your_email' SEL_API_PASSWORD='your_password' \
python scripts/fetch_sel_daily.py \
  --date 2025-10-10 \
  --participants certhr5fwl7p,certhckoz1h4
```

This step is independent from model evaluation; it prepares daily raw/normalized files.

Default behavior in `fetch_sel_daily.py`:
- Converts API energy to power (`kwh_per_interval` -> power, default output unit `w`).
- Reindexes to a fixed 1-minute grid.
- Interpolates only short gaps (`--interpolate-max-gap-points`).
- Drops house/day if mains has too many gaps (`--max-missing-ratio`).

If daily CSV has no ground-truth appliance labels, use overrides to set:
- `data.drop_unlabeled_centers: false`
- `evaluate.mask_unknown_for_plots: false`
- `data.preprocessing.unattributed_mains_mask.drop_rows: false`

Then run prediction on that fetched CSV:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --split-data-csv DATA/daily_sel_api/20251010/daily_20251010_merged.csv \
  --per-house \
  --house-overrides house_overrides_daily.example.yaml \
  --run
```

For stricter peak denoising + physical mains budget constraint:

```bash
python scripts/run_daily_eval.py \
  --base-config config_all_houses_all6_evgapfill_eval.yaml \
  --date 2025-10-10 \
  --split-data-csv DATA/daily_sel_api/20251010/daily_20251010_merged.csv \
  --per-house \
  --house-overrides house_overrides_daily_denoise.yaml \
  --run
```
