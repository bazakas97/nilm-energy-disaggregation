# Docker Run Guide

Docker is optional. The clean production setup is:

- keep the container one-shot
- let the host scheduler (`systemd timer` or `cron`) start it once per day
- mount the repo so fetched data and results stay on the host filesystem

This is more reliable than trying to keep a long-running scheduler alive inside the container.

## 1. Prepare `.env`

```bash
cd /path/to/REEFLEX-NILM
cp .env.example .env
```

Fill at least:

- `SEL_API_EMAIL`
- `SEL_API_PASSWORD`
- `DAILY_PARTICIPANTS`

By default, the daily service will:

- use `configs/active/release_eval.yaml`
- fetch the previous day (`yesterday` in the container timezone)
- write fetched data under `DATA/daily_sel_api/`
- continue to the next participant if one participant fails (`DAILY_KEEP_GOING=1`)

## 2. Build images

CPU:

```bash
docker compose build reeflex-nilm-cpu
```

GPU:

```bash
docker compose build reeflex-nilm-gpu
```

Daily production service (CPU image by default):

```bash
docker compose build reeflex-nilm-daily
```

## 3. Manual runs

### Standard inference

CPU:

```bash
docker compose run --rm reeflex-nilm-cpu --config configs/active/release_eval.yaml
```

GPU:

```bash
docker compose run --rm reeflex-nilm-gpu --config configs/active/release_eval.yaml
```

### Training

CPU:

```bash
docker compose run --rm reeflex-nilm-cpu --config configs/active/train_mains5_all10.yaml
```

GPU:

```bash
docker compose run --rm reeflex-nilm-gpu --config configs/active/train_mains5_all10.yaml
```

Full rebuilt-model training:

```bash
docker compose run --rm reeflex-nilm-gpu --config configs/active/train_sel_full_rebuilt_60_20_20.yaml
```

For a complete fetch -> split build -> training -> model bundle -> evaluation run, use:

```bash
START_DATE=2024-02-01 \
END_DATE=2026-04-15 \
CUDA_VISIBLE_DEVICES=1 \
RUN_DAILY_TEST=1 \
DAILY_TEST_DATE=2026-04-15 \
bash scripts/run_full_rebuild_pipeline.sh
```

The full rebuild wrapper reads `.env`, writes the fetched corpus under `DATA/daily_sel_api_full_corpus/`, writes rebuilt splits under `DATA/splits_sel_full_rebuilt_60_20_20/`, and stores the bundled model under `models/nilmformer_sel_full_rebuilt_60_20_20/`.

### One-shot daily production run

Run for the previous day:

```bash
docker compose run --rm reeflex-nilm-daily
```

Run for an explicit date:

```bash
docker compose run --rm reeflex-nilm-daily 2026-03-15
```

Dry-run without hitting SEL API:

```bash
DAILY_PRINT_ONLY=1 docker compose run --rm reeflex-nilm-daily 2026-03-15
```

The daily service calls `scripts/run_daily_production.sh`, which wraps `scripts/run_daily_pipeline.py` with env-driven defaults.
By default it also runs `scripts/sel_doctor.py` first. You can disable that with `DAILY_RUN_DOCTOR=0`.

## 4. Outputs

The project is mounted into the container as `/workspace`, so outputs are written back to the host repository:

- fetched daily data: `DATA/daily_sel_api/YYYYMMDD/`
- generated daily configs: `results/generated_configs/daily_eval_<split>_YYYYMMDD/`
- predictions and metrics: `results/csv/`
- plots: `results/plots_*/YYYYMMDD/<participant>/`

## 5. Run once per day with systemd

Example deploy files are already included:

- `deploy/systemd/reeflex-nilm-daily.service`
- `deploy/systemd/reeflex-nilm-daily.timer`

They assume the repo lives at:

- `/opt/reeflex-nilm`

If you deploy somewhere else, edit `WorkingDirectory=` first.

Install:

```bash
sudo cp deploy/systemd/reeflex-nilm-daily.service /etc/systemd/system/
sudo cp deploy/systemd/reeflex-nilm-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now reeflex-nilm-daily.timer
```

Check schedule:

```bash
systemctl list-timers reeflex-nilm-daily.timer
```

Follow logs:

```bash
journalctl -u reeflex-nilm-daily.service -f
```

Current timer schedule:

- every day at `02:30` local time

## 6. Cron alternative

If you do not want systemd, use:

- `deploy/cron/reeflex-nilm-daily.cron`

Example install:

```bash
sudo cp deploy/cron/reeflex-nilm-daily.cron /etc/cron.d/reeflex-nilm-daily
sudo chmod 644 /etc/cron.d/reeflex-nilm-daily
```

This also assumes the repo is deployed at `/opt/reeflex-nilm`.
