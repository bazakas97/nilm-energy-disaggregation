# Docker Run Guide

Docker support is optional. Use it only if you want a reproducible CPU/GPU runtime on another machine.

## Build images

```bash
docker compose build reeflex-nilm-cpu
docker compose build reeflex-nilm-gpu
```

## Run inference

CPU:

```bash
docker compose run --rm reeflex-nilm-cpu --config configs/active/release_eval.yaml
```

GPU:

```bash
docker compose run --rm reeflex-nilm-gpu --config configs/active/release_eval.yaml
```

## Run training

CPU:

```bash
docker compose run --rm reeflex-nilm-cpu --config configs/active/train_mains5_all10.yaml
```

GPU:

```bash
docker compose run --rm reeflex-nilm-gpu --config configs/active/train_mains5_all10.yaml
```

## Daily inference

```bash
docker compose run --rm --entrypoint python reeflex-nilm-cpu \
  scripts/run_daily_eval.py \
  --base-config configs/active/release_eval.yaml \
  --date yesterday \
  --per-house \
  --house-overrides configs/active/house_overrides_daily.example.yaml \
  --run
```

The project is mounted into the container as `/workspace`, so outputs are written back to the local `results/` directory.

