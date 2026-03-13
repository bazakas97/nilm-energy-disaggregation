# SEL API: Daily Data Fetch + NILM Inference

This repo includes a small helper to fetch one-day data from the Smart Energy Lab (SEL) API and run NILM inference.

## 1) Provide credentials (do not commit)

Set credentials via environment variables:

```bash
export SEL_API_EMAIL="you@example.com"
export SEL_API_PASSWORD="your-password"
```

You can also create a local `.env` (see `.env.example`) and export it in your shell before running.

## 2) Fetch one day for one or more participants

```bash
python scripts/fetch_sel_daily.py \
  --date 2026-03-12 \
  --participants certhr5fwl7p,certhckoz1h4 \
  --output-dir DATA/daily_sel_api
```

Output:

- `DATA/daily_sel_api/YYYYMMDD/<participant>.csv`
- `DATA/daily_sel_api/YYYYMMDD/daily_YYYYMMDD_merged.csv` (all participants merged)

Notes:

- The script normalizes columns and exports *power* (default unit: `w`), matching the NILM model expectations.
- Short gaps are interpolated (configurable flags in `--help`). Large missing segments are left as missing and can be filtered downstream.

## 3) Run inference for that day (per-house configs)

```bash
python scripts/run_daily_eval.py \
  --base-config config_release_eval.yaml \
  --date 2026-03-12 \
  --split test \
  --split-data-csv DATA/daily_sel_api/20260312/daily_20260312_merged.csv \
  --per-house \
  --house-overrides house_overrides_daily.example.yaml \
  --run
```

This produces:

- predictions CSV(s) under `results/csv/`
- Plotly HTML plots under `results/plots_*/`

If your daily data does not include appliance ground-truth labels (only `energy_mains` exists), metrics like F1/MAE are not meaningful; focus on predictions + plots.

