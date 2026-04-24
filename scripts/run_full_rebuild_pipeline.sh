#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"

START_DATE="${START_DATE:-2024-02-01}"
END_DATE="${END_DATE:-$(date -I)}"
CORPUS_DIR="${CORPUS_DIR:-DATA/daily_sel_api_full_corpus}"
SPLIT_DIR="${SPLIT_DIR:-DATA/splits_sel_full_rebuilt_60_20_20}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/active/train_sel_full_rebuilt_60_20_20.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/active/release_eval_sel_full_rebuilt.yaml}"
MODEL_NAME="${MODEL_NAME:-nilmformer_sel_full_rebuilt_60_20_20}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_FETCH="${RUN_FETCH:-1}"
RUN_BUILD_SPLITS="${RUN_BUILD_SPLITS:-1}"
RUN_DOCKER_BUILD="${RUN_DOCKER_BUILD:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_BUNDLE="${RUN_BUNDLE:-1}"
RUN_EVALUATE="${RUN_EVALUATE:-1}"
RUN_DAILY_TEST="${RUN_DAILY_TEST:-0}"
DAILY_TEST_DATE="${DAILY_TEST_DATE:-$END_DATE}"

if [[ -z "${SEL_API_EMAIL:-}" || -z "${SEL_API_PASSWORD:-}" ]]; then
  echo "Missing SEL_API_EMAIL/SEL_API_PASSWORD. Fill .env or export them first." >&2
  exit 1
fi

echo "Full rebuild pipeline"
echo "  START_DATE=$START_DATE"
echo "  END_DATE=$END_DATE"
echo "  CORPUS_DIR=$CORPUS_DIR"
echo "  SPLIT_DIR=$SPLIT_DIR"
echo "  TRAIN_CONFIG=$TRAIN_CONFIG"
echo "  EVAL_CONFIG=$EVAL_CONFIG"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  LOCAL_UID=$LOCAL_UID"
echo "  LOCAL_GID=$LOCAL_GID"

if [[ "$RUN_FETCH" == "1" ]]; then
  python scripts/fetch_sel_full_corpus.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output-dir "$CORPUS_DIR" \
    --max-missing-ratio "${FULL_FETCH_MAX_MISSING_RATIO:-0.2}" \
    --allowed-final-mains-missing-ratio "${FULL_FETCH_ALLOWED_FINAL_MAINS_MISSING_RATIO:-0.0}" \
    --interpolate-max-gap-points "${FULL_FETCH_INTERPOLATE_MAX_GAP_POINTS:-3}" \
    --interpolate-method "${FULL_FETCH_INTERPOLATE_METHOD:-time}" \
    --inter-request-sleep-seconds "${FULL_FETCH_INTER_REQUEST_SLEEP_SECONDS:-1.0}" \
    --max-api-attempts "${FULL_FETCH_MAX_API_ATTEMPTS:-3}" \
    --retry-sleep-seconds "${FULL_FETCH_RETRY_SLEEP_SECONDS:-5.0}"
fi

if [[ "$RUN_BUILD_SPLITS" == "1" ]]; then
  python scripts/build_splits_from_daily_corpus.py \
    --corpus-dir "$CORPUS_DIR" \
    --output-dir "$SPLIT_DIR" \
    --ratios "${SPLIT_RATIOS:-0.6,0.2,0.2}" \
    --min-usable-days "${MIN_USABLE_DAYS:-10}" \
    --min-rows-per-day "${MIN_ROWS_PER_DAY:-1000}" \
    --max-final-mains-missing-ratio "${MAX_FINAL_MAINS_MISSING_RATIO:-0.0}" \
    --min-mains-nonzero-ratio "${MIN_MAINS_NONZERO_RATIO:-0.03}"
fi

if [[ "$RUN_DOCKER_BUILD" == "1" ]]; then
  docker compose build reeflex-nilm-gpu
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" docker compose run --rm -T reeflex-nilm-gpu \
    --config "$TRAIN_CONFIG"
fi

if [[ "$RUN_BUNDLE" == "1" ]]; then
  mkdir -p "models/$MODEL_NAME"
  cp "results/models/${MODEL_NAME}_best_model.pth" "models/$MODEL_NAME/best_model.pth"
  cp "results/models/${MODEL_NAME}_best_model_meta.json" "models/$MODEL_NAME/meta.json"
  cp "results/models/${MODEL_NAME}_input_scaler.save" "models/$MODEL_NAME/input_scaler.save"
  cp "results/models/${MODEL_NAME}_output_scaler.save" "models/$MODEL_NAME/output_scaler.save"
fi

if [[ "$RUN_EVALUATE" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" docker compose run --rm -T reeflex-nilm-gpu \
    --config "$EVAL_CONFIG"
fi

if [[ "$RUN_DAILY_TEST" == "1" ]]; then
  DAILY_BASE_CONFIG="$EVAL_CONFIG" docker compose run --rm -T reeflex-nilm-daily "$DAILY_TEST_DATE"
fi

echo "Full rebuild pipeline finished."
