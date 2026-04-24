#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
  while IFS= read -r env_line || [ -n "$env_line" ]; do
    case "$env_line" in
      ''|'#'*)
        continue
        ;;
    esac
    env_key=${env_line%%=*}
    env_value=${env_line#*=}
    if [ -z "$env_key" ] || [ "$env_key" = "$env_line" ]; then
      continue
    fi
    eval "env_is_set=\${$env_key+x}"
    if [ -z "$env_is_set" ]; then
      export "$env_key=$env_value"
    fi
  done < "$PROJECT_DIR/.env"
fi

require_var() {
  var_name="$1"
  eval "var_value=\${$var_name:-}"
  if [ -z "$var_value" ]; then
    echo "Missing required environment variable: $var_name" >&2
    exit 1
  fi
}

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

require_var SEL_API_EMAIL
require_var SEL_API_PASSWORD
require_var DAILY_PARTICIPANTS

TARGET_DATE="${1:-${DAILY_TARGET_DATE:-}}"
if [ -z "$TARGET_DATE" ]; then
  TARGET_DATE=$(date -d 'yesterday' +%F)
fi

PYTHON_BIN="${DAILY_PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python3)
else
  echo "Neither python nor python3 is available in PATH." >&2
  exit 1
fi

BASE_CONFIG="${DAILY_BASE_CONFIG:-configs/active/release_eval.yaml}"
HOUSE_OVERRIDES="${DAILY_HOUSE_OVERRIDES:-configs/active/house_overrides_daily.example.yaml}"
OUTPUT_DIR="${DAILY_OUTPUT_DIR:-DATA/daily_sel_api}"
SPLIT_NAME="${DAILY_SPLIT:-test}"
SAMPLING_MINUTES="${DAILY_SAMPLING_MINUTES:-1}"
ENERGY_UNIT="${DAILY_ENERGY_UNIT:-kwh_per_interval}"
OUTPUT_POWER_UNIT="${DAILY_OUTPUT_POWER_UNIT:-w}"
MAX_MISSING_RATIO="${DAILY_MAX_MISSING_RATIO:-0.2}"
ALLOWED_FINAL_MAINS_MISSING_RATIO="${DAILY_ALLOWED_FINAL_MAINS_MISSING_RATIO:-0.0}"
INTERPOLATE_MAX_GAP_POINTS="${DAILY_INTERPOLATE_MAX_GAP_POINTS:-3}"
INTERPOLATE_METHOD="${DAILY_INTERPOLATE_METHOD:-time}"
RUN_DOCTOR="${DAILY_RUN_DOCTOR:-1}"
DOCTOR_PARTICIPANT="${DAILY_DOCTOR_PARTICIPANT:-}"

if is_truthy "$RUN_DOCTOR"; then
  if [ -z "$DOCTOR_PARTICIPANT" ]; then
    DOCTOR_PARTICIPANT=${DAILY_PARTICIPANTS%%,*}
  fi
  echo "[daily-production] running SEL doctor"
  "$PYTHON_BIN" scripts/sel_doctor.py \
    --date "$TARGET_DATE" \
    --participant "$DOCTOR_PARTICIPANT" \
    --base-config "$BASE_CONFIG"
fi

set -- \
  "$PYTHON_BIN" scripts/run_daily_pipeline.py \
  --date "$TARGET_DATE" \
  --participants "$DAILY_PARTICIPANTS" \
  --base-config "$BASE_CONFIG" \
  --house-overrides "$HOUSE_OVERRIDES" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT_NAME" \
  --sampling-minutes "$SAMPLING_MINUTES" \
  --energy-unit "$ENERGY_UNIT" \
  --output-power-unit "$OUTPUT_POWER_UNIT" \
  --max-missing-ratio "$MAX_MISSING_RATIO" \
  --allowed-final-mains-missing-ratio "$ALLOWED_FINAL_MAINS_MISSING_RATIO" \
  --interpolate-max-gap-points "$INTERPOLATE_MAX_GAP_POINTS" \
  --interpolate-method "$INTERPOLATE_METHOD"

if is_truthy "${DAILY_KEEP_GOING:-1}"; then
  set -- "$@" --keep-going
fi

if is_truthy "${DAILY_KEEP_PARTICIPANT_FILTER:-0}"; then
  set -- "$@" --keep-participant-filter
fi

if is_truthy "${DAILY_FETCH_ONLY:-0}"; then
  set -- "$@" --fetch-only
fi

if is_truthy "${DAILY_EVAL_ONLY:-0}"; then
  set -- "$@" --eval-only
fi

if [ -n "${DAILY_MERGED_CSV:-}" ]; then
  set -- "$@" --merged-csv "$DAILY_MERGED_CSV"
fi

if is_truthy "${DAILY_PRINT_ONLY:-0}"; then
  set -- "$@" --print-only
fi

echo "[daily-production] target_date=$TARGET_DATE participants=$DAILY_PARTICIPANTS"
echo "[daily-production] command: $*"
exec "$@"
