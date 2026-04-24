#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


TOKEN_URL_DEFAULT = "https://services.smartenergylab.com/livingenergy_manager/api/token/"
FETCH_URL_DEFAULT = "https://enershare.smartenergylab.pt/api/fetch-data"

OUTPUT_COLUMNS = [
    "timestamp",
    "participant",
    "energy_dish_washer",
    "energy_dryer",
    "energy_fridge_freezer",
    "energy_pv",
    "energy_washing_machine",
    "energy_mains",
    "energy_oven",
    "energy_ac",
    "energy_ev",
    "energy_induction_hob",
    "energy_ewh",
]

DEVICE_TYPE_TO_COLUMN = {
    "MAIN_METER": "energy_mains",
    "MAINMETER": "energy_mains",
    "METER": "energy_mains",
    "DISH_WASHER": "energy_dish_washer",
    "DISHWASHER": "energy_dish_washer",
    "DISH_WASHING_MACHINE": "energy_dish_washer",
    "DRYER": "energy_dryer",
    "TUMBLE_DRYER": "energy_dryer",
    "FRIDGE_FREEZER": "energy_fridge_freezer",
    "FRIDGE": "energy_fridge_freezer",
    "FREEZER": "energy_fridge_freezer",
    "PV": "energy_pv",
    "WASHING_MACHINE": "energy_washing_machine",
    "WASHER": "energy_washing_machine",
    "OVEN": "energy_oven",
    "AC": "energy_ac",
    "AIR_CONDITIONER": "energy_ac",
    "EV": "energy_ev",
    "EV_CHARGER": "energy_ev",
    "INDUCTION_HOB": "energy_induction_hob",
    "HOB": "energy_induction_hob",
    "CERAMIC_HOB": "energy_induction_hob",
    "CERAMICHOB": "energy_induction_hob",
    "EWH": "energy_ewh",
    "WATER_HEATER": "energy_ewh",
    "ELECTRIC_HEATER": "energy_ewh",
    "HP": "energy_ac",
    "HEAT_PUMP": "energy_ac",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch one-day SEL API data for participants.")
    parser.add_argument("--date", type=str, required=True, help="Target date YYYY-MM-DD.")
    parser.add_argument(
        "--participants",
        type=str,
        required=True,
        help="Comma-separated participant permanent codes.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="SEL account email (or env SEL_API_EMAIL).",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="SEL account password (or env SEL_API_PASSWORD).",
    )
    parser.add_argument("--token-url", type=str, default=TOKEN_URL_DEFAULT)
    parser.add_argument("--fetch-url", type=str, default=FETCH_URL_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="DATA/daily_sel_api",
        help="Output directory for raw + normalized daily files.",
    )
    parser.add_argument(
        "--sampling-minutes",
        type=int,
        default=1,
        help="Expected sampling step in minutes for daily reindexing.",
    )
    parser.add_argument(
        "--energy-unit",
        type=str,
        default="kwh_per_interval",
        choices=["kwh_per_interval", "kw", "w"],
        help="Unit of SEL 'energy' value in payload.",
    )
    parser.add_argument(
        "--output-power-unit",
        type=str,
        default="w",
        choices=["w", "kw"],
        help="Unit for exported power columns (use 'w' for current NILM model).",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.2,
        help="Drop participant/day if mains missing ratio exceeds this value.",
    )
    parser.add_argument(
        "--allowed-final-mains-missing-ratio",
        type=float,
        default=0.0,
        help=(
            "Allow this final mains missing ratio after interpolation and fill remaining "
            "mains gaps. Keep 0.0 for strict evaluation; use a small value for production."
        ),
    )
    parser.add_argument(
        "--interpolate-max-gap-points",
        type=int,
        default=3,
        help="Interpolate only short gaps up to this many points.",
    )
    parser.add_argument(
        "--interpolate-method",
        type=str,
        default="time",
        choices=["time", "linear", "nearest", "slinear", "quadratic"],
        help="Interpolation method for short gaps.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--inter-request-sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between sequential SEL API requests for the same participant.",
    )
    parser.add_argument(
        "--max-api-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts for each SEL API request.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=5.0,
        help="Sleep between failed SEL API attempts.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next participant if one participant fails.",
    )
    return parser.parse_args()


def normalize_participants(text):
    return [p.strip() for p in str(text).split(",") if p.strip()]


def normalize_device_type(device_type):
    text = str(device_type).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text


def map_device_to_column(device_type):
    key = normalize_device_type(device_type)
    return DEVICE_TYPE_TO_COLUMN.get(key)


def request_access_token(session, token_url, email, password, timeout):
    payload = {"email": email, "password": password}
    response = session.post(token_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    access = data.get("access")
    if not access:
        raise RuntimeError("Token response did not include 'access'.")
    return access


def call_sel_api(session, fetch_url, access_token, params, timeout):
    headers = {"access-token": access_token}
    response = session.get(fetch_url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def call_sel_api_with_retry(
    session,
    fetch_url,
    access_token,
    params,
    timeout,
    max_attempts,
    retry_sleep_seconds,
):
    last_exc = None
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return call_sel_api(
                session=session,
                fetch_url=fetch_url,
                access_token=access_token,
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"    request failed (attempt {attempt}/{attempts}): {exc}. "
                f"Retrying in {float(retry_sleep_seconds):.1f}s"
            )
            time.sleep(float(retry_sleep_seconds))
    raise last_exc


def build_day_index(target_date, sampling_minutes):
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1) - timedelta(minutes=sampling_minutes)
    return pd.date_range(start=start, end=end, freq=f"{int(sampling_minutes)}min")


def infer_period_minutes(index, fallback_minutes):
    if len(index) < 2:
        return float(max(1, fallback_minutes))
    diffs = pd.Series(index).sort_values().diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float(max(1, fallback_minutes))
    return float(max(1e-6, diffs.median()))


def convert_energy_to_power(series_energy, period_minutes, energy_unit, output_power_unit):
    values = pd.to_numeric(series_energy, errors="coerce").astype(np.float32)
    if energy_unit == "kwh_per_interval":
        # kWh per sampling-interval -> kW
        power_kw = values / (float(period_minutes) / 60.0)
    elif energy_unit == "kw":
        power_kw = values
    elif energy_unit == "w":
        power_kw = values / 1000.0
    else:
        raise ValueError(f"Unsupported energy_unit: {energy_unit}")

    if output_power_unit == "w":
        power = power_kw * 1000.0
    elif output_power_unit == "kw":
        power = power_kw
    else:
        raise ValueError(f"Unsupported output_power_unit: {output_power_unit}")
    return power.clip(lower=0.0)


def regularize_and_interpolate(
    series,
    full_index,
    max_gap_points,
    method,
):
    aligned = series.reindex(full_index)
    missing_before = int(aligned.isna().sum())
    if int(max_gap_points) > 0:
        filled = aligned.interpolate(
            method=method,
            limit=int(max_gap_points),
            limit_area="inside",
        )
    else:
        filled = aligned.copy()
    missing_after = int(filled.isna().sum())
    return filled, missing_before, missing_after


def fill_remaining_mains_gaps(series):
    return series.ffill().bfill()


def aggregate_device_series(
    rows,
    fallback_sampling_minutes,
    energy_unit,
    output_power_unit,
):
    df_dev = pd.DataFrame(rows)
    if "datetime" not in df_dev.columns or "energy" not in df_dev.columns:
        return None, None

    ts = pd.to_datetime(df_dev["datetime"], errors="coerce")
    energy = pd.to_numeric(df_dev["energy"], errors="coerce")
    valid = (~ts.isna()) & (~energy.isna())
    if not valid.any():
        return None, None

    grouped_energy = energy[valid].groupby(ts[valid]).sum().sort_index()
    period_minutes = infer_period_minutes(grouped_energy.index, fallback_sampling_minutes)
    grouped_power = convert_energy_to_power(
        series_energy=grouped_energy,
        period_minutes=period_minutes,
        energy_unit=energy_unit,
        output_power_unit=output_power_unit,
    )
    grouped_power = grouped_power.astype(np.float32)
    return grouped_power, period_minutes


def flatten_measurement_rows(value):
    """SEL sometimes returns sub-sensor maps instead of a flat rows list."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        rows = []
        for nested_value in value.values():
            rows.extend(flatten_measurement_rows(nested_value))
        return rows
    return []


def build_daily_frame(
    fetch_payload,
    participant,
    target_date,
    sampling_minutes,
    energy_unit,
    output_power_unit,
    max_missing_ratio,
    allowed_final_mains_missing_ratio,
    interpolate_max_gap_points,
    interpolate_method,
):
    data_block = fetch_payload.get("data", {}) or {}
    full_index = build_day_index(target_date=target_date, sampling_minutes=sampling_minutes)
    total_points = len(full_index)

    series_by_col = {}
    period_minutes_by_col = {}
    for device_type, raw_rows in data_block.items():
        column = map_device_to_column(device_type)
        if column is None:
            continue
        rows = flatten_measurement_rows(raw_rows)
        if len(rows) == 0:
            continue

        grouped_power, period_minutes = aggregate_device_series(
            rows=rows,
            fallback_sampling_minutes=sampling_minutes,
            energy_unit=energy_unit,
            output_power_unit=output_power_unit,
        )
        if grouped_power is None:
            continue
        if column in series_by_col:
            series_by_col[column] = series_by_col[column].add(grouped_power, fill_value=0.0)
        else:
            series_by_col[column] = grouped_power
        period_minutes_by_col[column] = period_minutes

    if "energy_mains" not in series_by_col:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {
            "rows": 0,
            "status": "skipped_no_mains",
            "reason": "No MAIN_METER in API payload for selected date.",
        }

    regularized = {}
    missing_stats = {}
    for col_name, raw_series in series_by_col.items():
        filled, miss_before, miss_after = regularize_and_interpolate(
            series=raw_series,
            full_index=full_index,
            max_gap_points=interpolate_max_gap_points,
            method=interpolate_method,
        )
        regularized[col_name] = filled
        missing_stats[col_name] = {
            "missing_before": miss_before,
            "missing_after": miss_after,
            "missing_ratio_before": float(miss_before / total_points) if total_points else 1.0,
            "missing_ratio_after": float(miss_after / total_points) if total_points else 1.0,
            "period_minutes_inferred": period_minutes_by_col.get(col_name),
        }

    mains_missing_before = missing_stats["energy_mains"]["missing_ratio_before"]
    mains_missing_after = missing_stats["energy_mains"]["missing_ratio_after"]
    if mains_missing_before > float(max_missing_ratio):
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {
            "rows": 0,
            "status": "skipped_too_many_mains_gaps",
            "reason": (
                f"Mains missing ratio before interpolation={mains_missing_before:.3f} "
                f"> max_missing_ratio={float(max_missing_ratio):.3f}"
            ),
            "missing": missing_stats,
        }
    filled_remaining_mains_points = 0
    if mains_missing_after > 0.0 and mains_missing_after <= float(allowed_final_mains_missing_ratio):
        mains_before_fill = regularized["energy_mains"]
        filled_remaining_mains_points = int(mains_before_fill.isna().sum())
        regularized["energy_mains"] = fill_remaining_mains_gaps(mains_before_fill)
        missing_stats["energy_mains"]["missing_after_final_fill"] = int(
            regularized["energy_mains"].isna().sum()
        )
        missing_stats["energy_mains"]["filled_remaining_points"] = filled_remaining_mains_points
        mains_missing_after = float(missing_stats["energy_mains"]["missing_after_final_fill"] / total_points) if total_points else 1.0
    if mains_missing_after > 0.0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {
            "rows": 0,
            "status": "skipped_unfilled_mains_gaps",
            "reason": (
                f"Mains still has missing ratio after interpolation={mains_missing_after:.3f}. "
                "Increase --interpolate-max-gap-points or use cleaner day."
            ),
            "missing": missing_stats,
        }

    merged = pd.DataFrame(index=full_index)
    for col_name, series in regularized.items():
        merged[col_name] = series.astype(np.float32)

    merged = merged.reset_index().rename(columns={"index": "timestamp"})
    for col in OUTPUT_COLUMNS:
        if col in {"timestamp", "participant"}:
            continue
        if col not in merged.columns:
            merged[col] = np.nan

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp")
    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged["participant"] = participant
    merged = merged[OUTPUT_COLUMNS].reset_index(drop=True)
    return merged, {
        "rows": int(len(merged)),
        "status": "ok",
        "missing": missing_stats,
        "output_power_unit": output_power_unit,
        "filled_remaining_mains_points": filled_remaining_mains_points,
    }


def main():
    args = parse_args()
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    day_tag = target_date.strftime("%Y%m%d")

    participants = normalize_participants(args.participants)
    if not participants:
        raise ValueError("No participants were provided.")

    email = args.email or os.getenv("SEL_API_EMAIL", "")
    password = args.password or os.getenv("SEL_API_PASSWORD", "")
    if not email or not password:
        raise ValueError("Missing credentials. Use --email/--password or SEL_API_EMAIL/SEL_API_PASSWORD env vars.")

    out_root = os.path.abspath(args.output_dir)
    out_day_dir = os.path.join(out_root, day_tag)
    os.makedirs(out_day_dir, exist_ok=True)

    session = requests.Session()
    access_token = request_access_token(
        session=session,
        token_url=args.token_url,
        email=email,
        password=password,
        timeout=args.timeout,
    )
    print("Authenticated SEL API.")

    merged_frames = []
    summary = {}
    for participant in participants:
        print(f"Fetching participant: {participant}")
        try:
            sensors_payload = call_sel_api_with_retry(
                session=session,
                fetch_url=args.fetch_url,
                access_token=access_token,
                params={
                    "request_type": "get_sensors_list",
                    "participant_permanent_code": participant,
                },
                timeout=args.timeout,
                max_attempts=args.max_api_attempts,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            if float(args.inter_request_sleep_seconds) > 0:
                time.sleep(float(args.inter_request_sleep_seconds))
            fetch_payload = call_sel_api_with_retry(
                session=session,
                fetch_url=args.fetch_url,
                access_token=access_token,
                params={
                    "request_type": "fetch",
                    "participant_permanent_code": participant,
                    "start_date": str(target_date),
                },
                timeout=args.timeout,
                max_attempts=args.max_api_attempts,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )

            sensors_path = os.path.join(out_day_dir, f"{participant}_sensors.json")
            raw_path = os.path.join(out_day_dir, f"{participant}_raw.json")
            with open(sensors_path, "w", encoding="utf-8") as f:
                json.dump(sensors_payload, f, ensure_ascii=False, indent=2)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(fetch_payload, f, ensure_ascii=False, indent=2)

            part_df, stats = build_daily_frame(
                fetch_payload=fetch_payload,
                participant=participant,
                target_date=target_date,
                sampling_minutes=int(args.sampling_minutes),
                energy_unit=args.energy_unit,
                output_power_unit=args.output_power_unit,
                max_missing_ratio=float(args.max_missing_ratio),
                allowed_final_mains_missing_ratio=float(args.allowed_final_mains_missing_ratio),
                interpolate_max_gap_points=int(args.interpolate_max_gap_points),
                interpolate_method=args.interpolate_method,
            )
            part_csv = os.path.join(out_day_dir, f"{participant}.csv")
            part_df.to_csv(part_csv, index=False)
            summary[participant] = {
                **stats,
                "csv": part_csv,
                "raw_json": raw_path,
                "sensors_json": sensors_path,
            }
            if len(part_df):
                merged_frames.append(part_df)
            print(
                f"  status={stats.get('status')} rows={stats.get('rows')} "
                f"unit={stats.get('output_power_unit', args.output_power_unit)}"
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            summary[participant] = {
                "rows": 0,
                "status": "fetch_failed",
                "reason": error_text,
            }
            print(f"  status=fetch_failed rows=0 reason={error_text}")
            if not args.keep_going:
                raise

    merged_csv = os.path.join(out_day_dir, f"daily_{day_tag}_merged.csv")
    if merged_frames:
        merged_df = pd.concat(
            [df.dropna(axis=1, how="all") for df in merged_frames if not df.empty],
            axis=0,
            ignore_index=True,
        )
        for col in OUTPUT_COLUMNS:
            if col not in merged_df.columns:
                merged_df[col] = np.nan
        merged_df = merged_df[OUTPUT_COLUMNS]
        merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], errors="coerce")
        merged_df = merged_df.sort_values(["participant", "timestamp"]).reset_index(drop=True)
        merged_df["timestamp"] = merged_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        merged_df.to_csv(merged_csv, index=False)
    else:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(merged_csv, index=False)

    summary_path = os.path.join(out_day_dir, f"daily_{day_tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": str(target_date),
                "participants": participants,
                "summary": summary,
                "merged_csv": merged_csv,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Merged daily CSV: {merged_csv}")
    print(f"Summary JSON    : {summary_path}")


if __name__ == "__main__":
    main()
