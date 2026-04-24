#!/usr/bin/env python3
"""Fetch a SEL API corpus over a date range.

This is intentionally a thin, resumable wrapper around fetch_sel_daily.py so the
daily production parser and the training-corpus parser stay identical.
"""

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


DEFAULT_PARTICIPANTS = [
    "certh15crq5w",
    "certh5tgbjbh",
    "certh6l51z3m",
    "certh6zpb681",
    "certh7ujyroq",
    "certh7zcqwmc",
    "certh97dgl14",
    "certhckoz1h4",
    "certhh0d0u30",
    "certhr5fwl7p",
    "certhtwo505o",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch all known SEL participants for a date range."
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument(
        "--participants",
        default=",".join(DEFAULT_PARTICIPANTS),
        help="Comma-separated participant ids.",
    )
    parser.add_argument(
        "--participants-file",
        default="",
        help="Optional file with one participant id per line. Overrides --participants.",
    )
    parser.add_argument(
        "--output-dir",
        default="DATA/daily_sel_api_full_corpus",
        help="Output root for daily raw JSON, normalized CSVs, and summaries.",
    )
    parser.add_argument("--sampling-minutes", type=int, default=1)
    parser.add_argument(
        "--energy-unit",
        default="kwh_per_interval",
        choices=["kwh_per_interval", "kw", "w"],
    )
    parser.add_argument("--output-power-unit", default="w", choices=["w", "kw"])
    parser.add_argument("--max-missing-ratio", type=float, default=0.2)
    parser.add_argument("--allowed-final-mains-missing-ratio", type=float, default=0.0)
    parser.add_argument("--interpolate-max-gap-points", type=int, default=3)
    parser.add_argument(
        "--interpolate-method",
        default="time",
        choices=["time", "linear", "nearest", "slinear", "quadratic"],
    )
    parser.add_argument("--inter-request-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-api-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--seconds-per-participant-day",
        type=float,
        default=3.0,
        help="Heuristic for ETA reporting only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch days even if all requested participant raw JSON files exist.",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def normalize_participants(text):
    return [p.strip() for p in str(text).split(",") if p.strip()]


def load_participants(args):
    if args.participants_file:
        path = Path(args.participants_file)
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            values.append(text)
        return values
    return normalize_participants(args.participants)


def day_tag(day):
    return day.strftime("%Y%m%d")


def day_complete(output_dir, day, participants):
    day_dir = Path(output_dir) / day_tag(day)
    if not day_dir.exists():
        return False
    for participant in participants:
        if not (day_dir / f"{participant}_raw.json").exists():
            return False
        if not (day_dir / f"{participant}_sensors.json").exists():
            return False
    return True


def build_daily_cmd(args, project_dir, day, participants):
    cmd = [
        sys.executable,
        "scripts/fetch_sel_daily.py",
        "--date",
        day.isoformat(),
        "--participants",
        ",".join(participants),
        "--output-dir",
        args.output_dir,
        "--sampling-minutes",
        str(args.sampling_minutes),
        "--energy-unit",
        args.energy_unit,
        "--output-power-unit",
        args.output_power_unit,
        "--max-missing-ratio",
        str(args.max_missing_ratio),
        "--allowed-final-mains-missing-ratio",
        str(args.allowed_final_mains_missing_ratio),
        "--interpolate-max-gap-points",
        str(args.interpolate_max_gap_points),
        "--interpolate-method",
        args.interpolate_method,
        "--inter-request-sleep-seconds",
        str(args.inter_request_sleep_seconds),
        "--max-api-attempts",
        str(args.max_api_attempts),
        "--retry-sleep-seconds",
        str(args.retry_sleep_seconds),
        "--keep-going",
    ]
    return cmd


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(project_dir / args.output_dir)

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise ValueError("--end-date must be >= --start-date")

    participants = load_participants(args)
    if not participants:
        raise ValueError("No participants provided.")

    all_days = list(iter_dates(start, end))
    pending_days = [
        day
        for day in all_days
        if args.force or not day_complete(args.output_dir, day, participants)
    ]
    participant_days = len(pending_days) * len(participants)
    eta_seconds = participant_days * float(args.seconds_per_participant_day)

    print("Full SEL corpus fetch plan:")
    print(f"  output_dir       : {args.output_dir}")
    print(f"  date_range       : {start.isoformat()} -> {end.isoformat()}")
    print(f"  total_days       : {len(all_days)}")
    print(f"  pending_days     : {len(pending_days)}")
    print(f"  participants     : {len(participants)}")
    print(f"  participant_days : {participant_days}")
    print(f"  rough_eta_hours  : {eta_seconds / 3600.0:.2f}")

    if args.estimate_only:
        return

    missing_email = not os.getenv("SEL_API_EMAIL")
    missing_password = not os.getenv("SEL_API_PASSWORD")
    if missing_email or missing_password:
        raise ValueError("Missing SEL_API_EMAIL/SEL_API_PASSWORD environment variables.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for i, day in enumerate(pending_days, start=1):
        cmd = build_daily_cmd(args, project_dir, day, participants)
        print(f"[{i}/{len(pending_days)}] {day.isoformat()}")
        print("Running:", " ".join(cmd))
        if not args.print_only:
            subprocess.run(cmd, cwd=str(project_dir), check=True)


if __name__ == "__main__":
    main()
