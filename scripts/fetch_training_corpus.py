#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch all participant/day data required by the training split CSVs."
    )
    parser.add_argument(
        "--split-dir",
        default="DATA/splits_nilm_mains5_rebuilt_60_20_20",
        help="Directory containing train.csv / val.csv / test.csv.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names to scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="DATA/daily_sel_api_training_corpus",
        help="Root directory where fetched daily files will be stored.",
    )
    parser.add_argument("--email", default="", help="SEL email. Optional if SEL_API_EMAIL is exported.")
    parser.add_argument(
        "--password",
        default="",
        help="SEL password. Optional if SEL_API_PASSWORD is exported.",
    )
    parser.add_argument(
        "--seconds-per-participant-day",
        type=float,
        default=3.0,
        help="Heuristic used only for rough ETA reporting.",
    )
    parser.add_argument(
        "--inter-request-sleep-seconds",
        type=float,
        default=1.0,
        help="Passed through to fetch_sel_daily.py.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only print counts and ETA. Do not fetch anything.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the generated fetch commands but do not execute them.",
    )
    return parser.parse_args()


def normalize_splits(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def collect_date_participants(csv_path):
    grouped = defaultdict(set)
    for chunk in pd.read_csv(csv_path, usecols=["timestamp", "participant"], chunksize=200_000):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "participant"])
        chunk["date"] = chunk["timestamp"].dt.strftime("%Y-%m-%d")
        for date_value, participants in chunk.groupby("date")["participant"]:
            grouped[str(date_value)].update(str(p) for p in participants.dropna().astype(str).unique().tolist())
    return grouped


def build_plan(split_dir, splits):
    grouped = defaultdict(set)
    split_counts = {}
    for split in splits:
        csv_path = Path(split_dir) / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        split_grouped = collect_date_participants(csv_path)
        split_counts[split] = {
            "dates": len(split_grouped),
            "participant_days": sum(len(v) for v in split_grouped.values()),
        }
        for date_value, participants in split_grouped.items():
            grouped[date_value].update(participants)
    return grouped, split_counts


def estimate_runtime_seconds(grouped, seconds_per_participant_day):
    participant_days = sum(len(v) for v in grouped.values())
    return float(participant_days) * float(seconds_per_participant_day)


def run_cmd(cmd, cwd, print_only):
    print("Running:", " ".join(cmd))
    if not print_only:
        subprocess.run(cmd, check=True, cwd=cwd)


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    split_dir = (project_dir / args.split_dir).resolve() if not os.path.isabs(args.split_dir) else Path(args.split_dir)
    output_dir = str((project_dir / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir))
    splits = normalize_splits(args.splits)
    if not splits:
        raise ValueError("No splits provided.")

    grouped, split_counts = build_plan(split_dir=split_dir, splits=splits)
    unique_days = sorted(grouped.keys())
    participant_days = sum(len(v) for v in grouped.values())
    participants = sorted({p for values in grouped.values() for p in values})
    eta_seconds = estimate_runtime_seconds(grouped, args.seconds_per_participant_day)

    print("Training fetch plan:")
    print(f"  split_dir        : {split_dir}")
    print(f"  splits           : {splits}")
    print(f"  unique_days      : {len(unique_days)}")
    print(f"  participants     : {len(participants)}")
    print(f"  participant_days : {participant_days}")
    print(f"  rough_eta_min    : {eta_seconds / 60.0:.1f}")
    print(f"  rough_eta_hours  : {eta_seconds / 3600.0:.2f}")
    print("  per_split        :", json.dumps(split_counts, indent=2))

    if args.estimate_only:
        return

    python_bin = sys.executable
    email = args.email or os.getenv("SEL_API_EMAIL", "")
    password = args.password or os.getenv("SEL_API_PASSWORD", "")
    if not email or not password:
        raise ValueError("Missing credentials. Use --email/--password or SEL_API_EMAIL/SEL_API_PASSWORD env vars.")

    for date_value in unique_days:
        participants_csv = ",".join(sorted(grouped[date_value]))
        cmd = [
            python_bin,
            "scripts/fetch_sel_daily.py",
            "--date",
            date_value,
            "--participants",
            participants_csv,
            "--output-dir",
            output_dir,
            "--inter-request-sleep-seconds",
            str(args.inter_request_sleep_seconds),
        ]
        if args.email:
            cmd.extend(["--email", args.email])
        if args.password:
            cmd.extend(["--password", args.password])
        run_cmd(cmd, cwd=str(project_dir), print_only=args.print_only)


if __name__ == "__main__":
    main()
