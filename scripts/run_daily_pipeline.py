#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch one SEL day and run per-house NILM inference."
    )
    parser.add_argument("--date", required=True, help="Target day in YYYY-MM-DD format.")
    parser.add_argument(
        "--participants",
        default="",
        help="Comma-separated participant ids. Required unless --eval-only is used.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/active/release_eval.yaml",
        help="Base evaluation config.",
    )
    parser.add_argument(
        "--house-overrides",
        default="configs/active/house_overrides_daily.example.yaml",
        help="Per-house overrides YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="DATA/daily_sel_api",
        help="Root output dir for fetched daily data.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split name to use in generated evaluation configs.",
    )
    parser.add_argument(
        "--email",
        default="",
        help="SEL email. Optional if SEL_API_EMAIL is already exported.",
    )
    parser.add_argument(
        "--password",
        default="",
        help="SEL password. Optional if SEL_API_PASSWORD is already exported.",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Fetch data only. Do not run evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip fetch and run evaluation on an existing merged CSV.",
    )
    parser.add_argument(
        "--merged-csv",
        default="",
        help="Optional explicit merged CSV path. Useful with --eval-only.",
    )
    parser.add_argument(
        "--sampling-minutes",
        type=int,
        default=1,
        help="Expected SEL sampling step in minutes.",
    )
    parser.add_argument(
        "--energy-unit",
        default="kwh_per_interval",
        choices=["kwh_per_interval", "kw", "w"],
        help="Unit of SEL payload values.",
    )
    parser.add_argument(
        "--output-power-unit",
        default="w",
        choices=["w", "kw"],
        help="Output unit used by the normalized daily CSV.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.2,
        help="Drop house/day if mains missing ratio is above this value.",
    )
    parser.add_argument(
        "--interpolate-max-gap-points",
        type=int,
        default=3,
        help="Interpolate only short gaps up to this many points.",
    )
    parser.add_argument(
        "--interpolate-method",
        default="time",
        choices=["time", "linear", "nearest", "slinear", "quadratic"],
        help="Interpolation method used during daily normalization.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to next participant if one per-house run fails.",
    )
    parser.add_argument(
        "--keep-participant-filter",
        action="store_true",
        help="Keep participant_data_filter enabled during daily evaluation.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the generated commands but do not execute them.",
    )
    return parser.parse_args()


def run_cmd(cmd, cwd, print_only):
    print("Running:", " ".join(cmd))
    if not print_only:
        subprocess.run(cmd, check=True, cwd=cwd)


def main():
    args = parse_args()
    if args.fetch_only and args.eval_only:
        raise SystemExit("--fetch-only and --eval-only are mutually exclusive.")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_bin = sys.executable
    day_tag = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y%m%d")

    merged_csv = args.merged_csv
    if not merged_csv:
        merged_csv = os.path.join(project_dir, args.output_dir, day_tag, f"daily_{day_tag}_merged.csv")
    elif not os.path.isabs(merged_csv):
        merged_csv = os.path.abspath(os.path.join(project_dir, merged_csv))

    if not args.eval_only:
        if not args.participants.strip():
            raise SystemExit("--participants is required unless --eval-only is used.")

        fetch_cmd = [
            python_bin,
            "scripts/fetch_sel_daily.py",
            "--date",
            args.date,
            "--participants",
            args.participants,
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
            "--interpolate-max-gap-points",
            str(args.interpolate_max_gap_points),
            "--interpolate-method",
            args.interpolate_method,
        ]
        if args.email:
            fetch_cmd.extend(["--email", args.email])
        if args.password:
            fetch_cmd.extend(["--password", args.password])
        run_cmd(fetch_cmd, cwd=project_dir, print_only=args.print_only)

    if args.fetch_only:
        return

    eval_cmd = [
        python_bin,
        "scripts/run_daily_eval.py",
        "--base-config",
        args.base_config,
        "--date",
        args.date,
        "--split",
        args.split,
        "--split-data-csv",
        merged_csv,
        "--per-house",
        "--house-overrides",
        args.house_overrides,
        "--run",
    ]
    if args.participants.strip():
        eval_cmd.extend(["--participants", args.participants])
    if args.keep_going:
        eval_cmd.append("--keep-going")
    if args.keep_participant_filter:
        eval_cmd.append("--keep-participant-filter")

    run_cmd(eval_cmd, cwd=project_dir, print_only=args.print_only)


if __name__ == "__main__":
    main()
