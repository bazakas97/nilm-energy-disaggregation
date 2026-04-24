#!/usr/bin/env python3
"""Build chronological train/val/test splits from fetched SEL daily CSVs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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

DEVICE_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in {"timestamp", "participant", "energy_mains"}]

ACTIVE_THRESHOLDS = {
    "energy_dish_washer": 100.0,
    "energy_dryer": 100.0,
    "energy_fridge_freezer": 50.0,
    "energy_pv": 100.0,
    "energy_washing_machine": 100.0,
    "energy_oven": 100.0,
    "energy_ac": 300.0,
    "energy_ev": 500.0,
    "energy_induction_hob": 100.0,
    "energy_ewh": 100.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 60/20/20 NILM splits from DATA/daily_sel_api_* corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        default="DATA/daily_sel_api_full_corpus",
        help="Root containing YYYYMMDD daily directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="DATA/splits_sel_full_rebuilt_60_20_20",
        help="Output split directory.",
    )
    parser.add_argument(
        "--participants",
        default="",
        help="Optional comma-separated participant filter.",
    )
    parser.add_argument(
        "--ratios",
        default="0.6,0.2,0.2",
        help="Train,val,test ratios by participant-day chronological order.",
    )
    parser.add_argument("--min-usable-days", type=int, default=10)
    parser.add_argument("--min-rows-per-day", type=int, default=1000)
    parser.add_argument(
        "--max-final-mains-missing-ratio",
        type=float,
        default=0.0,
        help="Reject CSV days where energy_mains still has NaNs above this ratio.",
    )
    parser.add_argument(
        "--min-mains-nonzero-ratio",
        type=float,
        default=0.03,
        help="Reject days whose mains is effectively zero for almost all rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read/audit corpus but do not write split CSVs.",
    )
    return parser.parse_args()


def normalize_csv_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ratios(text):
    values = [float(x) for x in normalize_csv_list(text)]
    if len(values) != 3:
        raise ValueError("--ratios must contain exactly train,val,test values.")
    total = sum(values)
    if total <= 0:
        raise ValueError("--ratios sum must be > 0.")
    return [v / total for v in values]


def resolve_path(project_dir, value):
    path = Path(value)
    if not path.is_absolute():
        path = project_dir / path
    return path


def load_summary(summary_path):
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def date_from_day_dir(day_dir):
    text = day_dir.name
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def read_participant_day(csv_path, participant, source_date):
    df = pd.read_csv(csv_path)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OUTPUT_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["participant"] = participant
    df["source_date"] = pd.to_datetime(source_date).strftime("%Y-%m-%d")
    return df


def day_quality_ok(df, min_rows, max_final_mains_missing_ratio, min_mains_nonzero_ratio):
    if len(df) < int(min_rows):
        return False, f"rows<{min_rows}"
    mains = pd.to_numeric(df["energy_mains"], errors="coerce")
    missing_ratio = float(mains.isna().mean()) if len(mains) else 1.0
    if missing_ratio > float(max_final_mains_missing_ratio):
        return False, f"mains_missing>{max_final_mains_missing_ratio}"
    mains_filled = mains.fillna(0.0)
    nonzero_ratio = float((mains_filled > 1e-6).mean()) if len(mains_filled) else 0.0
    if nonzero_ratio < float(min_mains_nonzero_ratio):
        return False, f"mains_nonzero<{min_mains_nonzero_ratio}"
    return True, "ok"


def collect_days(corpus_dir, participants_filter, args):
    by_participant = defaultdict(list)
    skipped = []
    for day_dir in sorted(Path(corpus_dir).glob("20??????")):
        if not day_dir.is_dir():
            continue
        source_date = date_from_day_dir(day_dir)
        if pd.isna(source_date):
            continue

        summary = load_summary(day_dir / f"daily_{day_dir.name}_summary.json")
        part_summary = summary.get("summary", {}) if isinstance(summary, dict) else {}

        participant_ids = sorted(part_summary.keys())
        if not participant_ids:
            participant_ids = sorted(p.name.removesuffix(".csv") for p in day_dir.glob("certh*.csv"))
        if participants_filter:
            participant_ids = [p for p in participant_ids if p in participants_filter]

        for participant in participant_ids:
            status = (part_summary.get(participant) or {}).get("status")
            if status and status != "ok":
                skipped.append(
                    {
                        "date": source_date.strftime("%Y-%m-%d"),
                        "participant": participant,
                        "reason": f"summary_status={status}",
                    }
                )
                continue

            csv_path = day_dir / f"{participant}.csv"
            if not csv_path.exists():
                skipped.append(
                    {
                        "date": source_date.strftime("%Y-%m-%d"),
                        "participant": participant,
                        "reason": "missing_csv",
                    }
                )
                continue

            df = read_participant_day(csv_path, participant, source_date)
            ok, reason = day_quality_ok(
                df,
                min_rows=args.min_rows_per_day,
                max_final_mains_missing_ratio=args.max_final_mains_missing_ratio,
                min_mains_nonzero_ratio=args.min_mains_nonzero_ratio,
            )
            if not ok:
                skipped.append(
                    {
                        "date": source_date.strftime("%Y-%m-%d"),
                        "participant": participant,
                        "reason": reason,
                    }
                )
                continue
            by_participant[participant].append(df)
    return by_participant, skipped


def split_dates_for_participant(day_frames, ratios):
    ordered = sorted(day_frames, key=lambda df: df["source_date"].iloc[0])
    n = len(ordered)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n_train < 1 and n > 0:
        n_train = 1
    if n_val < 1 and n > 2:
        n_val = 1
    if n_train + n_val >= n and n > 2:
        n_val = max(1, n - n_train - 1)
    n_test = max(0, n - n_train - n_val)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val : n_train + n_val + n_test],
    }


def compute_split_report(split_frames):
    rows = []
    for split, frames in split_frames.items():
        by_participant = defaultdict(list)
        for df in frames:
            by_participant[str(df["participant"].iloc[0])].append(df)
        for participant, part_frames in sorted(by_participant.items()):
            df = pd.concat(part_frames, ignore_index=True)
            mains = pd.to_numeric(df["energy_mains"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "participant": participant,
                    "split": split,
                    "days": int(df["source_date"].nunique()),
                    "rows": int(len(df)),
                    "mains_nonzero": int((mains > 1e-6).sum()),
                    "mains_nonzero_ratio": float((mains > 1e-6).mean()) if len(df) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def compute_device_availability(split_frames):
    rows = []
    for split, frames in split_frames.items():
        by_participant = defaultdict(list)
        for df in frames:
            by_participant[str(df["participant"].iloc[0])].append(df)
        for participant, part_frames in sorted(by_participant.items()):
            df = pd.concat(part_frames, ignore_index=True)
            for device in DEVICE_COLUMNS:
                values = pd.to_numeric(df[device], errors="coerce")
                known = int(values.notna().sum())
                active = int((values.fillna(0.0) > ACTIVE_THRESHOLDS[device]).sum())
                rows.append(
                    {
                        "split": split,
                        "participant": participant,
                        "device": device,
                        "rows": int(len(df)),
                        "known": known,
                        "known_ratio": float(known / len(df)) if len(df) else 0.0,
                        "active": active,
                        "active_ratio": float(active / len(df)) if len(df) else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def write_split_csvs(split_frames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frames in split_frames.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["participant", "timestamp"]).reset_index(drop=True)
            df = df.drop(columns=["source_date"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df = df[OUTPUT_COLUMNS]
        else:
            df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        df.to_csv(output_dir / f"{split}.csv", index=False)


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    corpus_dir = resolve_path(project_dir, args.corpus_dir)
    output_dir = resolve_path(project_dir, args.output_dir)
    participants_filter = set(normalize_csv_list(args.participants))
    ratios = parse_ratios(args.ratios)

    by_participant, skipped = collect_days(corpus_dir, participants_filter, args)
    kept_participants = {
        p: frames
        for p, frames in sorted(by_participant.items())
        if len(frames) >= int(args.min_usable_days)
    }
    dropped_for_days = [
        {"participant": p, "usable_days": len(frames), "reason": f"usable_days<{args.min_usable_days}"}
        for p, frames in sorted(by_participant.items())
        if len(frames) < int(args.min_usable_days)
    ]

    split_frames = {"train": [], "val": [], "test": []}
    participant_day_counts = {}
    for participant, frames in kept_participants.items():
        participant_day_counts[participant] = len(frames)
        part_split = split_dates_for_participant(frames, ratios)
        for split, split_part_frames in part_split.items():
            split_frames[split].extend(split_part_frames)

    split_report = compute_split_report(split_frames)
    availability = compute_device_availability(split_frames)
    split_rows = {
        split: int(sum(len(df) for df in frames))
        for split, frames in split_frames.items()
    }
    split_days = {
        split: int(sum(df["source_date"].nunique() for df in frames))
        for split, frames in split_frames.items()
    }
    summary = {
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "min_usable_days": int(args.min_usable_days),
        "kept_participants": sorted(kept_participants.keys()),
        "participant_usable_days": participant_day_counts,
        "rows": split_rows,
        "days": split_days,
        "skipped_days_count": len(skipped),
        "dropped_participants": dropped_for_days,
    }

    print("Split build summary:")
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_csvs(split_frames, output_dir)
    split_report.to_csv(output_dir / "split_report_by_participant.csv", index=False)
    availability.to_csv(output_dir / "device_availability_by_split_participant.csv", index=False)
    (output_dir / "split_summary.json").write_text(
        json.dumps({**summary, "skipped_days_sample": skipped[:200]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
