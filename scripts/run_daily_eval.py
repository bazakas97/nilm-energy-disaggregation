#!/usr/bin/env python3
import argparse
import copy
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and optionally run daily NILM evaluation configs.")
    parser.add_argument(
        "--base-config",
        type=str,
        default="config_release_eval.yaml",
        help="Base evaluation YAML config.",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Target day in YYYY-MM-DD (or 'today' / 'yesterday').",
    )
    parser.add_argument(
        "--participants",
        type=str,
        default="",
        help="Comma-separated participant codes. Empty = auto-discover from split CSV.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Override evaluation split name (train/val/test). Default: from base config.",
    )
    parser.add_argument(
        "--split-data-csv",
        type=str,
        default="",
        help="Optional CSV path to override paths.<split>_data for this daily run.",
    )
    parser.add_argument(
        "--per-house",
        action="store_true",
        help="Generate one config per house (participant).",
    )
    parser.add_argument(
        "--house-overrides",
        type=str,
        default="",
        help=(
            "Optional YAML with per-house overrides. Format: "
            "{defaults: {...}, houses: {participant: {...}}}"
        ),
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="",
        help="Output config path for single-run mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output config directory for --per-house mode.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="If set, execute run.py with generated config(s).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="In --run mode, continue with next participant if one run fails.",
    )
    parser.add_argument(
        "--keep-participant-filter",
        action="store_true",
        help=(
            "Keep data.preprocessing.participant_data_filter as-is. "
            "Default behavior disables it for daily windows."
        ),
    )
    return parser.parse_args()


def parse_target_date(text):
    value = str(text).strip().lower()
    if value == "today":
        return datetime.now().date()
    if value == "yesterday":
        return (datetime.now() - timedelta(days=1)).date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_day_bounds(day_obj):
    day = day_obj.strftime("%Y-%m-%d")
    return f"{day} 00:00:00", f"{day} 23:59:59"


def normalize_participants(participants_text):
    if not str(participants_text).strip():
        return []
    return [p.strip() for p in str(participants_text).split(",") if p.strip()]


def sanitize_tag(value):
    text = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return safe.strip("._") or "unknown"


def append_tags_to_file(path, tags):
    root, ext = os.path.splitext(path)
    suffix = "_".join(sanitize_tag(t) for t in tags if str(t).strip())
    return f"{root}_{suffix}{ext}" if suffix else path


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_house_profiles(path):
    if not path:
        return {"defaults": {}, "houses": {}}
    data = load_yaml(path)
    return {
        "defaults": data.get("defaults", {}) or {},
        "houses": data.get("houses", data.get("participants", {})) or {},
    }


def resolve_path(base_dir, path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.normpath(os.path.join(base_dir, path))


def discover_participants_from_split_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=["participant"])
    values = sorted(df["participant"].astype(str).dropna().unique().tolist())
    return [v for v in values if str(v).strip()]


def set_day_filter(cfg, split_name, start, end):
    data_cfg = cfg.setdefault("data", {})
    date_ranges = data_cfg.get("date_range_by_split") or {}
    date_ranges[split_name] = {"start": start, "end": end}
    data_cfg["date_range_by_split"] = date_ranges


def set_split_participants(cfg, split_name, participants):
    data_cfg = cfg.setdefault("data", {})
    data_cfg[f"participants_{split_name}"] = list(participants)
    split_map = data_cfg.get("participants_by_split")
    if isinstance(split_map, dict):
        split_map = copy.deepcopy(split_map)
    else:
        split_map = {}
    split_map[split_name] = list(participants)
    data_cfg["participants_by_split"] = split_map
    data_cfg["participants"] = []


def set_split_data_path(cfg, split_name, csv_path):
    paths_cfg = cfg.setdefault("paths", {})
    paths_cfg[f"{split_name}_data"] = csv_path


def apply_output_tags(cfg, day_tag, participant=None):
    paths_cfg = cfg.setdefault("paths", {})
    tags = [day_tag]
    if participant is not None:
        tags.insert(0, participant)

    if paths_cfg.get("predictions_csv"):
        paths_cfg["predictions_csv"] = append_tags_to_file(paths_cfg["predictions_csv"], tags)
    if paths_cfg.get("metrics_json"):
        paths_cfg["metrics_json"] = append_tags_to_file(paths_cfg["metrics_json"], tags)
    if paths_cfg.get("plots_dir"):
        if participant is None:
            paths_cfg["plots_dir"] = os.path.join(paths_cfg["plots_dir"], day_tag)
        else:
            paths_cfg["plots_dir"] = os.path.join(paths_cfg["plots_dir"], day_tag, sanitize_tag(participant))


def disable_participant_filter_for_daily(cfg):
    data_cfg = cfg.setdefault("data", {})
    preprocessing = data_cfg.setdefault("preprocessing", {})
    participant_filter_cfg = preprocessing.get("participant_data_filter")
    if isinstance(participant_filter_cfg, dict):
        participant_filter_cfg = copy.deepcopy(participant_filter_cfg)
    else:
        participant_filter_cfg = {}
    participant_filter_cfg["enabled"] = False
    preprocessing["participant_data_filter"] = participant_filter_cfg


def absolutize_paths(cfg, project_dir):
    paths_cfg = cfg.setdefault("paths", {})
    for key, value in list(paths_cfg.items()):
        if isinstance(value, str) and value and not os.path.isabs(value):
            paths_cfg[key] = os.path.normpath(os.path.join(project_dir, value))


def write_config(path, cfg):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_config(project_dir, config_path):
    cmd = [sys.executable, "run.py", "--config", config_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=project_dir)


def main():
    args = parse_args()
    base_config_path = os.path.abspath(args.base_config)
    project_dir = os.path.dirname(base_config_path)

    base_cfg = load_yaml(base_config_path)
    base_cfg["action"] = "evaluate"
    eval_cfg = base_cfg.setdefault("evaluate", {})
    paths_cfg = base_cfg.setdefault("paths", {})

    split_name = str(args.split or eval_cfg.get("split_name", "test")).strip().lower()
    eval_cfg["split_name"] = split_name

    target_day = parse_target_date(args.date)
    start, end = build_day_bounds(target_day)
    day_tag = target_day.strftime("%Y%m%d")

    house_profiles = load_house_profiles(resolve_path(project_dir, args.house_overrides) if args.house_overrides else "")
    participant_filter_disabled = not args.keep_participant_filter

    if args.per_house:
        if args.output_config:
            print("[WARN] --output-config is ignored in --per-house mode. Use --output-dir.")

        if args.output_dir:
            output_dir = resolve_path(project_dir, args.output_dir)
        else:
            output_dir = os.path.join(
                project_dir,
                "results",
                "generated_configs",
                f"daily_eval_{split_name}_{day_tag}",
            )
        os.makedirs(output_dir, exist_ok=True)

        participants = normalize_participants(args.participants)
        if not participants:
            split_path_key = f"{split_name}_data"
            split_csv_path = resolve_path(project_dir, args.split_data_csv) if args.split_data_csv else paths_cfg.get(split_path_key)
            if not split_csv_path:
                raise ValueError(f"Cannot auto-discover participants: missing paths.{split_path_key} in base config.")
            split_csv_path = resolve_path(project_dir, split_csv_path)
            participants = discover_participants_from_split_csv(split_csv_path)
        if not participants:
            raise ValueError("No participants found for per-house daily run.")

        generated_configs = []
        for participant in participants:
            cfg = copy.deepcopy(base_cfg)
            cfg = deep_merge(cfg, house_profiles.get("defaults", {}))
            cfg = deep_merge(cfg, house_profiles.get("houses", {}).get(participant, {}))
            cfg["action"] = "evaluate"

            set_day_filter(cfg, split_name, start, end)
            set_split_participants(cfg, split_name, [participant])
            if args.split_data_csv:
                set_split_data_path(cfg, split_name, resolve_path(project_dir, args.split_data_csv))
            apply_output_tags(cfg, day_tag=day_tag, participant=participant)
            if not args.keep_participant_filter:
                disable_participant_filter_for_daily(cfg)
            absolutize_paths(cfg, project_dir)

            out_cfg_path = os.path.join(output_dir, f"{sanitize_tag(participant)}.yaml")
            write_config(out_cfg_path, cfg)
            generated_configs.append((participant, out_cfg_path))
            print(f"Generated config [{participant}]: {out_cfg_path}")

        print(f"Day range: {start} -> {end}")
        print(f"Participants: {participants}")
        if participant_filter_disabled:
            print("Daily mode: participant_data_filter is disabled.")

        if not args.run:
            print("Dry mode only. Add --run to execute evaluation.")
            return

        failures = []
        for participant, cfg_path in generated_configs:
            try:
                run_config(project_dir, cfg_path)
            except subprocess.CalledProcessError as exc:
                failures.append((participant, exc.returncode))
                print(f"[ERROR] Participant {participant} failed with code {exc.returncode}")
                if not args.keep_going:
                    break

        if failures:
            print("Run completed with failures:", failures)
            raise SystemExit(1)
        print("Run completed successfully for all participants.")
        return

    cfg = copy.deepcopy(base_cfg)
    cfg = deep_merge(cfg, house_profiles.get("defaults", {}))
    cfg["action"] = "evaluate"
    set_day_filter(cfg, split_name, start, end)

    participants = normalize_participants(args.participants)
    if participants:
        set_split_participants(cfg, split_name, participants)
    if args.split_data_csv:
        set_split_data_path(cfg, split_name, resolve_path(project_dir, args.split_data_csv))

    apply_output_tags(cfg, day_tag=day_tag, participant=None)
    if not args.keep_participant_filter:
        disable_participant_filter_for_daily(cfg)
    absolutize_paths(cfg, project_dir)

    if args.output_config:
        output_config_path = resolve_path(project_dir, args.output_config)
    else:
        output_config_path = os.path.join(
            project_dir,
            "results",
            "generated_configs",
            f"daily_eval_{split_name}_{day_tag}.yaml",
        )
    write_config(output_config_path, cfg)

    print(f"Generated config: {output_config_path}")
    print(f"Day range: {start} -> {end}")
    if participants:
        print(f"Participants: {participants}")
    else:
        print("Participants: from base config (all/default)")
    if participant_filter_disabled:
        print("Daily mode: participant_data_filter is disabled.")

    if not args.run:
        print("Dry mode only. Add --run to execute evaluation.")
        return

    run_config(project_dir, output_config_path)


if __name__ == "__main__":
    main()
