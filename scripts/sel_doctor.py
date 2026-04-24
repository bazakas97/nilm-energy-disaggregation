#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import requests
import yaml


TOKEN_URL_DEFAULT = "https://services.smartenergylab.com/livingenergy_manager/api/token/"
FETCH_URL_DEFAULT = "https://enershare.smartenergylab.pt/api/fetch-data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight checks for the SEL-backed daily NILM pipeline."
    )
    parser.add_argument(
        "--date",
        default="",
        help="Target day in YYYY-MM-DD format. Defaults to yesterday if a fetch probe is requested.",
    )
    parser.add_argument(
        "--participant",
        default="",
        help="Participant permanent code for a real SEL fetch probe.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/active/release_eval.yaml",
        help="Evaluation config to validate.",
    )
    parser.add_argument(
        "--model-dir",
        default="",
        help="Optional explicit model bundle directory. Auto-detected from the config when omitted.",
    )
    parser.add_argument(
        "--token-url",
        default=TOKEN_URL_DEFAULT,
        help="SEL token endpoint.",
    )
    parser.add_argument(
        "--fetch-url",
        default=FETCH_URL_DEFAULT,
        help="SEL fetch endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Only run local filesystem/config/env checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary in addition to the human summary.",
    )
    return parser.parse_args()


def resolve_project_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(project_dir, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(project_dir, value))


def parse_target_date(text):
    value = str(text).strip().lower()
    if not value:
        return (datetime.now() - timedelta(days=1)).date()
    if value == "today":
        return datetime.now().date()
    if value == "yesterday":
        return (datetime.now() - timedelta(days=1)).date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_dir_from_config(config_path):
    config = load_yaml(config_path)
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    model_save = paths.get("model_save", "")
    if not model_save:
        return ""
    model_save_path = resolve_path(os.path.dirname(config_path), model_save)
    if not model_save_path:
        return ""
    return os.path.dirname(model_save_path)


def check_path(path, kind="file"):
    if kind == "dir":
        return os.path.isdir(path)
    return os.path.isfile(path)


def record(results, name, ok, detail):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})


def summarize(results):
    failed = [item for item in results if not item["ok"]]
    return {"ok": len(failed) == 0, "failed": failed, "results": results}


def request_access_token(session, token_url, email, password, timeout):
    response = session.post(
        token_url,
        json={"email": email, "password": password},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    access = payload.get("access")
    if not access:
        raise RuntimeError("Token response did not include 'access'.")
    return access


def main():
    args = parse_args()
    project_dir = resolve_project_dir()
    base_config = resolve_path(project_dir, args.base_config)
    model_dir = resolve_path(project_dir, args.model_dir) if args.model_dir else ""

    results = []

    record(results, "project_root", check_path(os.path.join(project_dir, "run.py")), project_dir)
    record(results, "base_config", check_path(base_config), base_config)

    config = {}
    if check_path(base_config):
        try:
            config = load_yaml(base_config)
            record(results, "base_config_yaml", True, "Loaded successfully")
        except Exception as exc:
            record(results, "base_config_yaml", False, f"{type(exc).__name__}: {exc}")

    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    for key in ("train_data", "val_data", "test_data"):
        path_value = paths.get(key, "")
        resolved = resolve_path(os.path.dirname(base_config), path_value) if path_value else ""
        record(results, f"config_path:{key}", bool(resolved and check_path(resolved)), resolved or "missing")

    if not model_dir and check_path(base_config):
        model_dir = load_model_dir_from_config(base_config)
    if model_dir:
        record(results, "model_dir", check_path(model_dir, kind="dir"), model_dir)
        for filename in ("best_model.pth", "input_scaler.save", "output_scaler.save", "meta.json"):
            path = os.path.join(model_dir, filename)
            record(results, f"model_file:{filename}", check_path(path), path)
    else:
        record(results, "model_dir", False, "Could not infer model directory from config")

    email = os.getenv("SEL_API_EMAIL", "")
    password = os.getenv("SEL_API_PASSWORD", "")
    record(results, "env:SEL_API_EMAIL", bool(email), "set" if email else "missing")
    record(results, "env:SEL_API_PASSWORD", bool(password), "set" if password else "missing")

    if not args.skip_network:
        if email and password:
            session = requests.Session()
            access_token = ""
            try:
                access_token = request_access_token(
                    session=session,
                    token_url=args.token_url,
                    email=email,
                    password=password,
                    timeout=args.timeout,
                )
                record(results, "sel_token", True, args.token_url)
            except Exception as exc:
                record(results, "sel_token", False, f"{type(exc).__name__}: {exc}")

            if access_token:
                if args.participant:
                    target_date = parse_target_date(args.date)
                    try:
                        response = session.get(
                            args.fetch_url,
                            params={
                                "request_type": "get_sensors_list",
                                "participant_permanent_code": args.participant,
                            },
                            headers={"access-token": access_token},
                            timeout=args.timeout,
                        )
                        response.raise_for_status()
                        sensors_payload = response.json()
                        sensor_count = len(sensors_payload.get("sensors", []) or [])
                        record(
                            results,
                            "sel_fetch_sensors",
                            True,
                            f"{args.participant} sensors={sensor_count}",
                        )
                    except Exception as exc:
                        record(results, "sel_fetch_sensors", False, f"{type(exc).__name__}: {exc}")

                    try:
                        response = session.get(
                            args.fetch_url,
                            params={
                                "request_type": "fetch",
                                "participant_permanent_code": args.participant,
                                "start_date": str(target_date),
                            },
                            headers={"access-token": access_token},
                            timeout=args.timeout,
                        )
                        response.raise_for_status()
                        fetch_payload = response.json()
                        device_count = len((fetch_payload.get("data") or {}).keys())
                        record(
                            results,
                            "sel_fetch_data",
                            True,
                            f"{args.participant} date={target_date} device_groups={device_count}",
                        )
                    except Exception as exc:
                        record(results, "sel_fetch_data", False, f"{type(exc).__name__}: {exc}")
                else:
                    record(
                        results,
                        "sel_fetch_probe",
                        True,
                        "Skipped real fetch probe. Pass --participant to test fetch-data.",
                    )
        else:
            record(results, "sel_network", False, "Missing SEL_API_EMAIL or SEL_API_PASSWORD")

    summary = summarize(results)
    print("SEL doctor")
    for item in results:
        status = "OK" if item["ok"] else "FAIL"
        print(f"[{status}] {item['name']}: {item['detail']}")
    print(f"Overall: {'OK' if summary['ok'] else 'FAIL'}")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
