import argparse
import os
import sys

import yaml

from evaluate import main as evaluate_main
from extractsynthdata import save_synthetic_datasets
from train import main as train_main


def resolve_config_paths(config, config_path):
    base_dir = os.path.dirname(os.path.abspath(config_path))

    paths = config.get("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str) and value and not os.path.isabs(value):
            paths[key] = os.path.normpath(os.path.join(base_dir, value))

    extract_cfg = config.get("extractsynthetic", {})
    base_path = extract_cfg.get("base_path")
    if isinstance(base_path, str) and base_path and not os.path.isabs(base_path):
        extract_cfg["base_path"] = os.path.normpath(os.path.join(base_dir, base_path))

    config["paths"] = paths
    config["extractsynthetic"] = extract_cfg
    return config


def main():
    parser = argparse.ArgumentParser(description="NILM Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = resolve_config_paths(config, args.config)

    action = config.get("action")
    if action == "train":
        train_main(config)
    elif action == "evaluate":
        evaluate_main(config)
    elif action == "extractsynthetic":
        extract_config = config.get("extractsynthetic", {})
        save_synthetic_datasets(**extract_config)
    else:
        print("Invalid action specified in config file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
