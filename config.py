"""YAML configuration loading with dotted command-line overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        dotted_key, raw_value = item.split("=", 1)
        _set_dotted(config, dotted_key, yaml.safe_load(raw_value))
    return config


def _set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted override, e.g. --set detector.backbone=mlp",
    )
    return parser
