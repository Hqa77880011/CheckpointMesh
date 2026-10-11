import torch
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure(commands):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=commands)
    parser.add_argument("--config", required=True)
    for key in ["data", "output", "device", "variant", "checkpoint"]:
        parser.add_argument("--" + key)
    for key in ["steps", "batch-size", "seed", "limit"]:
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key, value in vars(args).items():
        if key not in ["config", "command", "set"] and value is not None:
            config[key] = value
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in config:
            parser.error(f"Unknown configuration key: {key}")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            parser.error(f"{key} must be a JSON value")
    seed(config.get("seed", 2026))
    torch.set_num_threads(config.get("threads", 2))
    return args.command, config


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def device(config):
    value = config.get("device", "auto")
    return torch.device(
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        raise ValueError("No measurements to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(path, output, x, y):
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV has no measurements")
    figure, axis = plt.subplots(figsize=(7, 4))
    for variant in sorted(set(row["variant"] for row in rows)):
        subset = [row for row in rows if row["variant"] == variant]
        subset.sort(key=lambda row: float(row[x]))
        axis.plot(
            [float(row[x]) for row in subset],
            [float(row[y]) for row in subset],
            marker="o",
            label=variant,
        )
    axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    axis.legend()
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
