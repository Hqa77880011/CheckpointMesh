import torch
from pathlib import Path
import json
import random
import time
import numpy as np
from torch.nn import functional as F

from .io import configure, seed, device, json_write, csv_write, plot
from .data import prepare, tensors, network
from .store import CheckpointStore, policy

VARIANTS = [
    "checkpointmesh",
    "full",
    "local_delta",
    "no_parity",
    "no_compaction",
    "fixed_policy",
]


def training_state(model, optimizer, step, sampler):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "sampler": sampler,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def states_equal(first, second):
    if isinstance(first, torch.Tensor):
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            states_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(states_equal(a, b) for a, b in zip(first, second))
        )
    return first == second


def train(config):
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    seed(config["seed"])
    x, y = tensors(config)
    models = [network(config).to(device(config)) for _ in range(config["replicas"])]
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=config["lr"]) for model in models
    ]
    history = []
    latest_states = None
    last_checkpoint = 0
    sampler = np.random.default_rng(config["seed"])
    first_step = 0
    if config.get("resume"):
        resume = Path(config["resume"])
        restored = [
            torch.load(
                resume / f"worker_{index}.pt", map_location="cpu", weights_only=False
            )
            for index in range(config["replicas"])
        ]
        if len({state["step"] for state in restored}) != 1:
            raise ValueError(
                "All resumed replicas must belong to the same committed optimizer step"
            )
        for model, optimizer, state in zip(models, optimizers, restored):
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
        first_step = restored[0]["step"]
        sampler.bit_generator.state = restored[0]["sampler"]
        torch.set_rng_state(restored[0]["torch_rng"])
        np.random.set_state(restored[0]["numpy_rng"])
        random.setstate(restored[0]["python_rng"])
        if torch.cuda.is_available() and restored[0]["cuda_rng"]:
            torch.cuda.set_rng_state_all(restored[0]["cuda_rng"])
    if config["steps"] <= first_step:
        raise ValueError("steps must exceed the resumed optimizer step")
    store = CheckpointStore(config)
    last_checkpoint = first_step
    started = time.perf_counter()
    for step in range(first_step + 1, config["steps"] + 1):
        losses = []
        for model, optimizer in zip(models, optimizers):
            model.train()
            indices = sampler.integers(0, len(x), size=config["batch_size"])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(x[indices].to(device(config))), y[indices].to(device(config))
            )
            loss.backward()
            losses.append(loss.item())
        gradients = [
            sum(p.grad for p in parameters) / len(models)
            for parameters in zip(*(model.parameters() for model in models))
        ]
        for model, optimizer in zip(models, optimizers):
            for parameter, gradient in zip(model.parameters(), gradients):
                parameter.grad.copy_(gradient)
            optimizer.step()
        risk = config["risk_scores"][
            min(
                (step - 1) // config["risk_window_steps"],
                len(config["risk_scores"]) - 1,
            )
        ]
        selected = policy(risk, config)
        if config["variant"] == "fixed_policy":
            selected = policy(config["fixed_risk"], config)
        pause = 0.0
        if step - last_checkpoint >= selected["interval"] or step == config["steps"]:
            latest_states = [
                training_state(model, optimizer, step, sampler.bit_generator.state)
                for model, optimizer in zip(models, optimizers)
            ]
            pause = store.submit(latest_states, step, selected)
            last_checkpoint = step
        row = {
            "variant": config["variant"],
            "step": step,
            "loss": float(np.mean(losses)),
            "risk": risk,
            "pause_s": pause,
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    store.wait()
    recovery_start = time.perf_counter()
    lost = [0, 1] if config["failure"] == "dual_node" else [0]
    restored = store.restore(failure=config["failure"], lost=lost)
    recovery_s = time.perf_counter() - recovery_start
    exact = states_equal(latest_states, restored)
    if not exact:
        raise RuntimeError("Restored state differs from the committed state")
    summary = {
        "variant": config["variant"],
        "measurement": "local_storage_fault_injection",
        "failure": config["failure"],
        "completion_s": time.perf_counter() - started,
        "recovery_s": recovery_s,
        "exact_state": exact,
        "mean_pause_s": float(np.mean([row["pause_s"] for row in store.history])),
        "written_bytes": sum(row["written_bytes"] for row in store.history),
        "checkpoints": len(store.history),
    }
    output = Path(config["output"])
    csv_write(output / "training.csv", history)
    csv_write(output / "checkpoint_metrics.csv", store.history)
    json_write(output / "metrics.json", summary)
    json_write(output / "config.json", config)
    torch.save({"model": restored[0]["model"], "config": config}, output / "model.pt")
    store.close()
    print(json.dumps(summary), flush=True)
    return summary


def restore(config):
    store = CheckpointStore(config)
    lost = [0, 1] if config["failure"] == "dual_node" else [0]
    states = store.restore(failure=config["failure"], lost=lost)
    output = Path(config["output"]) / "restored"
    output.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(states):
        torch.save(state, output / f"worker_{index}.pt")
    store.close()
    print(json.dumps({"step": states[0]["step"], "workers": len(states)}))


@torch.no_grad()
def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model = network(state["config"]).to(device(config)).eval()
    model.load_state_dict(state["model"])
    x, y = tensors(config, "test")
    predictions = []
    for start in range(0, len(x), config["batch_size"]):
        predictions.append(
            model(x[start : start + config["batch_size"]].to(device(config)))
            .argmax(1)
            .cpu()
        )
    json_write(
        Path(config["output"]) / "evaluation.json",
        {"accuracy": 100 * (torch.cat(predictions) == y).float().mean().item()},
    )


def compare(config):
    rows = []
    for variant in config["variants"]:
        current = dict(
            config,
            variant=variant,
            failure="none",
            output=str(Path(config["output"]) / variant),
        )
        summary = train(current)
        rows.append(summary)
        for failure in config["failures"]:
            store = CheckpointStore(current)
            start = time.perf_counter()
            try:
                states = store.restore(
                    failure=failure, lost=[0, 1] if failure == "dual_node" else [0]
                )
                result = {
                    "variant": variant,
                    "failure": failure,
                    "recoverable": True,
                    "recovery_s": time.perf_counter() - start,
                    "step": states[0]["step"],
                }
            except RuntimeError as error:
                result = {
                    "variant": variant,
                    "failure": failure,
                    "recoverable": False,
                    "recovery_s": None,
                    "error": str(error),
                }
            store.close()
            rows.append(result)
    csv_write(Path(config["output"]) / "comparison.csv", rows)


def main():
    command, config = configure(
        ["prepare", "train", "restore", "evaluate", "compare", "plot"]
    )
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "restore":
        restore(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        plot(
            root / "checkpoint_metrics.csv",
            root / "checkpoint_cost.pdf",
            "step",
            "write_amplification",
        )
