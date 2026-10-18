# CheckpointMesh

Implementation of **CheckpointMesh: Risk-Adaptive Multi-Tier Delta Checkpointing with Cross-Worker Parity**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`configs/reproduce.json` generates a seeded classification workload for four logical worker replicas. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is available through `configs/cifar10.json`. Storage directories represent logical tiers; set `remote_root` to independent storage when measuring physical node durability.

```bash
python -m checkpointmesh prepare --config configs/reproduce.json
```

## Training and execution

`configs/reproduce.json` fixes seed `17`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m checkpointmesh train --config configs/reproduce.json --output outputs/checkpointmesh
```

## Evaluation

```bash
python -m checkpointmesh evaluate --config configs/reproduce.json --checkpoint outputs/checkpointmesh/model.pt --output outputs/checkpointmesh
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m checkpointmesh compare --config configs/reproduce.json --output outputs/comparison
python -m checkpointmesh plot --config configs/reproduce.json --output outputs/checkpointmesh
```

## Outputs

Run directories contain `training.csv`, `checkpoint_metrics.csv`, `metrics.json`, `model.pt`, `evaluation.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`. The plot command reads the main run's `checkpoint_metrics.csv` and writes `outputs/checkpointmesh/checkpoint_cost.pdf`.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training and full evaluation were not run for this review. The GitHub Actions workflow checks packaging and source code.
