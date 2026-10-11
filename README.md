# CheckpointMesh

Implementation of **CheckpointMesh: Risk-Adaptive Multi-Tier Delta Checkpointing with Cross-Worker Parity**. Typed tensor chunks are stored as compressed full bytes or exact XOR deltas. A committed manifest records their ancestry, placement and cross-worker parity.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and workflow

`configs/small.json` uses a seeded generated classification workload for inspecting the execution protocol with little compute. Labels come from a fixed linear decision rule; this dataset is a development workload, separate from the paper's vision and language benchmarks. `configs/cifar10.json` downloads the official [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) splits through torchvision and uses a CIFAR-style ResNet-18. `prepare` writes training and test arrays under the configured data directory. CIFAR test labels are used only by `evaluate`.

```bash
python -m checkpointmesh prepare --config configs/small.json
python -m checkpointmesh train --config configs/small.json --output outputs/mesh
python -m checkpointmesh evaluate --config configs/small.json --checkpoint outputs/mesh/model.pt --output outputs/mesh
python -m checkpointmesh restore --config configs/small.json --output outputs/mesh
python -m checkpointmesh plot --config configs/small.json --output outputs/mesh
```

The training harness runs synchronized model replicas locally and averages their gradients before each optimizer transition. It snapshots model, optimizer, Python/NumPy/Torch random state, sampler state and step identifier. It is a local checkpoint backend and protocol workload rather than a multi-node throughput harness.

`restore` loads the latest committed epoch and writes complete `restored/worker_N.pt` states. Continue from this directory with a new run directory and a larger total step count:

```bash
python -m checkpointmesh train --config configs/small.json --output outputs/resumed --steps 100 --set 'resume="outputs/mesh/restored"'
```

Resume loads all replica weights, optimizer states, RNG state and the shared sampler before the next step. The new output directory begins a fresh full checkpoint chain. To inspect failure-domain recovery:

```bash
python -m checkpointmesh restore --config configs/small.json --output outputs/mesh --set 'failure="node"'
python -m checkpointmesh restore --config configs/small.json --output outputs/mesh --set 'failure="dual_node"'
python -m checkpointmesh compare --config configs/small.json --output outputs/comparison
```

Failure injection removes selected storage tiers from the recovery path without deleting files. A node failure excludes local tiers for worker 0; a dual-node failure excludes workers 0 and 1. Process/GPU and NVMe failures use their surviving tiers. An epoch with insufficient surviving coded symbols raises a recovery error. Comparisons retain an explicit `recoverable=false` result rather than substituting a success value.

## Placement, parity and control

Commands read the JSON configuration. `--data`, `--output`, `--device`, `--variant`, `--steps`, `--batch-size`, `--seed`, `--limit` and `--checkpoint` override run settings. Other fields use `--set KEY=JSON`, for example `--set chunk_bytes=32768` or `--set 'failure="node"'`. String values need JSON quotes. Use separate output directories for policies.

| Setting | Meaning |
| --- | --- |
| `chunk_bytes` | Maximum raw chunk size, default 64 KiB |
| `replicas` | Data symbols per cross-worker stripe |
| `gpu_cache` | Enable tier-zero CUDA byte buffers |
| `remote_root` | Mounted remote storage path; empty uses a local remote-domain directory |
| `risk_scores`, `risk_window_steps` | Externally supplied risk trace and window length |
| `interval_steps` | Intervals for the four risk bands |
| `max_chain`, `chain_ratio` | Compaction limits on delta count and cumulative compressed bytes |

Risk boundaries are 0.10, 0.25 and 0.50. Intervals are [20, 10, 5, 2] optimizer steps in the local harness; the paper's time-based intervals can be represented by choosing equivalent step counts. Parity counts are [0, 1, 1, 2] and remote fractions increase with risk. Risk scores are inputs, not predictions derived from the injected future failures. A calibrated production risk model is outside this local harness.

Tier zero holds CUDA byte buffers, tier one holds process-resident host bytes, tier two writes local files, and tier three writes the configured remote path. Distinct directories represent logical failure domains on a single machine. For physical node durability, `remote_root` must reside on storage independent of the worker machine. Process-resident tiers are available during the same running process; a restarted recovery command uses the persistent tiers.

One parity symbol is XOR. Two parity symbols use systematic GF(256) coding and recover any two erased symbols in the stripe. Original symbol lengths are retained. CRC32 and byte-length checks validate reconstructed chunks before deserialization. Snapshot packing occurs before asynchronous persistence, so later optimizer changes cannot alter queued checkpoint bytes. Manifest publication uses atomic file replacement after required data and parity writes complete.

## Baselines and outputs

`full` writes full snapshots with remote copies. `local_delta` uses local incremental storage. `fixed_policy` holds risk policy constant; `no_parity` and `no_compaction` isolate parity and chain limits. `compare` runs the configured policies on the same data and checks process, accelerator, storage, node and dual-node recovery.

`training.csv` contains actual loss, checkpoint pause and elapsed time. `checkpoint_metrics.csv` reports write time, raw/full size, written bytes, delta size, write amplification and chain length. `metrics.json` reports completion/recovery time and exact state equality, including optimizer, RNG and sampler state. Recovery errors propagate for single-policy runs. `comparison.csv` preserves both successful and unrecoverable failure cases. `checkpoint_cost.pdf` visualizes write amplification over the committed steps.

The backend compacts by chain length and byte amplification. It does not infer a distributed restore-time service objective, emulate GPU/node hardware failures, or implement peer TCP transport. Directory fault injection and local timings describe the supplied backend. Checkpoints use Python serialization for non-tensor metadata and should be loaded only from trusted runs.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
