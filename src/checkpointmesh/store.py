import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import pickle
import time
import zlib
import numpy as np

from .coding import encode, decode
from .io import json_write


def pack_state(state, chunk_bytes):
    tensors = {}
    layouts = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            key = f"tensor_{len(layouts)}"
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            layouts[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).split(".")[-1],
                "bytes": len(raw),
            }
            tensors[key] = raw
            return ("__tensor__", key)
        if isinstance(value, dict):
            return ("__dict__", [(key, visit(item)) for key, item in value.items()])
        if isinstance(value, list):
            return ("__list__", [visit(item) for item in value])
        if isinstance(value, tuple):
            return ("__tuple__", [visit(item) for item in value])
        return ("__value__", value)

    tree = visit(state)
    tensors["metadata"] = pickle.dumps((tree, layouts), protocol=5)
    chunks = {}
    descriptions = {}
    for key, raw in tensors.items():
        names = []
        for start in range(0, len(raw), chunk_bytes):
            name = f"{key}:{start}"
            chunks[name] = raw[start : start + chunk_bytes]
            names.append(name)
        descriptions[key] = names
    return chunks, descriptions


def unpack_state(chunks, descriptions):
    raw = {
        key: b"".join(chunks[name] for name in names)
        for key, names in descriptions.items()
    }
    tree, layouts = pickle.loads(raw.pop("metadata"))
    tensors = {}
    for name, layout in layouts.items():
        if len(raw[name]) != layout["bytes"]:
            raise RuntimeError(f"Invalid tensor byte count: {name}")
        storage = torch.from_numpy(np.frombuffer(raw[name], dtype=np.uint8).copy())
        tensors[name] = storage.view(getattr(torch, layout["dtype"])).reshape(
            layout["shape"]
        )

    def restore(value):
        tag, item = value
        if tag == "__tensor__":
            return tensors[item]
        if tag == "__dict__":
            return {key: restore(data) for key, data in item}
        if tag == "__list__":
            return [restore(data) for data in item]
        if tag == "__tuple__":
            return tuple(restore(data) for data in item)
        return item

    return restore(tree)


def xor_bytes(first, second):
    if len(first) != len(second):
        raise ValueError("XOR delta operands must have equal lengths")
    return np.bitwise_xor(
        np.frombuffer(first, dtype=np.uint8), np.frombuffer(second, dtype=np.uint8)
    ).tobytes()


def policy(risk, config):
    if not 0 <= risk <= 1:
        raise ValueError("Risk score must be in [0, 1]")
    band = sum(risk >= threshold for threshold in [0.10, 0.25, 0.50])
    return {
        "interval": config["interval_steps"][band],
        "parity": [0, 1, 1, 2][band],
        "remote_fraction": [0.0, 0.25, 0.5, 1.0][band],
    }


class CheckpointStore:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output"]) / "checkpoints"
        self.remote = (
            Path(config["remote_root"])
            if config["remote_root"]
            else self.root / "remote"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.remote.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = []
        self.previous = None
        self.previous_step = None
        self.chain_length = 0
        self.chain_bytes = 0
        self.dram = {}
        self.gpu = {}
        self.history = []

    def submit(self, states, step, selected):
        start = time.perf_counter()
        packed = [pack_state(state, self.config["chunk_bytes"]) for state in states]
        pause = time.perf_counter() - start
        future = self.executor.submit(self.write, packed, step, dict(selected), pause)
        self.futures.append(future)
        return pause

    def write(self, packed, step, selected, pause):
        start = time.perf_counter()
        chunks = [values[0] for values in packed]
        full_size = sum(len(raw) for worker in chunks for raw in worker.values())
        variant = self.config["variant"]
        full = (
            self.previous is None
            or variant == "full"
            or self.chain_length >= self.config["max_chain"]
            or self.chain_bytes >= self.config["chain_ratio"] * full_size
        )
        if variant == "no_compaction" and self.previous is not None:
            full = False
        parity = (
            selected["parity"]
            if variant not in ["no_parity", "full", "local_delta"]
            else 0
        )
        remote_fraction = (
            selected["remote_fraction"]
            if variant not in ["full", "local_delta"]
            else 1.0
            if variant == "full"
            else 0.0
        )
        records = []
        for worker, values in enumerate(chunks):
            for key, current in values.items():
                previous = self.previous[worker].get(key) if self.previous else None
                if not full and previous == current:
                    records.append(
                        {
                            "worker": worker,
                            "key": key,
                            "codec": "same",
                            "crc": zlib.crc32(current),
                            "length": len(current),
                        }
                    )
                    continue
                codec = (
                    "xor"
                    if not full
                    and previous is not None
                    and len(previous) == len(current)
                    else "full"
                )
                payload = zlib.compress(
                    xor_bytes(current, previous) if codec == "xor" else current
                )
                records.append(
                    {
                        "worker": worker,
                        "key": key,
                        "codec": codec,
                        "crc": zlib.crc32(current),
                        "length": len(current),
                        "payload": payload,
                    }
                )
        by_worker = [
            [
                record
                for record in records
                if record["worker"] == worker and record["codec"] != "same"
            ]
            for worker in range(len(chunks))
        ]
        stripes = []
        written = delta_bytes = 0
        for index in range(max(map(len, by_worker), default=0)):
            members = [
                records[index] if index < len(records) else None
                for records in by_worker
            ]
            values = [record["payload"] if record else b"" for record in members]
            extra = encode(values, parity)
            locations = {}
            for symbol, payload in enumerate(values + extra):
                domain = (
                    f"worker_{symbol}"
                    if symbol < len(chunks)
                    else f"parity_{symbol - len(chunks)}"
                )
                local = self.root / domain / str(step) / f"{index}.bin"
                remote = self.remote / domain / str(step) / f"{index}.bin"
                copies = []
                if symbol < len(chunks):
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with local.open("wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    copies.append(
                        {"tier": 2, "domain": domain, "path": str(local.resolve())}
                    )
                    self.dram[(step, index, symbol)] = payload
                    copies.append(
                        {"tier": 1, "domain": domain, "key": [step, index, symbol]}
                    )
                    if self.config["gpu_cache"] and torch.cuda.is_available():
                        self.gpu[(step, index, symbol)] = torch.from_numpy(
                            np.frombuffer(payload, dtype=np.uint8).copy()
                        ).cuda()
                        copies.append(
                            {"tier": 0, "domain": domain, "key": [step, index, symbol]}
                        )
                    written += len(payload)
                    delta_bytes += len(payload)
                if symbol >= len(chunks) or (index * 31 + symbol * 17) % 100 < int(
                    remote_fraction * 100
                ):
                    remote.parent.mkdir(parents=True, exist_ok=True)
                    with remote.open("wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    copies.append(
                        {
                            "tier": 3,
                            "domain": f"remote_{domain}",
                            "path": str(remote.resolve()),
                        }
                    )
                    written += len(payload)
                locations[str(symbol)] = copies
                if symbol < len(chunks) and members[symbol] is not None:
                    members[symbol]["stripe"] = index
                    members[symbol]["symbol"] = symbol
            stripes.append(
                {
                    "lengths": list(map(len, values)),
                    "locations": locations,
                    "k": len(chunks),
                    "r": parity,
                }
            )
        for record in records:
            record.pop("payload", None)
        manifest = {
            "step": step,
            "full": full,
            "parent": None if full else self.previous_step,
            "records": records,
            "stripes": stripes,
            "descriptions": [value[1] for value in packed],
            "policy": selected,
        }
        path = self.root / f"epoch_{step}.json"
        temporary = path.with_suffix(".pending")
        json_write(temporary, manifest)
        os.replace(temporary, path)
        json_write(self.remote / f"epoch_{step}.json", manifest)
        temporary = self.root / "latest.pending"
        json_write(temporary, {"step": step})
        os.replace(temporary, self.root / "latest.json")
        self.previous = chunks
        self.previous_step = step
        self.chain_length = 0 if full else self.chain_length + 1
        self.chain_bytes = 0 if full else self.chain_bytes + delta_bytes
        if full:
            self.dram = {
                key: value for key, value in self.dram.items() if key[0] >= step
            }
            self.gpu = {key: value for key, value in self.gpu.items() if key[0] >= step}
        row = {
            "variant": variant,
            "step": step,
            "pause_s": pause,
            "write_s": time.perf_counter() - start,
            "written_bytes": written,
            "full_bytes": full_size,
            "delta_bytes": delta_bytes,
            "write_amplification": written / max(full_size, 1),
            "chain_length": self.chain_length,
            "parity": parity,
            "full": full,
        }
        self.history.append(row)
        return manifest

    def wait(self):
        for future in self.futures:
            future.result()
        self.futures.clear()

    def read_symbol(self, step, stripe, symbol, copies, failure, lost):
        for copy in copies:
            tier = copy["tier"]
            domain = copy["domain"]
            if (
                failure == "process"
                and tier in (0, 1)
                or failure == "gpu"
                and tier == 0
                or failure == "nvme"
                and tier == 2
            ):
                continue
            if (
                failure in ["node", "dual_node"]
                and domain in {f"worker_{i}" for i in lost}
                and tier < 3
            ):
                continue
            key = (step, stripe, symbol)
            if tier == 0 and key in self.gpu:
                return self.gpu[key].cpu().numpy().tobytes()
            if tier == 1 and key in self.dram:
                return self.dram[key]
            if tier >= 2 and Path(copy["path"]).exists():
                return Path(copy["path"]).read_bytes()
        return None

    def restore_chunks(self, step, failure="none", lost=()):
        manifest = __import__("json").loads(
            (self.root / f"epoch_{step}.json").read_text()
        )
        chunks = (
            [{} for _ in manifest["descriptions"]]
            if manifest["full"]
            else self.restore_chunks(manifest["parent"], failure, lost)[0]
        )
        decoded = {}
        for stripe_index, stripe in enumerate(manifest["stripes"]):
            available = {}
            for key, copies in stripe["locations"].items():
                value = self.read_symbol(
                    step, stripe_index, int(key), copies, failure, lost
                )
                if value is not None:
                    available[int(key)] = value
            decoded[stripe_index] = decode(
                available, stripe["k"], stripe["r"], stripe["lengths"]
            )
        next_chunks = [{} for _ in chunks]
        for record in manifest["records"]:
            worker, key = record["worker"], record["key"]
            if record["codec"] == "same":
                value = chunks[worker][key]
            else:
                payload = zlib.decompress(decoded[record["stripe"]][record["symbol"]])
                value = (
                    xor_bytes(payload, chunks[worker][key])
                    if record["codec"] == "xor"
                    else payload
                )
            if len(value) != record["length"] or zlib.crc32(value) != record["crc"]:
                raise RuntimeError("Checkpoint chunk failed length or CRC verification")
            next_chunks[worker][key] = value
        return next_chunks, manifest

    def restore(self, step=None, failure="none", lost=()):
        self.wait()
        if step is None:
            step = __import__("json").loads((self.root / "latest.json").read_text())[
                "step"
            ]
        chunks, manifest = self.restore_chunks(step, failure, lost)
        states = [
            unpack_state(values, names)
            for values, names in zip(chunks, manifest["descriptions"])
        ]
        if any(state["step"] != step for state in states):
            raise RuntimeError("Workers disagree on the committed optimizer step")
        return states

    def close(self):
        self.wait()
        self.executor.shutdown()
