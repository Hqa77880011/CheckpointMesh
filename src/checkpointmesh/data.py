import torch
from pathlib import Path
import numpy as np
from torch import nn
from torchvision import datasets, models
from .io import seed


def prepare(config):
    out = Path(config["data"])
    out.mkdir(parents=True, exist_ok=True)
    if config["dataset"] == "generated":
        rng = np.random.default_rng(config["seed"])
        weights = rng.normal(size=(config["input_dim"], config["classes"])).astype(
            "float32"
        )
        for split in ["train", "test"]:
            x = rng.normal(size=(config["samples"], config["input_dim"])).astype(
                "float32"
            )
            y = np.argmax(x @ weights, axis=1).astype("int64")
            np.savez_compressed(out / f"{split}.npz", x=x, y=y)
    elif config["dataset"] in ["cifar10", "cifar100"]:
        cls = datasets.CIFAR10 if config["dataset"] == "cifar10" else datasets.CIFAR100
        for split in ["train", "test"]:
            data = cls(str(out / "raw"), train=split == "train", download=True)
            count = config.get("limit", 0) or len(data)
            np.savez_compressed(
                out / f"{split}.npz",
                x=data.data[:count],
                y=np.asarray(data.targets[:count]),
            )
    else:
        raise ValueError("dataset must be generated, cifar10 or cifar100")


def tensors(config, split="train"):
    with np.load(Path(config["data"]) / f"{split}.npz", allow_pickle=False) as values:
        x, y = values["x"], values["y"]
    if config.get("limit"):
        x, y = x[: config["limit"]], y[: config["limit"]]
    images = x.ndim == 4
    x = torch.from_numpy(x.copy()).float()
    if images:
        x = x.permute(0, 3, 1, 2) / 255
        x = (x - 0.5) / 0.5
    return x, torch.from_numpy(y.copy()).long()


def network(config, stateless=False):
    seed(config["seed"])
    if config["dataset"] == "generated":
        return nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["classes"]),
        )
    model = models.resnet18(
        weights=None,
        num_classes=10 if config["dataset"] == "cifar10" else 100,
        norm_layer=(lambda channels: nn.GroupNorm(8, channels))
        if stateless
        else nn.BatchNorm2d,
    )
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    return model
