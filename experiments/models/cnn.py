import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm

from typing import OrderedDict, Union, TypeVar, Type, Dict, List
from torch.utils.data import DataLoader
from dataclasses import dataclass


X = TypeVar("X", bound="CNN")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=8, stride=2, padding=3), nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(128, 256, kernel_size=3), nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6400, out_features=10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x is [B, 3, 32, 32] for CIFAR10
        logits = self.cnn(x)
        return logits

    @classmethod
    def load(cls: Type[X], path: Union[str, os.PathLike]) -> X:
        model = cls()
        state_dict = torch.load(path)
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).mean()


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss(reduction="none")(logits, labels)


@dataclass
class PredictionMetrics:
    losses: np.ndarray
    labels: np.ndarray
    logits: np.ndarray

    def __post_init__(self):
        assert self.losses.shape == self.labels.shape
        assert self.logits.shape[0] == self.losses.shape[0] or len(self.losses) == 0
        assert len(self.losses.shape) == 1

    @property
    def preds(self) -> float:
        if len(self.losses) == 0:
            return float("nan")
        return np.argmax(self.logits, axis=1)

    @property
    def accuracy(self) -> float:
        if len(self.losses) == 0:
            return float("nan")
        return compute_accuracy(self.preds, self.labels)

    @property
    def loss(self) -> float:
        if len(self.losses) == 0:
            return float("nan")
        return np.mean(self.losses)

    def __str__(self) -> str:
        if len(self.labels) == 0:
            return f"PredictionMetrics<n=0>"
        return f"PredictionMetrics<accuracy={self.accuracy:.4f}, loss={self.loss:.4f}, n={len(self.losses)}>"


def compute_prediction_metrics(model: CNN, device: torch.device, data_loader: DataLoader) -> PredictionMetrics:
    if len(data_loader) == 0:
        return PredictionMetrics(
            losses=np.array([], dtype=np.float64), logits=np.array([[]], dtype=np.float64), labels=np.array([], dtype=np.int64)
        )

    model.to(device)
    model.eval()

    losses = []
    logits = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Test ", unit="batch", disable=None):
            inputs = batch["image"].to(device)
            target = batch["label"].to(device)

            output = model(inputs)
            loss = compute_loss(output, target).detach().cpu().numpy()
            logits.append(output.detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())
            losses.append(loss)

    metrics = PredictionMetrics(
        losses=np.concatenate(losses),
        logits=np.concatenate(logits),
        labels=np.concatenate(labels)
    )
    return metrics


def collate_image_batch(batch: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.stack([torch.tensor(sample["image"]) for sample in batch]).to(device),
        "label": torch.tensor([sample["label"] for sample in batch], dtype=torch.long).to(device)
    }
