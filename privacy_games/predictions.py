import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch import nn
from datasets import Dataset
from typing import Optional


def predict(model: nn.Module, data: Dataset, batch_size: int, device: Optional[str] = None, label_column: str = "labels") -> pd.DataFrame:
    """
    Compute predictions for a dataset using a model

    :param model: Model to use for predictions. We assume that the model returns a struct with a `logits` member
    :param data: Dataset to compute predictions on
    :param batch_size: Batch size for predictions
    :param device: Device to use for predictions
    :param label_column: Name of the column in the dataset that contains the labels
    :return: DataFrame of predictions with columns: logits, labels
    """
    model.to(device)
    predictions = []
    with data.formatted_as(type="torch"):
        with torch.no_grad():
            for batch in DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False):
                batch = {k: t.to(model.device) for k, t in batch.items()}
                labels = batch.pop(label_column)
                output = model(**batch, labels=labels)
                batch_df = pd.DataFrame({
                    "logits": [output.logits.cpu().numpy()[i] for i in range(len(labels))],
                    "labels": [labels.cpu().numpy()[i] for i in range(len(labels))]
                })
                predictions.append(batch_df)
    return pd.concat(predictions, ignore_index=True)

