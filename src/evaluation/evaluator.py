from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import compute_metrics


def run_eval(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].squeeze(1).to(device)  # [B, T]
            labels = batch["label"].cpu().numpy()

            features = backbone(waveform)
            logits = classifier(features).squeeze(-1)  # [B]
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    return compute_metrics(all_labels.astype(int), all_preds, all_probs)
