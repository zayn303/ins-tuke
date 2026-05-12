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
    all_domain_ids = []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].squeeze(1).to(device)
            labels = batch["label"].cpu().numpy()

            features = backbone(waveform)
            logits = classifier(features).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)
            if "domain_id" in batch:
                all_domain_ids.append(batch["domain_id"].cpu().numpy())

    if not all_labels:
        return {"uar": float("nan"), "auc_roc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = compute_metrics(all_labels.astype(int), all_preds, all_probs)

    if all_domain_ids:
        all_domain_ids = np.concatenate(all_domain_ids)
        for d in np.unique(all_domain_ids):
            mask = all_domain_ids == d
            if mask.sum() < 2:
                continue
            dm = compute_metrics(all_labels[mask].astype(int), all_preds[mask], all_probs[mask])
            for k, v in dm.items():
                metrics[f"{k}_d{int(d)}"] = v

    return metrics
