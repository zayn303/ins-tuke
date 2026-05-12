from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import compute_metrics


def _group_mean(
    probs: np.ndarray,
    labels: np.ndarray,
    keys: List[str],
    domain_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate probabilities by key (mean), return (mean_probs, labels, domain_ids) per key."""
    buckets: Dict[str, Dict] = defaultdict(lambda: {"probs": [], "label": None, "domain": None})
    for p, lab, k, d in zip(probs, labels, keys, domain_ids):
        buckets[k]["probs"].append(p)
        buckets[k]["label"] = lab
        buckets[k]["domain"] = d
    out_probs = []
    out_labels = []
    out_domains = []
    for k, v in buckets.items():
        out_probs.append(float(np.mean(v["probs"])))
        out_labels.append(int(v["label"]))
        out_domains.append(int(v["domain"]))
    return np.array(out_probs), np.array(out_labels), np.array(out_domains)


def run_eval(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    all_labels: List[int] = []
    all_probs: List[float] = []
    all_domain_ids: List[int] = []
    all_recording_ids: List[str] = []
    all_subject_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["label"].cpu().numpy()

            features = backbone(input_values, attention_mask=attention_mask)
            logits = classifier(features).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
            if "domain_id" in batch:
                all_domain_ids.extend(batch["domain_id"].cpu().numpy().tolist())
            all_recording_ids.extend(batch.get("recording_id", [""] * len(labels)))
            all_subject_ids.extend(batch.get("subject_id", [""] * len(labels)))

    if not all_labels:
        return {"uar": float("nan"), "auc_roc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}

    seg_probs = np.array(all_probs)
    seg_labels = np.array(all_labels).astype(int)
    seg_domains = np.array(all_domain_ids) if all_domain_ids else np.zeros_like(seg_labels)
    seg_rec_ids = all_recording_ids
    seg_subj_ids = all_subject_ids

    seg_preds = (seg_probs >= 0.5).astype(int)
    seg_metrics = compute_metrics(seg_labels, seg_preds, seg_probs)

    rec_probs, rec_labels, rec_domains = _group_mean(seg_probs, seg_labels, seg_rec_ids, seg_domains)
    rec_preds = (rec_probs >= 0.5).astype(int)
    rec_metrics = compute_metrics(rec_labels.astype(int), rec_preds, rec_probs)

    subj_probs, subj_labels, subj_domains = _group_mean(seg_probs, seg_labels, seg_subj_ids, seg_domains)
    subj_preds = (subj_probs >= 0.5).astype(int)
    subj_metrics = compute_metrics(subj_labels.astype(int), subj_preds, subj_probs)

    metrics: Dict[str, float] = dict(subj_metrics)
    metrics["uar_segment"] = seg_metrics["uar"]
    metrics["uar_recording"] = rec_metrics["uar"]

    if all_domain_ids:
        for d in np.unique(subj_domains):
            mask = subj_domains == d
            if mask.sum() < 1:
                continue
            try:
                dm = compute_metrics(
                    subj_labels[mask].astype(int),
                    subj_preds[mask],
                    subj_probs[mask],
                )
                for k, v in dm.items():
                    metrics[f"{k}_d{int(d)}"] = v
            except Exception:
                continue

    return metrics
