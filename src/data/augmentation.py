from __future__ import annotations
import random
from typing import List, Dict, Any
import torch
import numpy as np


def _collate_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "subject_id": [s["subject_id"] for s in batch],
        "recording_id": [s["recording_id"] for s in batch],
        "task_code": [s.get("task_code", "") for s in batch],
        "path": [s.get("path", "") for s in batch],
    }


def mixup_collate_fn(alpha: float = 0.2):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        domain_map: Dict[int, List[int]] = {}
        for i, sample in enumerate(batch):
            d = sample["domain_id"]
            domain_map.setdefault(d, []).append(i)

        input_values = torch.stack([s["input_values"] for s in batch])
        attention_mask = torch.stack([s["attention_mask"] for s in batch])
        labels = torch.tensor([s["label"] for s in batch], dtype=torch.float32)
        domain_ids = torch.tensor([s["domain_id"] for s in batch], dtype=torch.long)

        mixed_inputs = input_values.clone()
        mixed_labels = labels.clone()

        for i, sample in enumerate(batch):
            my_domain = sample["domain_id"]
            other_domains = [d for d in domain_map if d != my_domain]
            if not other_domains:
                continue
            other_domain = random.choice(other_domains)
            j = random.choice(domain_map[other_domain])
            lam = float(np.random.beta(alpha, alpha))
            mixed_inputs[i] = lam * input_values[i] + (1 - lam) * input_values[j]
            mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[j]

        out = {
            "input_values": mixed_inputs,
            "attention_mask": attention_mask,
            "label": mixed_labels,
            "domain_id": domain_ids,
        }
        out.update(_collate_meta(batch))
        return out

    return collate


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "input_values": torch.stack([s["input_values"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "label": torch.tensor([s["label"] for s in batch], dtype=torch.float32),
        "domain_id": torch.tensor([s["domain_id"] for s in batch], dtype=torch.long),
    }
    out.update(_collate_meta(batch))
    return out
