from __future__ import annotations
import random
from typing import List, Dict, Any
import torch
import numpy as np


def mixup_collate_fn(alpha: float = 0.2):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        domain_map: Dict[int, List[int]] = {}
        for i, sample in enumerate(batch):
            d = sample["domain_id"]
            domain_map.setdefault(d, []).append(i)

        waveforms = torch.stack([s["waveform"] for s in batch])
        labels = torch.tensor([s["label"] for s in batch], dtype=torch.float32)
        domain_ids = torch.tensor([s["domain_id"] for s in batch], dtype=torch.long)

        mixed_waveforms = waveforms.clone()
        mixed_labels = labels.clone()

        for i, sample in enumerate(batch):
            my_domain = sample["domain_id"]
            other_domains = [d for d in domain_map if d != my_domain]

            if not other_domains:
                continue

            other_domain = random.choice(other_domains)
            j = random.choice(domain_map[other_domain])

            lam = float(np.random.beta(alpha, alpha))
            mixed_waveforms[i] = lam * waveforms[i] + (1 - lam) * waveforms[j]
            mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[j]

        return {
            "waveform": mixed_waveforms,
            "label": mixed_labels,
            "domain_id": domain_ids,
        }

    return collate


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "waveform": torch.stack([s["waveform"] for s in batch]),
        "label": torch.tensor([s["label"] for s in batch], dtype=torch.float32),
        "domain_id": torch.tensor([s["domain_id"] for s in batch], dtype=torch.long),
    }
