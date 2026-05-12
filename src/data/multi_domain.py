from __future__ import annotations
import random
from typing import List, Optional, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset


def subject_split(dataset, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Set[str], Set[str]]:
    """Returns (train_subject_ids, val_subject_ids) — disjoint sets."""
    all_subjects = sorted(set(dataset.get_subject_ids()))
    rng = random.Random(seed)
    rng.shuffle(all_subjects)
    n_train = max(1, int(len(all_subjects) * train_ratio))
    train_ids = set(all_subjects[:n_train])
    val_ids = set(all_subjects[n_train:])
    return train_ids, val_ids


def _filter_by_subjects(dataset, subject_ids: Set[str]) -> Subset:
    indices = [
        i for i, s in enumerate(dataset.samples)
        if s["subject_id"] in subject_ids
    ]
    return Subset(dataset, indices)


def build_loaders(
    source_datasets: List,
    held_out_domain: Optional[int],
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    collate_fn=None,
) -> Tuple[DataLoader, DataLoader]:
    train_subsets = []
    val_subsets = []

    for ds in source_datasets:
        train_ids, val_ids = subject_split(ds, train_ratio=0.8, seed=seed)
        train_subsets.append(_filter_by_subjects(ds, train_ids))
        val_subsets.append(_filter_by_subjects(ds, val_ids))

    train_data = ConcatDataset(train_subsets)
    val_data = ConcatDataset(val_subsets)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_test_loader(
    test_dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def get_domain_datasets(cfg, all_datasets: List) -> Tuple[List, object]:
    held_out = cfg.held_out_domain
    source = [ds for ds in all_datasets if ds.domain_id != held_out]
    test_ds = all_datasets[held_out]
    return source, test_ds
