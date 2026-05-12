from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.utils.reproducibility import seed_everything
from src.utils.logging import init_wandb, log_metrics, finish_wandb
from src.data.italian_pd import ItalianPDDataset
from src.data.mdvr_kcl import MDVRKCLDataset
from src.data.voice_samples_ah import VoiceSamplesAHDataset
from src.data.multi_domain import build_loaders, build_test_loader, get_domain_datasets, subject_split, _filter_by_subjects
from src.data.augmentation import mixup_collate_fn, default_collate_fn
from src.models.backbone import SpeechBackbone
from src.models.classifier import PDClassifier


_DOMAIN_CLASSES = [ItalianPDDataset, MDVRKCLDataset, VoiceSamplesAHDataset]
_DOMAIN_PATHS = [
    "Italian_Parkinsons_Voice_and_Speech/italian_parkinson",
    "Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both early and advanced Parkinson's disease patients and healthy controls/Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both",
    "Voice Samples for Patients with Parkinson’s Disease and Healthy Controls",
]


def _build_datasets(cfg: DictConfig):
    data_root = Path(cfg.data_root)
    datasets = []
    for cls, rel_path in zip(_DOMAIN_CLASSES, _DOMAIN_PATHS):
        root = data_root / rel_path
        ds = cls(root, sample_rate=cfg.sample_rate, max_duration=cfg.max_duration)
        datasets.append(ds)
    return datasets


def _build_trainer(cfg: DictConfig, backbone, classifier, device, source_datasets):
    method = cfg.method_name
    if method == "erm":
        from src.training.erm_trainer import ERMTrainer
        return ERMTrainer(backbone, classifier, lr=cfg.lr, weight_decay=cfg.weight_decay, device=device)
    elif method == "difl":
        from src.training.difl_trainer import DIFLTrainer
        return DIFLTrainer(
            backbone, classifier,
            source_domain_ids=[ds.domain_id for ds in source_datasets],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            lambda_grl_max=cfg.lambda_grl_max,
            total_epochs=cfg.epochs,
            device=device,
        )
    elif method == "mixup":
        from src.training.mixup_trainer import MixupTrainer
        return MixupTrainer(backbone, classifier, lr=cfg.lr, weight_decay=cfg.weight_decay, device=device)
    elif method == "maml":
        from src.training.maml_trainer import MAMLTrainer
        return MAMLTrainer(
            backbone, classifier,
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            inner_lr=cfg.inner_lr,
            n_inner_steps=cfg.n_inner_steps,
            n_episodes_per_epoch=cfg.n_episodes_per_epoch,
            device=device,
        )
    elif method == "coral":
        from src.training.coral_trainer import CORALTrainer
        return CORALTrainer(
            backbone, classifier,
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            lambda_coral=cfg.lambda_coral,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _build_maml_domain_loaders(source_datasets, cfg):
    domain_loaders = {}
    for ds in source_datasets:
        train_ids, _ = subject_split(ds, train_ratio=0.8, seed=cfg.seed)
        train_subset = _filter_by_subjects(ds, train_ids)
        loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=default_collate_fn,
        )
        domain_loaders[ds.domain_id] = loader
    return domain_loaders


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(OmegaConf.to_yaml(cfg))

    all_datasets = _build_datasets(cfg)
    source_datasets, test_dataset = get_domain_datasets(cfg, all_datasets)

    collate = mixup_collate_fn(cfg.get("mixup_alpha", 0.2)) if cfg.method_name == "mixup" else default_collate_fn

    train_loader, val_loader = build_loaders(
        source_datasets,
        held_out_domain=cfg.held_out_domain,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        collate_fn=collate,
    )
    test_loader = build_test_loader(test_dataset, batch_size=cfg.batch_size)

    backbone = SpeechBackbone(
        cfg.model,
        freeze_backbone=cfg.freeze_backbone,
        unfreeze_top_n_layers=cfg.unfreeze_top_n_layers,
    ).to(device)
    classifier = PDClassifier().to(device)

    trainer = _build_trainer(cfg, backbone, classifier, device, source_datasets)

    if cfg.method_name == "maml":
        maml_domain_loaders = _build_maml_domain_loaders(source_datasets, cfg)

    run = init_wandb(cfg)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_uar = -1.0

    for epoch in range(cfg.epochs):
        if cfg.method_name == "maml":
            train_loss = trainer.train_epoch(maml_domain_loaders)
        elif cfg.method_name == "difl":
            train_loss = trainer.train_epoch(train_loader, epoch=epoch)
        else:
            train_loss = trainer.train_epoch(train_loader)

        print(f"Epoch {epoch + 1}/{cfg.epochs} | train_loss={train_loss:.4f}")

        val_metrics = trainer.eval_epoch(val_loader)

        log_payload = {"train/loss": train_loss, "epoch": epoch + 1}
        log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
        log_metrics(run, log_payload, step=epoch + 1)

        print(f"  val_uar={val_metrics.get('uar', float('nan')):.4f}")

        if val_metrics.get("uar", -1) > best_uar:
            best_uar = val_metrics["uar"]
            ckpt_path = ckpt_dir / f"{cfg.method_name}_{cfg.model}_held{cfg.held_out_domain}_best.pt"
            trainer.save_checkpoint(ckpt_path, epoch=epoch + 1, metrics=val_metrics)

    test_metrics = trainer.eval_epoch(test_loader)
    print(f"\nTest metrics (held_out_domain={cfg.held_out_domain}): {test_metrics}")
    log_metrics(run, {f"test/{k}": v for k, v in test_metrics.items()})

    finish_wandb(run)


if __name__ == "__main__":
    main()
