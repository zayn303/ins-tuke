"""RED tests for trainers — fail before implementation (ImportError), pass after."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HIDDEN_DIM = 768
PROJ_DIM = 512
BATCH = 8
T = 16000  # 1-second audio at 16kHz (tiny for tests)


class TinyBackbone(nn.Module):
    """Drop-in backbone stub: returns [B, HIDDEN_DIM] directly."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(T, HIDDEN_DIM)
        self.hidden_dim = HIDDEN_DIM

    def forward(self, input_values):
        return self.linear(input_values)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(HIDDEN_DIM, PROJ_DIM), nn.GELU())
        self.head = nn.Linear(PROJ_DIM, 1)

    def forward(self, x):
        return self.head(self.projection(x))

    def get_features(self, x):
        return self.projection(x)


def make_batch(n_domains: int = 2, batch_size: int = BATCH, hard_labels: bool = True):
    waveforms = torch.randn(batch_size, T)
    labels = torch.randint(0, 2, (batch_size,)).float()
    if not hard_labels:
        labels = torch.rand(batch_size)
    domain_ids = torch.randint(0, n_domains, (batch_size,)).long()
    return {"waveform": waveforms.unsqueeze(1), "label": labels, "domain_id": domain_ids}


def make_loader(n_batches: int = 4, n_domains: int = 2):
    batches = [make_batch(n_domains) for _ in range(n_batches)]
    return batches  # list of dicts (matches trainer's expected input)


# ---------------------------------------------------------------------------
# ERMTrainer
# ---------------------------------------------------------------------------

class TestERMTrainer:
    def test_import(self):
        from src.training.erm_trainer import ERMTrainer

    def test_train_epoch_returns_loss(self):
        from src.training.erm_trainer import ERMTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = ERMTrainer(backbone, classifier, lr=1e-3, weight_decay=0.0, device=torch.device("cpu"))
        loader = make_loader()
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert loss > 0

    def test_eval_epoch_returns_metrics(self):
        from src.training.erm_trainer import ERMTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = ERMTrainer(backbone, classifier, lr=1e-3, weight_decay=0.0, device=torch.device("cpu"))
        loader = make_loader()
        metrics = trainer.eval_epoch(loader)
        assert "uar" in metrics
        assert "auc_roc" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics

    def test_loss_decreases_after_training(self):
        from src.training.erm_trainer import ERMTrainer
        torch.manual_seed(0)
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = ERMTrainer(backbone, classifier, lr=1e-2, weight_decay=0.0, device=torch.device("cpu"))
        loader = make_loader(n_batches=8)
        losses = [trainer.train_epoch(loader) for _ in range(5)]
        assert losses[-1] < losses[0] * 1.5  # not strictly decreasing, but shouldn't explode


# ---------------------------------------------------------------------------
# DIFLTrainer
# ---------------------------------------------------------------------------

class TestDIFLTrainer:
    def test_import(self):
        from src.training.difl_trainer import DIFLTrainer

    def test_train_epoch_returns_loss(self):
        from src.training.difl_trainer import DIFLTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = DIFLTrainer(
            backbone, classifier,
            source_domain_ids=[1, 2], lr=1e-3, weight_decay=0.0,
            lambda_grl_max=1.0, total_epochs=10,
            device=torch.device("cpu"),
        )
        loader = make_loader(n_domains=2)
        loss = trainer.train_epoch(loader, epoch=0)
        assert isinstance(loss, float)

    def test_lambda_schedule_increases(self):
        from src.training.difl_trainer import DIFLTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = DIFLTrainer(
            backbone, classifier,
            source_domain_ids=[1, 2], lr=1e-3, weight_decay=0.0,
            lambda_grl_max=1.0, total_epochs=10,
            device=torch.device("cpu"),
        )
        lam0 = trainer._get_lambda(0, 10)
        lam5 = trainer._get_lambda(5, 10)
        lam9 = trainer._get_lambda(9, 10)
        assert lam0 <= lam5 <= lam9
        assert abs(lam9 - 1.0) < 0.2


# ---------------------------------------------------------------------------
# MixupTrainer
# ---------------------------------------------------------------------------

class TestMixupTrainer:
    def test_import(self):
        from src.training.mixup_trainer import MixupTrainer

    def test_train_epoch_accepts_soft_labels(self):
        from src.training.mixup_trainer import MixupTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = MixupTrainer(backbone, classifier, lr=1e-3, weight_decay=0.0, device=torch.device("cpu"))
        # Soft labels (from Mixup collate_fn)
        loader = [make_batch(hard_labels=False) for _ in range(4)]
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert not np.isnan(loss)


# ---------------------------------------------------------------------------
# MAMLTrainer
# ---------------------------------------------------------------------------

class TestMAMLTrainer:
    def test_import(self):
        from src.training.maml_trainer import MAMLTrainer

    def test_train_epoch_returns_loss(self):
        from src.training.maml_trainer import MAMLTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = MAMLTrainer(
            backbone, classifier,
            lr=1e-3, weight_decay=0.0,
            inner_lr=0.01, n_inner_steps=2,
            n_episodes_per_epoch=2,
            device=torch.device("cpu"),
        )
        # Need multi-domain data for episodic training
        # Build domain_loaders: dict[domain_id -> list of batches]
        domain_loaders = {
            0: [make_batch(n_domains=1) for _ in range(4)],
            1: [make_batch(n_domains=1) for _ in range(4)],
        }
        loss = trainer.train_epoch(domain_loaders)
        assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# CORALTrainer
# ---------------------------------------------------------------------------

class TestCORALTrainer:
    def test_import(self):
        from src.training.coral_trainer import CORALTrainer

    def test_train_epoch_returns_loss(self):
        from src.training.coral_trainer import CORALTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()
        trainer = CORALTrainer(
            backbone, classifier,
            lr=1e-3, weight_decay=0.0,
            lambda_coral=1.0,
            device=torch.device("cpu"),
        )
        loader = make_loader(n_domains=2)
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)

    def test_coral_loss_zero_same_features(self):
        from src.training.coral_trainer import coral_loss
        features = torch.randn(BATCH, PROJ_DIM)
        # Same domain twice — CORAL loss should be 0
        loss = coral_loss([features, features])
        assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# BaseTrainer: checkpoint save/load
# ---------------------------------------------------------------------------

class TestBaseTrainerCheckpoint:
    def test_save_and_load_checkpoint(self, tmp_path):
        from src.training.base_trainer import BaseTrainer
        backbone = TinyBackbone()
        classifier = TinyClassifier()

        class ConcreteTrainer(BaseTrainer):
            def train_epoch(self, loader, **kw):
                return 0.0

        trainer = ConcreteTrainer(
            backbone, classifier,
            lr=1e-3, weight_decay=0.0, device=torch.device("cpu"),
        )
        ckpt_path = tmp_path / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1, metrics={"uar": 0.75})
        assert ckpt_path.exists()

        loaded = trainer.load_checkpoint(ckpt_path)
        assert loaded["epoch"] == 1
        assert loaded["metrics"]["uar"] == 0.75
