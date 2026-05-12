from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import torch

from src.utils.reproducibility import seed_everything
from src.data.italian_pd import ItalianPDDataset
from src.data.mdvr_kcl import MDVRKCLDataset
from src.data.voice_samples_ah import VoiceSamplesAHDataset
from src.data.multi_domain import build_test_loader
from src.models.backbone import SpeechBackbone
from src.models.classifier import PDClassifier
from src.evaluation.evaluator import run_eval


_DOMAIN_CLASSES = [ItalianPDDataset, MDVRKCLDataset, VoiceSamplesAHDataset]
_DOMAIN_PATHS = [
    "Italian_Parkinsons_Voice_and_Speech/italian_parkinson",
    "Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both early and advanced Parkinson's disease patients and healthy controls/Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both",
    "Voice Samples for Patients with Parkinson's Disease and Healthy Controls",
]


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(cfg.data_root)
    domain_idx = cfg.held_out_domain
    cls = _DOMAIN_CLASSES[domain_idx]
    root = data_root / _DOMAIN_PATHS[domain_idx]
    ds = cls(root, sample_rate=cfg.sample_rate, max_duration=cfg.max_duration)
    loader = build_test_loader(ds, batch_size=cfg.batch_size)

    backbone = SpeechBackbone(cfg.model, freeze_backbone=True).to(device)
    classifier = PDClassifier().to(device)

    ckpt_dir = Path(cfg.checkpoint_dir)
    pattern = f"{cfg.method_name}_{cfg.model}_held{cfg.held_out_domain}_best.pt"
    ckpt_path = ckpt_dir / pattern

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    classifier.load_state_dict(ckpt["classifier_state"])

    metrics = run_eval(backbone, classifier, loader, device)
    print(f"Evaluation results (domain={domain_idx}, method={cfg.method_name}, model={cfg.model}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
