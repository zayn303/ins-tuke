import os

import wandb
from omegaconf import OmegaConf


def init_wandb(cfg) -> wandb.run.__class__ | None:
    mode = "offline" if (getattr(cfg, "wandb_offline", False) or os.environ.get("WANDB_MODE") == "offline") else "online"
    try:
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=mode,
        )
    except Exception as e:
        print(f"[wandb] init failed — running without tracking: {e}")
        return None


def log_metrics(run, metrics: dict, step: int = None) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is None:
        return
    run.finish()
