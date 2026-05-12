from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class PDDataset(Dataset, ABC):
    domain_id: int

    def __init__(self, root: Path, sample_rate: int = 16000, max_duration: float = 10.0):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    @abstractmethod
    def _load_samples(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        from src.utils.audio import preprocess_audio
        waveform = preprocess_audio(Path(s["path"]), self.sample_rate, self.max_duration)
        return {
            "waveform": waveform,
            "label": s["label"],
            "domain_id": self.__class__.domain_id,
            "subject_id": s["subject_id"],
            "path": s["path"],
        }

    def get_subject_ids(self) -> List[str]:
        return [s["subject_id"] for s in self.samples]
