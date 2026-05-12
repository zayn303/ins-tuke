from pathlib import Path
from .base_dataset import PDDataset


class ItalianPDDataset(PDDataset):
    domain_id = 0

    _HC_FRAGMENTS = ("Healthy",)
    _PD_FRAGMENTS = ("Parkinson",)

    def _load_samples(self) -> None:
        for top_dir in sorted(self.root.iterdir()):
            if not top_dir.is_dir():
                continue
            name = top_dir.name
            if any(f in name for f in self._HC_FRAGMENTS):
                label = 0
            elif any(f in name for f in self._PD_FRAGMENTS):
                label = 1
            else:
                continue

            for subdir in sorted(top_dir.rglob("*")):
                if not subdir.is_dir():
                    continue
                wavs = sorted(subdir.glob("*.wav"))
                if not wavs:
                    continue
                subject_id = subdir.name
                for wav in wavs:
                    self.samples.append({
                        "path": str(wav),
                        "label": label,
                        "subject_id": subject_id,
                    })
