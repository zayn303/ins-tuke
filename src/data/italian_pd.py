import re
from pathlib import Path
from .base_dataset import PDDataset


_TASK_PREFIX_RE = re.compile(r"^([A-Z]+)\d")


def _task_code(name: str) -> str:
    m = _TASK_PREFIX_RE.match(name)
    return m.group(1) if m else ""


class ItalianPDDataset(PDDataset):
    domain_id = 0
    EXCLUDED_TASKS = frozenset({"VA", "VE", "VI", "VO", "VU"})

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
                    self.recordings.append({
                        "path": str(wav),
                        "label": label,
                        "subject_id": subject_id,
                        "recording_id": str(wav.relative_to(self.root)),
                        "task_code": _task_code(wav.name),
                    })
