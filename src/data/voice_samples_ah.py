import re
from pathlib import Path
from .base_dataset import PDDataset

_SUBJECT_RE = re.compile(r"^AH_([^_]+)_")


class VoiceSamplesAHDataset(PDDataset):
    domain_id = 2

    def _load_samples(self) -> None:
        label_dirs = {
            "HC_AH": 0,
            "PD_AH": 1,
        }
        for outer_name, label in label_dirs.items():
            outer = self.root / outer_name / outer_name
            if not outer.exists():
                continue
            for wav in sorted(outer.glob("*.wav")):
                m = _SUBJECT_RE.match(wav.name)
                subject_id = m.group(1) if m else wav.stem
                self.samples.append({
                    "path": str(wav),
                    "label": label,
                    "subject_id": subject_id,
                })
