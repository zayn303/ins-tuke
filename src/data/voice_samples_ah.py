import re
import sys
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
            double = self.root / outer_name / outer_name
            single = self.root / outer_name
            if double.exists():
                outer = double
            elif single.exists():
                outer = single
            else:
                print(f"WARN VoiceSamplesAH: neither {double} nor {single} exists", file=sys.stderr)
                continue
            for wav in sorted(outer.glob("*.wav")):
                m = _SUBJECT_RE.match(wav.name)
                subject_id = m.group(1) if m else wav.stem
                self.recordings.append({
                    "path": str(wav),
                    "label": label,
                    "subject_id": subject_id,
                    "recording_id": str(wav.relative_to(self.root)),
                    "task_code": "AH",
                })
        if not self.recordings:
            root_contents = [p.name for p in sorted(self.root.iterdir())] if self.root.exists() else ["<root missing>"]
            print(
                f"WARN VoiceSamplesAH: 0 recordings from {self.root}. Root contains: {root_contents}",
                file=sys.stderr,
            )
