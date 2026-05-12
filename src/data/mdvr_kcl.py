import re
from pathlib import Path
from .base_dataset import PDDataset

_SUBJECT_RE = re.compile(r"^(ID\d+)_")


class MDVRKCLDataset(PDDataset):
    domain_id = 1

    def _load_samples(self) -> None:
        for split_dir in sorted(self.root.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name  # "ReadText" or "SpontaneousDialogue"
            for label_dir in sorted(split_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                name = label_dir.name.upper()
                if name == "HC":
                    label = 0
                elif name == "PD":
                    label = 1
                else:
                    continue
                for wav in sorted(label_dir.glob("*.wav")):
                    m = _SUBJECT_RE.match(wav.name)
                    subject_id = m.group(1) if m else wav.stem
                    self.recordings.append({
                        "path": str(wav),
                        "label": label,
                        "subject_id": subject_id,
                        "recording_id": str(wav.relative_to(self.root)),
                        "task_code": split_name,
                    })
