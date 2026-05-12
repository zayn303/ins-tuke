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
            # Try single-level first (root/HC_AH/*.wav), then double-level as fallback
            outer = self.root / outer_name
            if not outer.exists():
                outer = self.root / outer_name / outer_name
            if not outer.exists():
                print(f"WARN VoiceSamplesAH: neither {self.root / outer_name} "
                      f"nor {self.root / outer_name / outer_name} exists", file=sys.stderr)
                continue
            for wav in sorted(outer.glob("*.wav")):
                m = _SUBJECT_RE.match(wav.name)
                subject_id = m.group(1) if m else wav.stem
                self.samples.append({
                    "path": str(wav),
                    "label": label,
                    "subject_id": subject_id,
                })
        if not self.samples:
            print(f"WARN VoiceSamplesAH: 0 samples loaded from {self.root}. "
                  f"Expected HC_AH/ and PD_AH/ subdirs with *.wav files.", file=sys.stderr)
