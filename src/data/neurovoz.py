import csv
import re
import sys
from pathlib import Path
from typing import Dict, Optional

from .base_dataset import PDDataset


_SUBJECT_ID_RE = re.compile(r"^\d{4}$")


def _parse_filename(name: str):
    """Parse {LABEL}_{TASK}_{ID4}.wav where TASK may contain underscores.

    Returns (label_name, task_code, subject_id) or None.
    """
    stem = name[:-4] if name.endswith(".wav") else name
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    label_name = parts[0]
    if label_name not in ("HC", "PD"):
        return None
    subject_id = parts[-1]
    if not _SUBJECT_ID_RE.match(subject_id):
        return None
    task_code = "_".join(parts[1:-1])
    if not task_code:
        return None
    return label_name, task_code, subject_id


class NeurovozDataset(PDDataset):
    domain_id = 3

    _METADATA_REL = {
        "HC": "../metadata/metadata_hc.csv",
        "PD": "../metadata/metadata_pd.csv",
    }

    def _load_samples(self) -> None:
        metadata_by_basename = self._load_metadata()

        n_unparsed = 0
        for wav in sorted(self.root.glob("*.wav")):
            parsed = _parse_filename(wav.name)
            if parsed is None:
                print(f"WARN NeurovozDataset: unparsable filename {wav.name}", file=sys.stderr)
                n_unparsed += 1
                continue
            label_name, task_code, subject_id = parsed
            label = 0 if label_name == "HC" else 1
            row = metadata_by_basename.get(wav.name)
            self.recordings.append({
                "path": str(wav),
                "label": label,
                "subject_id": subject_id,
                "recording_id": wav.stem,
                "task_code": task_code,
                "metadata": row,
            })

        if n_unparsed:
            print(f"WARN NeurovozDataset: {n_unparsed} files had unparsable names", file=sys.stderr)

        self._check_subject_id_disjointness()
        self._warn_missing_metadata()

    def _load_metadata(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for label_name, rel in self._METADATA_REL.items():
            csv_path = (self.root / rel).resolve()
            if not csv_path.exists():
                print(
                    f"WARN NeurovozDataset: metadata file missing {csv_path}",
                    file=sys.stderr,
                )
                continue
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    audio = row.get("Audio") or ""
                    basename = Path(audio.replace("\\", "/")).name
                    if basename:
                        result[basename] = row
        return result

    def _check_subject_id_disjointness(self) -> None:
        hc_ids = {r["subject_id"] for r in self.recordings if r["label"] == 0}
        pd_ids = {r["subject_id"] for r in self.recordings if r["label"] == 1}
        collisions = hc_ids & pd_ids
        if not collisions:
            return
        print(
            f"WARN NeurovozDataset: HC/PD subject ID collision ({len(collisions)} IDs) - prefixing",
            file=sys.stderr,
        )
        for r in self.recordings:
            prefix = "HC" if r["label"] == 0 else "PD"
            r["subject_id"] = f"{prefix}_{r['subject_id']}"

    def _warn_missing_metadata(self) -> None:
        n_missing = sum(1 for r in self.recordings if r.get("metadata") is None)
        if n_missing:
            print(
                f"WARN NeurovozDataset: {n_missing} recordings missing metadata",
                file=sys.stderr,
            )
