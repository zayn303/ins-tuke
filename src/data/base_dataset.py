from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import soundfile as sf
import torch
from torch.utils.data import Dataset


class PDDataset(Dataset, ABC):
    domain_id: int
    EXCLUDED_TASKS: frozenset = frozenset()

    def __init__(
        self,
        root: Path,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        segment_seconds: float = 10.0,
        segment_stride_seconds: float = 5.0,
        min_segment_seconds: float = 1.0,
        model_name: str = "wav2vec2",
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.segment_seconds = segment_seconds
        self.segment_stride_seconds = segment_stride_seconds
        self.min_segment_seconds = min_segment_seconds
        self.model_name = model_name

        self.recordings: List[Dict[str, Any]] = []
        self.segments: List[Dict[str, Any]] = []

        self._load_samples()
        self._apply_task_filter()
        self._build_segment_manifest()

    def _apply_task_filter(self) -> None:
        if self.EXCLUDED_TASKS:
            self.recordings = [
                r for r in self.recordings
                if r.get("task_code", "") not in self.EXCLUDED_TASKS
            ]

    @abstractmethod
    def _load_samples(self) -> None:
        """Subclasses populate self.recordings with dicts containing:
        path, label, subject_id, recording_id, task_code (optional).
        """
        pass

    def _build_segment_manifest(self) -> None:
        max_samples = int(self.segment_seconds * self.sample_rate)
        stride_samples = int(self.segment_stride_seconds * self.sample_rate)
        min_samples = int(self.min_segment_seconds * self.sample_rate)

        n_skipped = 0
        for rec in self.recordings:
            try:
                info = sf.info(str(rec["path"]))
                duration_s = info.frames / info.samplerate
            except Exception as e:
                print(f"WARN PDDataset: cannot probe {rec['path']}: {e}", file=sys.stderr)
                n_skipped += 1
                continue

            total_samples = int(duration_s * self.sample_rate)

            if total_samples <= max_samples:
                self.segments.append({
                    "path": rec["path"],
                    "label": rec["label"],
                    "subject_id": rec["subject_id"],
                    "recording_id": rec.get("recording_id", rec["path"]),
                    "task_code": rec.get("task_code", ""),
                    "domain_id": self.__class__.domain_id,
                    "start_sample": 0,
                    "end_sample": total_samples,
                    "is_partial": total_samples < max_samples,
                })
                continue

            start = 0
            while start < total_samples:
                end = min(start + max_samples, total_samples)
                if (end - start) < min_samples:
                    break
                self.segments.append({
                    "path": rec["path"],
                    "label": rec["label"],
                    "subject_id": rec["subject_id"],
                    "recording_id": rec.get("recording_id", rec["path"]),
                    "task_code": rec.get("task_code", ""),
                    "domain_id": self.__class__.domain_id,
                    "start_sample": start,
                    "end_sample": end,
                    "is_partial": (end - start) < max_samples,
                })
                if end >= total_samples:
                    break
                start += stride_samples

        if n_skipped:
            print(
                f"WARN PDDataset: skipped {n_skipped} recordings due to probe errors",
                file=sys.stderr,
            )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from src.utils.audio import load_audio_slice, get_feature_extractor, featurise

        s = self.segments[idx]
        waveform, _ = load_audio_slice(
            Path(s["path"]),
            self.sample_rate,
            s["start_sample"],
            s["end_sample"],
        )
        max_samples = int(self.segment_seconds * self.sample_rate)
        fe = get_feature_extractor(self.model_name)
        feats = featurise(waveform, self.sample_rate, fe, max_samples)

        return {
            "input_values": feats["input_values"],
            "attention_mask": feats["attention_mask"],
            "label": s["label"],
            "domain_id": s["domain_id"],
            "subject_id": s["subject_id"],
            "recording_id": s["recording_id"],
            "task_code": s["task_code"],
            "path": s["path"],
        }

    def get_subject_ids(self) -> List[str]:
        """Returns one subject_id per segment (used by subject_split)."""
        return [s["subject_id"] for s in self.segments]

    @property
    def samples(self):
        """Backwards-compat alias for code that still references .samples."""
        return self.segments
