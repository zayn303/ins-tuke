"""Verify all dataset roots exist and load expected samples + segments.

Run on login node before submitting SLURM:
    python scripts/precheck_data.py
    python scripts/precheck_data.py /custom/data/root
    python scripts/precheck_data.py --check-hf
    python scripts/precheck_data.py /custom/data/root --check-hf

Exits 1 if any domain has missing root, 0 samples, or (with --check-hf) missing HF cache.
"""
import argparse
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.italian_pd import ItalianPDDataset
from src.data.mdvr_kcl import MDVRKCLDataset
from src.data.voice_samples_ah import VoiceSamplesAHDataset
from src.data.neurovoz import NeurovozDataset


_DOMAIN_CLASSES = [ItalianPDDataset, MDVRKCLDataset, VoiceSamplesAHDataset, NeurovozDataset]
_DOMAIN_PATHS = [
    "Italian_Parkinsons_Voice_and_Speech/italian_parkinson",
    "Mobile Device Voice Recordings at Kings College London (MDVR-KCL) from both early and advanced Parkinsons disease patients and healthy controls/Mobile Device Voice Recordings at Kings College London (MDVR-KCL) from both",
    "Voice Samples for Patients with Parkinsons Disease and Healthy Controls",
    "neurovoz_v3/data/audios",
]


_NEUROVOZ_VOWELS = {
    "A1", "A2", "A3", "E1", "E2", "E3",
    "I1", "I2", "I3", "O1", "O2", "O3",
    "U1", "U2", "U3",
}
_NEUROVOZ_DDK = {"PATAKA"}
_NEUROVOZ_FREE = {"FREE"}


def _neurovoz_task_breakdown(recordings):
    from collections import Counter
    counts = Counter()
    for r in recordings:
        t = r.get("task_code", "")
        if t in _NEUROVOZ_VOWELS:
            counts["vowel"] += 1
        elif t in _NEUROVOZ_DDK:
            counts["ddk"] += 1
        elif t in _NEUROVOZ_FREE:
            counts["free"] += 1
        else:
            counts["word"] += 1
    return dict(counts)


def check_hf_cache() -> int:
    from src.utils.audio import _MODEL_IDS
    from transformers import AutoFeatureExtractor

    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        print("WARN: HF_HOME not set; using default HuggingFace cache location")

    failed = False
    for name, model_id in _MODEL_IDS.items():
        try:
            AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        except Exception as exc:
            print(f"FAIL HF cache: {model_id} — {exc}")
            failed = True

    if failed:
        return 1

    cache_root = hf_home or "~/.cache/huggingface"
    models_str = " ".join(_MODEL_IDS.values())
    print(f"OK HF cache: {models_str} at {cache_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("data_root", nargs="?", default="/home/ak562fx/ins-tuke/Data")
    parser.add_argument("--check-hf", action="store_true")
    args, _ = parser.parse_known_args()

    data_root = Path(args.data_root)
    print(f"data_root: {data_root}")
    print(f"data_root.exists(): {data_root.exists()}")
    if not data_root.exists():
        print("FAIL: data_root missing")
        return 1
    print("data_root contents:")
    for p in sorted(data_root.iterdir()):
        print(f"  {p.name}")
    print()

    fail = False
    for cls, rel in zip(_DOMAIN_CLASSES, _DOMAIN_PATHS):
        root = data_root / rel
        print(f"=== {cls.__name__} (domain_id={cls.domain_id}) ===")
        print(f"  rel:    {rel}")
        print(f"  root:   {root}")
        print(f"  exists: {root.exists()}")
        if not root.exists():
            print("  FAIL: root missing")
            fail = True
            print()
            continue
        ds = cls(root, sample_rate=16000, max_duration=10.0)
        n_rec = len(ds.recordings)
        n_seg = len(ds.segments)
        print(f"  n_recordings: {n_rec}")
        print(f"  n_segments:   {n_seg}")
        if n_rec == 0:
            print("  FAIL: 0 recordings")
            fail = True
            print()
            continue
        n_pd = sum(r["label"] for r in ds.recordings)
        n_hc = n_rec - n_pd
        subjects = len({r["subject_id"] for r in ds.recordings})
        print(f"  HC={n_hc}  PD={n_pd}  subjects={subjects}")

        seg_per_rec = Counter(s["recording_id"] for s in ds.segments)
        counts = list(seg_per_rec.values())
        if counts:
            print(
                f"  seg-per-rec: median={statistics.median(counts)} "
                f"min={min(counts)} max={max(counts)}"
            )

        task_counts = Counter(r.get("task_code", "") for r in ds.recordings)
        print(f"  tasks: {dict(task_counts)}")

        if cls.__name__ == "NeurovozDataset":
            breakdown = _neurovoz_task_breakdown(ds.recordings)
            print(
                f"  task breakdown: "
                f"vowel={breakdown.get('vowel', 0)}  "
                f"word={breakdown.get('word', 0)}  "
                f"ddk={breakdown.get('ddk', 0)}  "
                f"free={breakdown.get('free', 0)}"
            )

        print(f"  first sample: {ds.recordings[0]['path']}")
        print(f"  last sample:  {ds.recordings[-1]['path']}")
        print()

    if fail:
        print("PRECHECK FAILED")
        return 1
    print("PRECHECK PASSED")

    if args.check_hf:
        return check_hf_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
