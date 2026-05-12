"""Verify all dataset roots exist and load expected samples + segments.

Run on login node before submitting SLURM:
    python scripts/precheck_data.py
    python scripts/precheck_data.py /custom/data/root

Exits 1 if any domain has missing root or 0 samples.
"""
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.italian_pd import ItalianPDDataset
from src.data.mdvr_kcl import MDVRKCLDataset
from src.data.voice_samples_ah import VoiceSamplesAHDataset


_DOMAIN_CLASSES = [ItalianPDDataset, MDVRKCLDataset, VoiceSamplesAHDataset]
_DOMAIN_PATHS = [
    "Italian_Parkinsons_Voice_and_Speech/italian_parkinson",
    "Mobile Device Voice Recordings at Kings College London (MDVR-KCL) from both early and advanced Parkinsons disease patients and healthy controls/Mobile Device Voice Recordings at Kings College London (MDVR-KCL) from both",
    "Voice Samples for Patients with Parkinsons Disease and Healthy Controls",
]


def main() -> int:
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ak562fx/ins-tuke/Data")
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

        print(f"  first sample: {ds.recordings[0]['path']}")
        print(f"  last sample:  {ds.recordings[-1]['path']}")
        print()

    if fail:
        print("PRECHECK FAILED")
        return 1
    print("PRECHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
