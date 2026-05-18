import sys
sys.path.insert(0, '.')
from src.data.italian_pd import ItalianPDDataset
from src.data.neurovoz import NeurovozDataset
from src.data.pc_gita import PCGITADataset
from pathlib import Path
from collections import Counter

for cls, path in [
    (ItalianPDDataset, "Data/Italian_Parkinsons_Voice_and_Speech/italian_parkinson"),
    (NeurovozDataset, "Data/neurovoz_v3/data/audios"),
    (PCGITADataset, "Data/PC-GITA_per_task_44100Hz"),
]:
    ds = cls.__new__(cls)
    ds.root = Path(path); ds.recordings = []
    ds._load_samples()
    n_before = len(ds.recordings)
    ds._apply_task_filter()
    n_after = len(ds.recordings)
    codes = Counter(r["task_code"] for r in ds.recordings)
    print(f"\n{cls.__name__}: {n_after} kept / {n_before} total (filtered {n_before - n_after})")
    for k, v in codes.most_common():
        print(f"  {k}: {v}")
