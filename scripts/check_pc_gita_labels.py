#!/usr/bin/env python3
"""Diagnose PC-GITA label distribution without loading audio (no soundfile probing).

Run: python scripts/check_pc_gita_labels.py [data_root]
Default data_root: Data/PC-GITA_per_task_44100Hz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pc_gita import _HC_TOKENS, _PD_TOKENS, _extract_label

DEFAULT_ROOT = Path("Data/PC-GITA_per_task_44100Hz")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    nested = root / root.name
    if nested.is_dir():
        root = nested

    if not root.exists():
        print(f"ERROR: data root not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {root}")
    print(f"HC tokens: {sorted(_HC_TOKENS)}")
    print(f"PD tokens: {sorted(_PD_TOKENS)}")
    print()

    n_hc = n_pd = n_unknown = 0
    unknown_examples: list[str] = []

    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root)
        parts = rel.parts
        if any(p.startswith("__") for p in parts[:-1]):
            continue
        dir_parts = parts[:-1]
        label = _extract_label(dir_parts)
        if label is None:
            n_unknown += 1
            if len(unknown_examples) < 10:
                unknown_examples.append(str(rel))
        elif label == 0:
            n_hc += 1
        else:
            n_pd += 1

    n_total = n_hc + n_pd + n_unknown
    print(f"Total wav files (non-__): {n_total}")
    print(f"  HC (label=0): {n_hc}")
    print(f"  PD (label=1): {n_pd}")
    print(f"  Unknown label: {n_unknown}")
    if n_total > 0:
        print(f"  Skip rate: {100 * n_unknown / n_total:.1f}%")

    if unknown_examples:
        print(f"\nUnknown-label paths (first {len(unknown_examples)}):")
        for ex in unknown_examples:
            print(f"  {ex}")

    print()
    if n_hc == 0 or n_pd == 0:
        print("WARNING: one class has 0 files — label mapping is broken")
    elif n_hc > n_pd * 3 or n_pd > n_hc * 3:
        print(f"WARNING: severe class imbalance (HC={n_hc}, PD={n_pd}) — possible label inversion")
    else:
        print(f"OK: balanced classes (HC={n_hc}, PD={n_pd}), skip={n_unknown}")

    if n_total > 0 and n_unknown / n_total > 0.10:
        print(f"ERROR: skip rate {100 * n_unknown / n_total:.1f}% > 10% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
