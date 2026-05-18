import sys
from pathlib import Path

from .base_dataset import PDDataset

_HC_TOKENS = frozenset(("hc", "control"))
_PD_TOKENS = frozenset(("pd", "patologica"))


def _extract_label(parts):
    for part in parts:
        # match exact folder name or prefix before _ (e.g., "HC_las que sobraron")
        first = part.lower().split("_")[0]
        if first in _HC_TOKENS:
            return 0
        if first in _PD_TOKENS:
            return 1
    return None


def _task_code(parts):
    cat = parts[0].lower()
    if cat == "ddk analysis":
        # DDK analysis/{sub_task}/sin normalizar/{label}/filename
        return parts[1]
    if cat == "modulated vowels":
        # modulated vowels/{label}/{vowel}/filename
        return f"vowel_{parts[2]}"
    if cat == "words":
        # Words/Sin normalizar/{label}/{word}/filename
        return parts[3] if len(parts) > 3 else parts[-2]
    if cat == "sentences":
        # sentences/{name}/sin normalizar/{label}/filename
        return f"sentence_{parts[1]}"
    if cat == "sentences2":
        # sentences2/{name}/non-normalized/{label}/filename
        return f"sentence2_{parts[1]}"
    if cat == "monologue":
        return "monologue"
    if cat == "read text":
        return "read_text"
    # fallback: immediate parent folder above the file
    return parts[-2] if len(parts) >= 2 else ""


class PCGITADataset(PDDataset):
    domain_id = 4
    EXCLUDED_TASKS = frozenset({"vowel_A", "vowel_E", "vowel_I", "vowel_O", "vowel_U"})

    def _load_samples(self) -> None:
        root = self.root
        # Double-nested root: Data/PC-GITA_per_task_44100Hz/PC-GITA_per_task_44100Hz/
        nested = root / root.name
        if nested.is_dir():
            root = nested
            self.root = root

        n_unknown = 0
        for wav in sorted(root.rglob("*.wav")):
            rel = wav.relative_to(root)
            parts = rel.parts  # all parts including filename

            # Skip paths containing dirs starting with __
            if any(p.startswith("__") for p in parts[:-1]):
                continue

            dir_parts = parts[:-1]  # directory parts only

            label = _extract_label(dir_parts)
            if label is None:
                print(
                    f"WARN PCGITADataset: unknown label in path {rel}",
                    file=sys.stderr,
                )
                n_unknown += 1
                continue

            subject_id = wav.stem.split("_")[0]
            code = _task_code(parts)

            self.recordings.append({
                "path": str(wav),
                "label": label,
                "subject_id": subject_id,
                "recording_id": str(rel),
                "task_code": code,
            })

        if n_unknown:
            print(
                f"WARN PCGITADataset: {n_unknown} files skipped (unknown label)",
                file=sys.stderr,
            )
        n_total_seen = len(self.recordings) + n_unknown
        if n_total_seen > 0 and n_unknown / n_total_seen > 0.10:
            raise RuntimeError(
                f"PCGITADataset: {n_unknown}/{n_total_seen} files "
                f"({100 * n_unknown / n_total_seen:.1f}%) have unknown labels — "
                f"check _HC_TOKENS/_PD_TOKENS in src/data/pc_gita.py"
            )
