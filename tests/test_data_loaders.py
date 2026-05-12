"""RED tests for data loaders — run before implementation exists, expect ImportError or assertion failures."""
import pytest
import numpy as np
import torch
import soundfile as sf
from pathlib import Path


def make_wav(path: Path, duration_s: float = 2.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(duration_s * sr), dtype=np.float32)
    sf.write(str(path), samples, sr)


# ---------------------------------------------------------------------------
# ItalianPDDataset
# ---------------------------------------------------------------------------

class TestItalianPDDataset:
    def test_hc_label_young(self, tmp_path):
        subj = tmp_path / "15 Young Healthy Control" / "Alberto R"
        make_wav(subj / "B1.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path)
        assert len(ds) == 1
        sample = ds[0]
        assert sample["label"] == 0

    def test_hc_label_elderly(self, tmp_path):
        subj = tmp_path / "22 Elderly Healthy Control" / "Maria G"
        make_wav(subj / "VA1.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path)
        sample = ds[0]
        assert sample["label"] == 0

    def test_pd_label(self, tmp_path):
        subj = tmp_path / "28 People with Parkinson's disease" / "1-5" / "Carlo M"
        make_wav(subj / "D1.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path)
        sample = ds[0]
        assert sample["label"] == 1

    def test_waveform_shape(self, tmp_path):
        subj = tmp_path / "15 Young Healthy Control" / "Subj1"
        make_wav(subj / "B1.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path, sample_rate=16000, max_duration=10.0)
        sample = ds[0]
        assert sample["waveform"].shape == (1, 160000)

    def test_domain_id(self, tmp_path):
        subj = tmp_path / "15 Young Healthy Control" / "Subj1"
        make_wav(subj / "B1.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path)
        sample = ds[0]
        assert sample["domain_id"] == 0

    def test_subject_id_present(self, tmp_path):
        subj = tmp_path / "15 Young Healthy Control" / "Alberto R"
        make_wav(subj / "B1.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path)
        sample = ds[0]
        assert sample["subject_id"] == "Alberto R"

    def test_multiple_files_same_subject(self, tmp_path):
        subj = tmp_path / "15 Young Healthy Control" / "Subj1"
        for task in ["B1", "D1", "VA1"]:
            make_wav(subj / f"{task}.wav")
        from src.data.italian_pd import ItalianPDDataset
        ds = ItalianPDDataset(tmp_path)
        assert len(ds) == 3
        assert all(s["subject_id"] == "Subj1" for s in ds)


# ---------------------------------------------------------------------------
# MDVRKCLDataset
# ---------------------------------------------------------------------------

class TestMDVRKCLDataset:
    def test_hc_label(self, tmp_path):
        hc_dir = tmp_path / "ReadText" / "HC"
        make_wav(hc_dir / "ID1_hc_0_0_0.wav")
        from src.data.mdvr_kcl import MDVRKCLDataset
        ds = MDVRKCLDataset(tmp_path)
        sample = ds[0]
        assert sample["label"] == 0

    def test_pd_label(self, tmp_path):
        pd_dir = tmp_path / "ReadText" / "PD"
        make_wav(pd_dir / "ID2_pd_20_1_1.wav")
        from src.data.mdvr_kcl import MDVRKCLDataset
        ds = MDVRKCLDataset(tmp_path)
        sample = ds[0]
        assert sample["label"] == 1

    def test_subject_id_from_filename(self, tmp_path):
        pd_dir = tmp_path / "SpontaneousDialogue" / "PD"
        make_wav(pd_dir / "ID5_pd_30_0_1.wav")
        from src.data.mdvr_kcl import MDVRKCLDataset
        ds = MDVRKCLDataset(tmp_path)
        sample = ds[0]
        assert sample["subject_id"] == "ID5"

    def test_domain_id(self, tmp_path):
        hc_dir = tmp_path / "ReadText" / "HC"
        make_wav(hc_dir / "ID1_hc_0_0_0.wav")
        from src.data.mdvr_kcl import MDVRKCLDataset
        ds = MDVRKCLDataset(tmp_path)
        assert ds[0]["domain_id"] == 1

    def test_waveform_shape(self, tmp_path):
        hc_dir = tmp_path / "ReadText" / "HC"
        make_wav(hc_dir / "ID1_hc_0_0_0.wav")
        from src.data.mdvr_kcl import MDVRKCLDataset
        ds = MDVRKCLDataset(tmp_path, sample_rate=16000, max_duration=10.0)
        assert ds[0]["waveform"].shape == (1, 160000)


# ---------------------------------------------------------------------------
# VoiceSamplesAHDataset
# ---------------------------------------------------------------------------

class TestVoiceSamplesAHDataset:
    def test_hc_label(self, tmp_path):
        hc_dir = tmp_path / "HC_AH" / "HC_AH"
        make_wav(hc_dir / "AH_001-abc123.wav")
        from src.data.voice_samples_ah import VoiceSamplesAHDataset
        ds = VoiceSamplesAHDataset(tmp_path)
        assert ds[0]["label"] == 0

    def test_pd_label(self, tmp_path):
        pd_dir = tmp_path / "PD_AH" / "PD_AH"
        make_wav(pd_dir / "AH_002-def456.wav")
        from src.data.voice_samples_ah import VoiceSamplesAHDataset
        ds = VoiceSamplesAHDataset(tmp_path)
        assert ds[0]["label"] == 1

    def test_subject_id_from_filename(self, tmp_path):
        hc_dir = tmp_path / "HC_AH" / "HC_AH"
        make_wav(hc_dir / "AH_042-xyz789.wav")
        from src.data.voice_samples_ah import VoiceSamplesAHDataset
        ds = VoiceSamplesAHDataset(tmp_path)
        assert ds[0]["subject_id"] == "042"

    def test_domain_id(self, tmp_path):
        hc_dir = tmp_path / "HC_AH" / "HC_AH"
        make_wav(hc_dir / "AH_001-abc.wav")
        from src.data.voice_samples_ah import VoiceSamplesAHDataset
        ds = VoiceSamplesAHDataset(tmp_path)
        assert ds[0]["domain_id"] == 2

    def test_waveform_shape(self, tmp_path):
        hc_dir = tmp_path / "HC_AH" / "HC_AH"
        make_wav(hc_dir / "AH_001-abc.wav")
        from src.data.voice_samples_ah import VoiceSamplesAHDataset
        ds = VoiceSamplesAHDataset(tmp_path, sample_rate=16000, max_duration=10.0)
        assert ds[0]["waveform"].shape == (1, 160000)


# ---------------------------------------------------------------------------
# MultiDomainDataset / subject-level split
# ---------------------------------------------------------------------------

class TestMultiDomainSplit:
    def _build_fake_domain(self, root: Path, domain: str, n_subjects: int, n_files_each: int = 2):
        """Build fake ItalianPD-style structure for testing multi-domain."""
        for i in range(n_subjects):
            subj_dir = root / "15 Young Healthy Control" / f"Subject{i:02d}"
            for j in range(n_files_each):
                make_wav(subj_dir / f"B{j}.wav")

    def test_no_subject_leak_across_train_val(self, tmp_path):
        from src.data.italian_pd import ItalianPDDataset
        from src.data.multi_domain import subject_split

        domain_root = tmp_path / "domain0"
        self._build_fake_domain(domain_root, "italian_pd", n_subjects=10)
        ds = ItalianPDDataset(domain_root)
        train_ids, val_ids = subject_split(ds, train_ratio=0.8, seed=42)
        assert len(train_ids & val_ids) == 0, "Subject leak between train and val"

    def test_split_ratio_approximately_correct(self, tmp_path):
        from src.data.italian_pd import ItalianPDDataset
        from src.data.multi_domain import subject_split

        domain_root = tmp_path / "domain0"
        self._build_fake_domain(domain_root, "italian_pd", n_subjects=10)
        ds = ItalianPDDataset(domain_root)
        train_ids, val_ids = subject_split(ds, train_ratio=0.8, seed=42)
        total = len(train_ids) + len(val_ids)
        assert abs(len(train_ids) / total - 0.8) <= 0.15

    def test_build_loaders_returns_three_loaders(self, tmp_path):
        from src.data.italian_pd import ItalianPDDataset
        from src.data.multi_domain import build_loaders
        from torch.utils.data import DataLoader

        domain_root = tmp_path / "domain0"
        self._build_fake_domain(domain_root, "italian_pd", n_subjects=10)
        ds = ItalianPDDataset(domain_root)
        train_loader, val_loader = build_loaders([ds], held_out_domain=None, batch_size=4, seed=42)
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
