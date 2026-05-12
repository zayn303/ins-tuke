from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F


def load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    if sr != sample_rate:
        waveform = F.resample(waveform, sr, sample_rate)
    return waveform.float()


def pad_or_trim(waveform: torch.Tensor, max_samples: int) -> torch.Tensor:
    t = waveform.shape[1]
    if t < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - t))
    elif t > max_samples:
        waveform = waveform[:, :max_samples]
    return waveform


def normalise(waveform: torch.Tensor) -> torch.Tensor:
    return waveform / (waveform.abs().max() + 1e-8)


def preprocess_audio(
    path: Path,
    sample_rate: int = 16000,
    max_duration: float = 10.0,
) -> torch.Tensor:
    max_samples = int(sample_rate * max_duration)
    waveform = load_audio(path, sample_rate)
    waveform = pad_or_trim(waveform, max_samples)
    return normalise(waveform)
