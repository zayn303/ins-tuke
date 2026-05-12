from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F


_MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base",
}


def load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != sample_rate:
        waveform = F.resample(waveform, sr, sample_rate)
    return waveform.float()


def load_audio_slice(
    path: Path,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
) -> Tuple[torch.Tensor, int]:
    """Load a slice of a wav at native sample rate, resample, return waveform [1, T] and actual sample count.

    The slice is taken at the native sample rate then resampled, so start/end refer to TARGET-rate samples.
    Returns (waveform, n_samples_actual) where n_samples_actual <= end_sample - start_sample.
    """
    info = sf.info(str(path))
    native_sr = info.samplerate

    if native_sr == sample_rate:
        native_start = start_sample
        native_frames = end_sample - start_sample
    else:
        native_start = int(start_sample * native_sr / sample_rate)
        native_frames = int((end_sample - start_sample) * native_sr / sample_rate)

    data, sr = sf.read(
        str(path),
        dtype="float32",
        start=native_start,
        frames=native_frames,
        always_2d=False,
    )
    if data.ndim > 1:
        data = data.mean(axis=1)

    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != sample_rate:
        waveform = F.resample(waveform, sr, sample_rate)
    return waveform.float(), waveform.shape[1]


@lru_cache(maxsize=8)
def get_feature_extractor(model_name: str):
    from transformers import AutoFeatureExtractor
    model_id = _MODEL_IDS.get(model_name, model_name)
    return AutoFeatureExtractor.from_pretrained(model_id)


def featurise(
    waveform: torch.Tensor,
    sample_rate: int,
    feature_extractor,
    max_samples: int,
) -> Dict[str, torch.Tensor]:
    """Apply HF FeatureExtractor and right-pad to max_samples.

    Input waveform: [1, T] tensor at sample_rate.
    Returns {input_values: [max_samples], attention_mask: [max_samples]} both float32/int64.
    """
    wav_np = waveform.squeeze(0).numpy()
    if wav_np.shape[0] > max_samples:
        wav_np = wav_np[:max_samples]

    out = feature_extractor(
        wav_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=max_samples,
        truncation=True,
        return_attention_mask=True,
    )
    input_values = out["input_values"].squeeze(0)
    attention_mask = out["attention_mask"].squeeze(0).long()
    return {"input_values": input_values, "attention_mask": attention_mask}


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
