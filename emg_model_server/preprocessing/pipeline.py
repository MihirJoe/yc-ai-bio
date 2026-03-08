"""EMG preprocessing pipeline: filtering, windowing, feature extraction."""

import logging
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def bandpass_filter(
    data: np.ndarray,
    low: float,
    high: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Apply butterworth bandpass filter."""
    nyq = sample_rate / 2
    low_norm = max(low / nyq, 0.001)
    high_norm = min(high / nyq, 0.999)
    b, a = scipy_signal.butter(order, [low_norm, high_norm], btype="band")
    if data.ndim == 1:
        return scipy_signal.filtfilt(b, a, data)
    return np.apply_along_axis(
        lambda x: scipy_signal.filtfilt(b, a, x),
        axis=0,
        arr=data,
    )


def notch_filter(data: np.ndarray, freq: float, sample_rate: float, q: float = 30) -> np.ndarray:
    """Apply notch filter (e.g. 50/60 Hz line noise)."""
    w0 = freq / (sample_rate / 2)
    b, a = scipy_signal.iirnotch(w0, q)
    if data.ndim == 1:
        return scipy_signal.filtfilt(b, a, data)
    return np.apply_along_axis(
        lambda x: scipy_signal.filtfilt(b, a, x),
        axis=0,
        arr=data,
    )


def normalize_signal(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize signal (z-score or min-max)."""
    if data.size == 0:
        return data
    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return data - mean
        return (data - mean) / std
    # min-max
    mn, mx = np.min(data), np.max(data)
    if mx - mn < 1e-10:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


def resample(data: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    """Resample to target sample rate."""
    if abs(orig_sr - target_sr) < 0.1:
        return data
    num_samples = int(data.shape[0] * target_sr / orig_sr)
    if data.ndim == 1:
        return scipy_signal.resample(data, num_samples)
    return scipy_signal.resample(data, num_samples, axis=0)


def preprocess_emg(
    data: np.ndarray,
    sample_rate: float,
    *,
    target_sample_rate: int | None = 1000,
    bandpass_low: int = 20,
    bandpass_high: int = 500,
    notch_freq: int | None = 50,
    normalize: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline: resample, bandpass, notch, normalize.

    Args:
        data: EMG (samples,) or (samples, channels)
        sample_rate: Current sample rate
        target_sample_rate: Target rate (None = skip resample)
        bandpass_low: Low cutoff Hz
        bandpass_high: High cutoff Hz
        notch_freq: Line noise Hz (None = skip notch)
        normalize: Apply z-score normalization

    Returns:
        Preprocessed EMG
    """
    out = np.asarray(data, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1)

    sr = float(sample_rate)
    if target_sample_rate and abs(sr - target_sample_rate) > 1:
        out = resample(out, sr, target_sample_rate)
        sr = float(target_sample_rate)

    out = bandpass_filter(out, bandpass_low, bandpass_high, sr)
    if notch_freq is not None:
        out = notch_filter(out, notch_freq, sr)

    if normalize:
        out = normalize_signal(out, "zscore")

    return out


def window_signal(
    data: np.ndarray,
    sample_rate: float,
    window_size_ms: float = 200,
    overlap_ratio: float = 0.5,
) -> list[np.ndarray]:
    """
    Split signal into overlapping windows.

    Args:
        data: (samples,) or (samples, channels)
        sample_rate: Hz
        window_size_ms: Window length in ms
        overlap_ratio: 0-1, overlap proportion

    Returns:
        List of window arrays
    """
    if data.size == 0:
        return []
    win_len = int(window_size_ms / 1000 * sample_rate)
    if win_len < 1:
        win_len = 1
    step = max(1, int(win_len * (1 - overlap_ratio)))
    windows = []
    for start in range(0, data.shape[0] - win_len + 1, step):
        windows.append(data[start : start + win_len])
    return windows


def rms(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """Root mean square."""
    return np.sqrt(np.mean(sig.astype(float) ** 2, axis=axis))


def mav(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean absolute value."""
    return np.mean(np.abs(sig.astype(float)), axis=axis)


def waveform_length(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """Waveform length (sum of absolute differences)."""
    return np.sum(np.abs(np.diff(sig.astype(float), axis=axis)), axis=axis)


def zero_crossing_rate(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """Zero crossing rate."""
    s = np.sign(sig)
    zc = np.sum(np.abs(np.diff(s, axis=axis)), axis=axis) / 2
    n = sig.shape[axis] - 1
    return zc / max(n, 1)


def median_frequency_proxy(sig: np.ndarray, sample_rate: float, axis: int = -1) -> np.ndarray:
    """
    Proxy for median frequency using spectral centroid.
    Returns approximate median frequency in Hz.
    """
    sig = np.asarray(sig, dtype=float)
    if axis == -1:
        axis = sig.ndim - 1
    n = sig.shape[axis]
    fft_vals = np.abs(np.fft.rfft(sig, axis=axis))
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    # Broadcast freqs for summation
    for _ in range(fft_vals.ndim - 1):
        freqs = np.expand_dims(freqs, axis=-1)
    total = np.sum(fft_vals, axis=axis, keepdims=True)
    total = np.where(total < 1e-12, 1, total)
    centroid = np.squeeze(np.sum(fft_vals * freqs, axis=axis, keepdims=True) / total)
    return centroid if np.isscalar(centroid) else np.asarray(centroid)


def extract_features(
    sig: np.ndarray,
    sample_rate: float,
    channel: int = 0,
) -> dict[str, float]:
    """
    Extract common EMG features from a window.

    Args:
        sig: (samples,) or (samples, channels)
        sample_rate: Hz
        channel: Channel index if multi-channel

    Returns:
        Dict of feature name -> value
    """
    if sig.ndim > 1:
        sig = sig[:, channel]
    sig = np.asarray(sig, dtype=float).flatten()
    if sig.size == 0:
        return {
            "rms": 0.0,
            "mav": 0.0,
            "waveform_length": 0.0,
            "zero_crossing_rate": 0.0,
            "median_freq_proxy": 0.0,
        }
    return {
        "rms": float(rms(sig)),
        "mav": float(mav(sig)),
        "waveform_length": float(waveform_length(sig)),
        "zero_crossing_rate": float(zero_crossing_rate(sig)),
        "median_freq_proxy": float(median_frequency_proxy(sig, sample_rate)),
    }
