"""Load EMG data from various file formats."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_emg_from_path(
    path: str | Path,
    sample_rate: int = 1000,
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load EMG data from CSV, NPY, or NPZ file.

    Args:
        path: Path to file
        sample_rate: Default sample rate if not inferrable
        channel_names: Optional channel names

    Returns:
        (data, sample_rate) - data as (samples,) or (samples, channels)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EMG file not found: {path}")

    suffix = path.suffix.lower()
    data: np.ndarray
    sr = sample_rate

    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        obj = np.load(path)
        keys = list(obj.keys())
        if "emg" in keys:
            data = obj["emg"]
        elif "data" in keys:
            data = obj["data"]
        elif keys:
            data = obj[keys[0]]
        else:
            raise ValueError(f"No data arrays in NPZ: {path}")
        if "sample_rate" in keys or "sr" in keys:
            sr = int(obj.get("sample_rate", obj.get("sr", sample_rate)))
    elif suffix in (".csv", ".txt"):
        data = np.genfromtxt(path, delimiter=",", skip_header=0, filling_values=0)
        if data.ndim == 1:
            data = data[:, np.newaxis] if data.size > 0 else data.reshape(0, 1)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim == 2 and data.shape[1] > 1:
        pass
    else:
        data = np.atleast_2d(data.T).T

    return data, sr


def load_emg(
    source: np.ndarray | str | Path,
    sample_rate: int = 1000,
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load EMG from array or file path.

    Args:
        source: numpy array or path
        sample_rate: Sample rate (used if source is array or not in file)
        channel_names: Optional channel names

    Returns:
        (data, sample_rate)
    """
    if isinstance(source, (str, Path)):
        return load_emg_from_path(source, sample_rate, channel_names)
    arr = np.asarray(source)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, sample_rate
