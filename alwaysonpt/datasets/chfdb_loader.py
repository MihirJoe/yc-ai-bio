"""
Loader for PhysioNet BIDMC Congestive Heart Failure ECG database (chfdb).
15 subjects with severe CHF (NYHA 3-4), ~20hr 2-channel ECGs at 250Hz.
https://physionet.org/content/chfdb/1.0.0/
"""

import numpy as np
from pathlib import Path
from alwaysonpt.datasets.base import BioSignalRecord

FS = 250
WINDOW_SECONDS = 30
WINDOWS_PER_PATIENT = 5


def load_chfdb(data_dir: str, max_records: int = None,
               window_s: int = WINDOW_SECONDS,
               windows_per_patient: int = WINDOWS_PER_PATIENT) -> list:
    """
    Load chfdb dataset. Extracts fixed-length windows from long recordings.

    Each recording is ~20 hours. We extract `windows_per_patient` windows
    of `window_s` seconds each, evenly spaced through the recording.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"chfdb data not found: {data_dir}")

    records = []
    patient_ids = _find_patients(data_path)

    for pid in patient_ids:
        signal_data = _read_record(data_path, pid)
        if signal_data is None:
            continue

        n_samples = signal_data.shape[0]
        window_samples = window_s * FS

        if n_samples < window_samples:
            continue

        spacing = (n_samples - window_samples) // max(1, windows_per_patient - 1)
        offsets = [i * spacing for i in range(windows_per_patient)]

        for wi, offset in enumerate(offsets):
            end = offset + window_samples
            if end > n_samples:
                break

            window = signal_data[offset:end]
            signals = {'ecg_ch1': window[:, 0]}
            if window.shape[1] > 1:
                signals['ecg_ch2'] = window[:, 1]

            records.append(BioSignalRecord(
                record_id=f"chfdb_{pid}_w{wi}",
                domain='ecg_2ch',
                signals=signals,
                fs=FS,
                duration_s=float(window_s),
                metadata={
                    'patient_id': pid,
                    'window_index': wi,
                    'offset_s': float(offset / FS),
                },
                ground_truth={
                    'nyha': 4,  # all chfdb patients are NYHA 3-4 (severe)
                    'condition': 'chf',
                },
            ))

        if max_records and len(records) >= max_records:
            break

    if max_records:
        records = records[:max_records]
    return records


def _find_patients(data_path: Path) -> list:
    """Find patient record IDs (chf01, chf02, ...)."""
    patient_ids = set()
    for f in data_path.iterdir():
        if f.suffix in ('.dat', '.hea'):
            patient_ids.add(f.stem)
    return sorted(patient_ids)


def _read_record(data_path: Path, record_id: str) -> np.ndarray:
    """Read a WFDB record, return (n_samples, n_channels) array."""
    try:
        import wfdb
        record = wfdb.rdrecord(str(data_path / record_id))
        return record.p_signal
    except ImportError:
        dat_path = data_path / f"{record_id}.dat"
        if dat_path.exists():
            try:
                data = np.fromfile(str(dat_path), dtype=np.int16)
                data = data.reshape(-1, 2).astype(np.float64)
                return data / 200.0
            except Exception:
                pass
    except Exception:
        pass
    return None
