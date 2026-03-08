"""
Loader for PhysioNet CHF RR Interval Database (chf2db).
29 subjects with CHF, NYHA classes I/II/III. Beat annotation files (RR intervals).
https://physionet.org/content/chf2db/1.0.0/
"""

import re
import numpy as np
from pathlib import Path
from alwaysonpt.datasets.base import BioSignalRecord

NYHA_MAP = {
    range(201, 213): 1,    # chf201-chf212: NYHA I
    range(213, 221): 2,    # chf213-chf220: NYHA II
    range(221, 230): 3,    # chf221-chf229: NYHA III
}


def load_chf2db(data_dir: str, max_records: int = None,
                max_beats: int = 50000) -> list:
    """
    Load chf2db dataset. Extracts RR intervals from beat annotation files.

    The database contains beat annotations (not raw ECG).
    We compute RR intervals from successive beat times.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"chf2db data not found: {data_dir}")

    records = []
    patient_files = _find_annotation_files(data_path)

    for pid, ann_path in patient_files:
        rr_intervals = _extract_rr_intervals(data_path, pid, ann_path, max_beats)
        if rr_intervals is None or len(rr_intervals) < 100:
            continue

        nyha = _get_nyha_class(pid)
        duration_s = float(np.sum(rr_intervals))

        records.append(BioSignalRecord(
            record_id=f"chf2db_{pid}",
            domain='rr_interval',
            signals={'rr_intervals': rr_intervals},
            fs=0,  # event-based, not uniformly sampled
            duration_s=duration_s,
            metadata={
                'patient_id': pid,
                'n_beats': len(rr_intervals),
            },
            ground_truth={
                'nyha': nyha,
                'condition': 'chf',
            },
        ))

        if max_records and len(records) >= max_records:
            break

    return records


def _find_annotation_files(data_path: Path) -> list:
    """Find patient IDs and their annotation files."""
    patients = []
    seen = set()

    for f in sorted(data_path.iterdir()):
        if f.suffix in ('.ecg', '.atr'):
            pid = f.stem
            if pid not in seen:
                seen.add(pid)
                patients.append((pid, f))

    if not patients:
        for f in sorted(data_path.iterdir()):
            if f.suffix == '.hea':
                pid = f.stem
                if pid not in seen and pid.startswith('chf'):
                    seen.add(pid)
                    patients.append((pid, f))

    return patients


def _extract_rr_intervals(data_path: Path, pid: str,
                           ann_path: Path, max_beats: int) -> np.ndarray:
    """Extract RR intervals from annotation file."""
    try:
        import wfdb
        ann = wfdb.rdann(str(data_path / pid), 'ecg')
        samples = ann.sample[:max_beats]
        rr = np.diff(samples) / 128.0  # 128 Hz original sampling
        rr = rr[(rr > 0.2) & (rr < 3.0)]  # filter physiological range
        return rr
    except ImportError:
        pass
    except Exception:
        pass

    ecg_path = data_path / f"{pid}.ecg"
    if ecg_path.exists():
        try:
            with open(ecg_path, 'rb') as f:
                raw = np.frombuffer(f.read(), dtype=np.uint32)
            samples = raw[:max_beats]
            rr = np.diff(samples.astype(np.float64)) / 128.0
            rr = rr[(rr > 0.2) & (rr < 3.0)]
            return rr
        except Exception:
            pass

    return None


def _get_nyha_class(pid: str) -> int:
    """Determine NYHA class from patient ID number."""
    match = re.search(r'(\d+)', pid)
    if not match:
        return 2  # default
    num = int(match.group(1))
    for num_range, nyha in NYHA_MAP.items():
        if num in num_range:
            return nyha
    return 2
