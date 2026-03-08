"""
Loader for PhysioNet Gait in Neurodegenerative Disease database (gaitndd).
15 PD + 20 HD + 13 ALS + 16 controls. Stride/swing/stance timing data.
https://physionet.org/content/gaitndd/1.0.0/
"""

import numpy as np
from pathlib import Path
from alwaysonpt.datasets.base import BioSignalRecord

DISEASE_PREFIXES = {
    'als': 'als',
    'hunt': 'huntingtons',
    'park': 'parkinsons',
    'control': 'control',
}

TS_COLUMNS = [
    'elapsed_time',
    'left_stride', 'right_stride',
    'left_swing', 'right_swing',
    'left_swing_pct', 'right_swing_pct',
    'left_stance', 'right_stance',
    'left_stance_pct', 'right_stance_pct',
    'double_support', 'double_support_pct',
]


def load_gaitndd(data_dir: str, max_records: int = None) -> list:
    """
    Load gaitndd dataset from .ts (time series) files.

    Each .ts file has 13 columns of gait timing data.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"gaitndd data not found: {data_dir}")

    subject_info = _load_subject_descriptions(data_path)

    ts_files = sorted(data_path.glob("*.ts"))
    records = []

    for fpath in ts_files:
        subject_name = fpath.stem

        disease = _classify_subject(subject_name)

        try:
            data = np.loadtxt(str(fpath))
        except Exception:
            continue

        if data.ndim != 2 or data.shape[1] < 13:
            continue

        signals = {}
        for i, col_name in enumerate(TS_COLUMNS):
            if col_name != 'elapsed_time':
                signals[col_name] = data[:, i]

        duration_s = float(data[-1, 0]) if len(data) > 1 else 0.0

        meta = {'subject_name': subject_name}
        gt = {'disease': disease}

        if subject_name in subject_info:
            meta.update(subject_info[subject_name])
            for k in ('age', 'gender', 'height', 'weight', 'speed'):
                if k in subject_info[subject_name]:
                    gt[k] = subject_info[subject_name][k]

        records.append(BioSignalRecord(
            record_id=f"gaitndd_{subject_name}",
            domain='gait_timing',
            signals=signals,
            fs=0,  # event-based (one row per stride)
            duration_s=duration_s,
            metadata=meta,
            ground_truth=gt,
        ))

        if max_records and len(records) >= max_records:
            break

    return records


def _classify_subject(name: str) -> str:
    """Determine disease from subject file name."""
    name_lower = name.lower()
    for prefix, disease in DISEASE_PREFIXES.items():
        if name_lower.startswith(prefix):
            return disease
    return 'control'


def _load_subject_descriptions(data_path: Path) -> dict:
    """Parse subject-description.txt if available."""
    info = {}
    for candidate in ['subject-description.txt', 'Subject-description.txt']:
        desc_path = data_path / candidate
        if desc_path.exists():
            try:
                lines = desc_path.read_text().splitlines()
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        sid = parts[0].strip()
                        entry = {}
                        for i, key in enumerate(['age', 'gender', 'height', 'weight', 'speed'], 1):
                            if i < len(parts):
                                try:
                                    entry[key] = float(parts[i]) if key != 'gender' else parts[i].strip()
                                except ValueError:
                                    entry[key] = parts[i].strip()
                        info[sid] = entry
            except Exception:
                pass
            break
    return info
