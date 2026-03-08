"""
Loader for PhysioNet Gait in Parkinson's Disease database (gaitpdb).
93 PD patients + 73 healthy controls, VGRF at 100Hz from 16 foot sensors.
https://physionet.org/content/gaitpdb/1.0.0/
"""

import re
import numpy as np
from pathlib import Path
from alwaysonpt.datasets.base import BioSignalRecord

FS = 100  # 100 Hz sampling rate

# File naming: [Study][Group][Number]_[Walk].txt
# Study: Ga (dual task), Ju (rhythmic auditory stim), Si (treadmill)
# Group: Co (control), Pt (patient)
FILE_PATTERN = re.compile(r'^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$')

DEMOGRAPHICS_FILE = 'demographics.txt'


def load_gaitpdb(data_dir: str, max_records: int = None) -> list:
    """
    Load gaitpdb dataset. Returns list of BioSignalRecord.

    Each file has 19 columns:
      col 0: time (s)
      cols 1-8: VGRF from 8 left foot sensors (N)
      cols 9-16: VGRF from 8 right foot sensors (N)
      col 17: total force, left foot (N)
      col 18: total force, right foot (N)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"gaitpdb data not found: {data_dir}")

    demographics = _load_demographics(data_path)

    txt_files = sorted(data_path.glob("*.txt"))
    records = []

    for fpath in txt_files:
        match = FILE_PATTERN.match(fpath.name)
        if not match:
            continue

        study, group, subject_num, walk_num = match.groups()
        subject_id = f"{study}{group}{subject_num}"

        try:
            data = np.loadtxt(str(fpath))
        except Exception:
            continue

        if data.ndim != 2 or data.shape[1] < 19:
            continue

        signals = {
            'left_total_force': data[:, 17],
            'right_total_force': data[:, 18],
        }
        for i in range(8):
            signals[f'left_sensor_{i}'] = data[:, 1 + i]
            signals[f'right_sensor_{i}'] = data[:, 9 + i]

        duration_s = float(data[-1, 0] - data[0, 0]) if len(data) > 1 else 0.0

        meta = {
            'study': study,
            'subject_id': subject_id,
            'walk': int(walk_num),
        }
        gt = {
            'group': 'control' if group == 'Co' else 'patient',
        }

        demo_key = subject_id
        if demo_key in demographics:
            meta.update(demographics[demo_key])
            if 'hoehn_yahr' in demographics[demo_key]:
                gt['hoehn_yahr'] = demographics[demo_key]['hoehn_yahr']
            if 'updrs' in demographics[demo_key]:
                gt['updrs'] = demographics[demo_key]['updrs']
            if 'speed' in demographics[demo_key]:
                gt['speed'] = demographics[demo_key]['speed']

        records.append(BioSignalRecord(
            record_id=f"gaitpdb_{fpath.stem}",
            domain='gait_vgrf',
            signals=signals,
            fs=FS,
            duration_s=duration_s,
            metadata=meta,
            ground_truth=gt,
        ))

        if max_records and len(records) >= max_records:
            break

    return records


def _load_demographics(data_path: Path) -> dict:
    """Parse demographics file if available."""
    demo = {}
    for candidate in ['demographics.txt', 'Demographics.txt']:
        demo_path = data_path / candidate
        if demo_path.exists():
            try:
                lines = demo_path.read_text().splitlines()
                for line in lines[1:]:  # skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        sid = parts[0].strip()
                        info = {}
                        try:
                            info['age'] = float(parts[1])
                        except (ValueError, IndexError):
                            pass
                        try:
                            info['gender'] = parts[2].strip()
                        except IndexError:
                            pass
                        if len(parts) > 3:
                            try:
                                info['height'] = float(parts[3])
                            except (ValueError, IndexError):
                                pass
                        if len(parts) > 4:
                            try:
                                info['weight'] = float(parts[4])
                            except (ValueError, IndexError):
                                pass
                        if len(parts) > 5:
                            try:
                                hy = float(parts[5])
                                info['hoehn_yahr'] = hy
                            except (ValueError, IndexError):
                                pass
                        if len(parts) > 6:
                            try:
                                info['updrs'] = float(parts[6])
                            except (ValueError, IndexError):
                                pass
                        if len(parts) > 7:
                            try:
                                info['speed'] = float(parts[7])
                            except (ValueError, IndexError):
                                pass
                        demo[sid] = info
            except Exception:
                pass
            break
    return demo
