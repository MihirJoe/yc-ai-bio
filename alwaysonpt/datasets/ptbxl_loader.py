"""
Loader for PhysioNet PTB-XL 12-lead ECG database.
21,801 clinical 12-lead ECGs, 10s each, 500Hz, annotated with SCP codes.
https://physionet.org/content/ptb-xl/1.0.3/
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from alwaysonpt.datasets.base import BioSignalRecord

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

SUPERCLASS_MAP = {
    'NORM': 'NORM',
    'MI': 'MI',
    'STTC': 'STTC',
    'CD': 'CD',
    'HYP': 'HYP',
}


def load_ptbxl(data_dir: str, max_records: int = 200,
               sampling_rate: int = 500, stratify: bool = True) -> list:
    """
    Load PTB-XL dataset. Returns list of BioSignalRecord.

    Uses wfdb to read signal files. Falls back to numpy if wfdb unavailable.
    Stratifies by diagnostic superclass if stratify=True.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"PTB-XL data not found: {data_dir}")

    db_csv = data_path / 'ptbxl_database.csv'
    if not db_csv.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found in {data_dir}")

    df = pd.read_csv(db_csv, index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    scp_df = _load_scp_statements(data_path)
    if scp_df is not None:
        df = _add_superclass(df, scp_df)

    if stratify and 'diagnostic_superclass' in df.columns and max_records:
        df = _stratified_sample(df, max_records)
    elif max_records:
        df = df.head(max_records)

    records_dir = f'records{sampling_rate}'
    fname_col = 'filename_hr' if sampling_rate == 500 else 'filename_lr'

    records = []
    for ecg_id, row in df.iterrows():
        fname = row[fname_col]
        record_path = data_path / fname

        signal_data = _read_ecg(record_path, sampling_rate)
        if signal_data is None:
            continue

        signals = {}
        for i, lead in enumerate(LEAD_NAMES):
            if i < signal_data.shape[1]:
                signals[lead] = signal_data[:, i]

        duration_s = float(signal_data.shape[0] / sampling_rate)

        meta = {
            'ecg_id': int(ecg_id),
            'age': row.get('age', None),
            'sex': row.get('sex', None),
            'recording_date': str(row.get('recording_date', '')),
        }

        gt = {
            'scp_codes': row['scp_codes'],
        }
        if 'diagnostic_superclass' in row and pd.notna(row.get('diagnostic_superclass')):
            gt['diagnostic_superclass'] = row['diagnostic_superclass']

        records.append(BioSignalRecord(
            record_id=f"ptbxl_{ecg_id}",
            domain='ecg_12lead',
            signals=signals,
            fs=sampling_rate,
            duration_s=duration_s,
            metadata=meta,
            ground_truth=gt,
        ))

    return records


def _read_ecg(record_path: Path, fs: int) -> np.ndarray:
    """Read ECG signal using wfdb or numpy fallback."""
    try:
        import wfdb
        record = wfdb.rdrecord(str(record_path))
        return record.p_signal
    except ImportError:
        dat_path = record_path.with_suffix('.dat')
        hea_path = record_path.with_suffix('.hea')
        if dat_path.exists():
            try:
                data = np.fromfile(str(dat_path), dtype=np.int16)
                n_leads = 12
                data = data.reshape(-1, n_leads).astype(np.float64)
                return data / 1000.0
            except Exception:
                pass
    except Exception:
        pass
    return None


def _load_scp_statements(data_path: Path) -> pd.DataFrame:
    """Load SCP statement descriptions."""
    scp_path = data_path / 'scp_statements.csv'
    if scp_path.exists():
        return pd.read_csv(scp_path, index_col=0)
    return None


def _add_superclass(df: pd.DataFrame, scp_df: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic_superclass column based on SCP codes."""
    def get_superclass(scp_codes):
        if not scp_codes:
            return None
        for code in scp_codes:
            if code in scp_df.index:
                row = scp_df.loc[code]
                if hasattr(row, 'diagnostic_class') and pd.notna(row.get('diagnostic_class', None)):
                    return str(row['diagnostic_class'])
                if hasattr(row, 'diagnostic') and row.get('diagnostic', 0) == 1:
                    for sc in SUPERCLASS_MAP:
                        if sc in str(code).upper():
                            return sc
        max_code = max(scp_codes, key=lambda c: scp_codes[c])
        if max_code in scp_df.index:
            row = scp_df.loc[max_code]
            if hasattr(row, 'diagnostic_class') and pd.notna(row.get('diagnostic_class', None)):
                return str(row['diagnostic_class'])
        return None

    df['diagnostic_superclass'] = df['scp_codes'].apply(get_superclass)
    return df


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample evenly across diagnostic superclasses."""
    classes = df['diagnostic_superclass'].dropna().unique()
    if len(classes) == 0:
        return df.head(n)

    per_class = max(1, n // len(classes))
    sampled = []
    for cls in classes:
        subset = df[df['diagnostic_superclass'] == cls]
        sampled.append(subset.sample(min(per_class, len(subset)), random_state=42))

    result = pd.concat(sampled)
    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra = remaining.sample(min(n - len(result), len(remaining)), random_state=42)
        result = pd.concat([result, extra])

    return result
