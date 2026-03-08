"""
AlwaysOnPT — EMG Data Loader
Loads knee EMG dataset from Zhang et al. 2017 (S1File).
14 subjects × 3 exercises (standing, sitting, gait).
Preprocesses, segments into 4 motion classes, returns structured dataset.
"""

import os
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path


@dataclass
class EMGSegment:
    subject_id: int
    motion_class: str  # 'standing', 'sitting', 'stance', 'swing'
    emg: np.ndarray    # filtered EMG signal (mV), 1kHz
    gonio: np.ndarray  # knee joint angle (degrees), 1kHz
    fs: int = 1000
    duration_s: float = 0.0

    def __post_init__(self):
        self.duration_s = len(self.emg) / self.fs

    @classmethod
    def from_raw(cls, emg: np.ndarray, fs: int = 1000,
                 gonio: np.ndarray = None, subject_id: int = 0,
                 motion_class: str = 'unknown') -> 'EMGSegment':
        """Create an EMGSegment from raw arrays (no goniometer required)."""
        if gonio is None:
            gonio = np.zeros_like(emg)
        return cls(
            subject_id=subject_id,
            motion_class=motion_class,
            emg=emg,
            gonio=gonio,
            fs=fs,
        )


@dataclass
class EMGRecording:
    subject_id: int
    exercise: str       # 'standing', 'sitting', 'gait'
    emg_raw: np.ndarray
    gonio: np.ndarray
    fs: int = 1000
    n_samples: int = 0
    header_info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_samples = len(self.emg_raw)


@dataclass
class EMGDataset:
    recordings: list       # list of EMGRecording
    segments: list         # list of EMGSegment (after segmentation)
    subjects: list         # list of subject IDs
    class_counts: dict     # motion_class -> count
    metadata: dict         # summary stats


def parse_emg_file(filepath: str) -> EMGRecording:
    """Parse a single EMG data file. Handles both standard and subject-14 formats."""
    path = Path(filepath)
    filename = path.name

    subject_match = re.match(r'(\d+)(standing|sitting|gait)\.txt', filename)
    if not subject_match:
        raise ValueError(f"Unexpected filename format: {filename}")

    subject_id = int(subject_match.group(1))
    exercise = subject_match.group(2)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_info = {'filename': filename}
    data_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and re.match(r'^-?\d', stripped):
            data_start = i
            break
        header_info[f'header_{i}'] = stripped

    emg_vals = []
    gonio_vals = []

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                emg_vals.append(float(parts[0]))
                gonio_vals.append(float(parts[1]))
            except ValueError:
                continue

    emg_raw = np.array(emg_vals, dtype=np.float64)
    gonio = np.array(gonio_vals, dtype=np.float64)

    return EMGRecording(
        subject_id=subject_id,
        exercise=exercise,
        emg_raw=emg_raw,
        gonio=gonio,
        header_info=header_info,
    )


def _clean_goniometer(gonio: np.ndarray) -> np.ndarray:
    """Interpolate NaN values and ensure positive goniometer readings."""
    clean = gonio.copy()
    nan_mask = np.isnan(clean)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 2:
            clean[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_idx, clean[valid_idx]
            )
        else:
            clean[nan_mask] = 0.0
    if np.nanmean(clean) < 0:
        clean = np.abs(clean)
    return clean


def highpass_filter(signal: np.ndarray, cutoff: float = 20.0,
                    fs: int = 1000, order: int = 4) -> np.ndarray:
    """Apply Butterworth high-pass filter (20 Hz, as per paper)."""
    clean = signal.copy()
    nan_mask = np.isnan(clean)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 2:
            clean[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_idx, clean[valid_idx]
            )
        else:
            clean[nan_mask] = 0.0
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, clean)


def segment_repetitions(emg: np.ndarray, gonio: np.ndarray,
                        exercise: str, fs: int = 1000,
                        min_duration_s: float = 0.3) -> list:
    """
    Segment a recording into individual motion repetitions.

    For standing/sitting: detect motion onset/offset from goniometer derivative.
    For gait: detect individual gait cycles and split into stance/swing.
    """
    min_samples = int(min_duration_s * fs)

    if exercise in ('standing', 'sitting'):
        return _segment_discrete_motions(emg, gonio, exercise, fs, min_samples)
    elif exercise == 'gait':
        return _segment_gait_phases(emg, gonio, fs, min_samples)
    else:
        raise ValueError(f"Unknown exercise: {exercise}")


def _segment_discrete_motions(emg: np.ndarray, gonio: np.ndarray,
                               exercise: str, fs: int,
                               min_samples: int) -> list:
    """
    Segment standing/sitting into individual repetitions using goniometer.

    Standing: goniometer peaks at max flexion during each leg-raise rep.
    Find peaks, segment each rep between surrounding valleys.

    Sitting: goniometer valleys at max extension during each leg-extend rep.
    Find valleys, segment each rep between surrounding peaks.
    """
    gonio_smooth = np.convolve(gonio, np.ones(100) / 100, mode='same')
    gonio_range = np.max(gonio_smooth) - np.min(gonio_smooth)

    if gonio_range < 10:
        return []

    min_distance = int(0.8 * fs)
    prominence = gonio_range * 0.25

    if exercise == 'standing':
        peaks, _ = find_peaks(gonio_smooth, distance=min_distance,
                              prominence=prominence)
        boundaries, _ = find_peaks(-gonio_smooth, distance=int(0.3 * fs),
                                   prominence=prominence * 0.3)
    else:
        peaks, _ = find_peaks(-gonio_smooth, distance=min_distance,
                              prominence=prominence)
        boundaries, _ = find_peaks(gonio_smooth, distance=int(0.3 * fs),
                                   prominence=prominence * 0.3)

    if len(peaks) == 0:
        return []

    segments = []
    for peak in peaks:
        left_bounds = boundaries[boundaries < peak]
        right_bounds = boundaries[boundaries > peak]

        start = left_bounds[-1] if len(left_bounds) > 0 else max(0, peak - int(1.0 * fs))
        end = right_bounds[0] if len(right_bounds) > 0 else min(len(emg), peak + int(1.0 * fs))

        if (end - start) >= min_samples:
            segments.append({
                'motion_class': exercise,
                'emg': emg[start:end],
                'gonio': gonio[start:end],
                'start_idx': start,
                'end_idx': end,
            })

    return segments


def _segment_gait_phases(emg: np.ndarray, gonio: np.ndarray,
                          fs: int, min_samples: int) -> list:
    """
    Segment gait into stance and swing phases using goniometer.

    Gait cycle: the goniometer shows cyclical knee flexion/extension.
    - Swing phase: knee flexes more (higher angle, goniometer rising)
    - Stance phase: knee relatively extended (lower angle, goniometer falling/stable)

    We detect gait cycles by finding local minima in the goniometer signal
    (full extension at heel strike), then split each cycle at the local maximum
    (peak flexion during swing).
    """
    kernel = int(0.05 * fs)
    gonio_smooth = np.convolve(gonio, np.ones(kernel) / kernel, mode='same')

    gonio_range = np.max(gonio_smooth) - np.min(gonio_smooth)
    if gonio_range < 5:
        return []

    min_distance = int(0.8 * fs)
    prominence = gonio_range * 0.10

    minima, _ = find_peaks(-gonio_smooth, distance=min_distance,
                            prominence=prominence)

    if len(minima) < 2:
        return []

    segments = []

    for i in range(len(minima) - 1):
        cycle_start = minima[i]
        cycle_end = minima[i + 1]

        cycle_gonio = gonio_smooth[cycle_start:cycle_end]
        if len(cycle_gonio) == 0:
            continue

        peak_idx = np.argmax(cycle_gonio) + cycle_start

        stance_emg = emg[cycle_start:peak_idx]
        stance_gonio = gonio[cycle_start:peak_idx]

        if len(stance_emg) >= min_samples:
            segments.append({
                'motion_class': 'stance',
                'emg': stance_emg,
                'gonio': stance_gonio,
                'start_idx': cycle_start,
                'end_idx': peak_idx,
            })

        swing_emg = emg[peak_idx:cycle_end]
        swing_gonio = gonio[peak_idx:cycle_end]

        if len(swing_emg) >= min_samples:
            segments.append({
                'motion_class': 'swing',
                'emg': swing_emg,
                'gonio': swing_gonio,
                'start_idx': peak_idx,
                'end_idx': cycle_end,
            })

    return segments


def load_dataset(data_dir: str = None) -> EMGDataset:
    """
    Load and preprocess the full knee EMG dataset.

    Returns an EMGDataset with all recordings loaded, filtered,
    and segmented into the 4 motion classes.
    """
    if data_dir is None:
        base = Path(__file__).parent.parent
        data_dir = str(base / "data" / "knee_emg" / "S1File" / "Data")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    recordings = []
    all_segments = []
    subjects = set()

    for fpath in txt_files:
        try:
            rec = parse_emg_file(str(fpath))
        except (ValueError, Exception) as e:
            print(f"  Skipping {fpath.name}: {e}")
            continue

        emg_filtered = highpass_filter(rec.emg_raw)
        gonio_clean = _clean_goniometer(rec.gonio)
        subjects.add(rec.subject_id)
        recordings.append(rec)

        raw_segments = segment_repetitions(
            emg_filtered, gonio_clean, rec.exercise
        )

        for seg_data in raw_segments:
            segment = EMGSegment(
                subject_id=rec.subject_id,
                motion_class=seg_data['motion_class'],
                emg=seg_data['emg'],
                gonio=seg_data['gonio'],
            )
            all_segments.append(segment)

    subjects = sorted(subjects)
    class_counts = {}
    for seg in all_segments:
        class_counts[seg.motion_class] = class_counts.get(seg.motion_class, 0) + 1

    metadata = {
        'n_subjects': len(subjects),
        'n_recordings': len(recordings),
        'n_segments': len(all_segments),
        'class_counts': class_counts,
        'data_dir': data_dir,
    }

    print(f"Loaded {len(recordings)} recordings from {len(subjects)} subjects")
    print(f"Segmented into {len(all_segments)} segments: {class_counts}")

    return EMGDataset(
        recordings=recordings,
        segments=all_segments,
        subjects=subjects,
        class_counts=class_counts,
        metadata=metadata,
    )


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"\nDataset summary:")
    print(f"  Subjects: {dataset.subjects}")
    print(f"  Recordings: {dataset.metadata['n_recordings']}")
    print(f"  Segments: {dataset.metadata['n_segments']}")
    for cls, count in sorted(dataset.class_counts.items()):
        print(f"    {cls}: {count}")

    if dataset.segments:
        seg = dataset.segments[0]
        print(f"\nSample segment: subject={seg.subject_id}, "
              f"class={seg.motion_class}, "
              f"duration={seg.duration_s:.2f}s, "
              f"samples={len(seg.emg)}")
