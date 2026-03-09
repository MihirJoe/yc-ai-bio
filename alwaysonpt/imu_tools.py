"""
IMU signal processing tools for orientation data (pitch/yaw/roll).

Designed for low-rate IMU data (~10 Hz) from wearable sensors.
All functions operate on numpy arrays and return dicts for REPL use.
"""

import numpy as np
from scipy import signal as scipy_signal


def compute_orientation_stats(
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
) -> dict:
    """Compute summary statistics for each IMU axis.

    Args:
        pitch: Pitch angle array (degrees)
        yaw: Yaw angle array (degrees)
        roll: Roll angle array (degrees)

    Returns:
        Dict with mean, std, min, max, range for each axis.
    """
    result = {}
    for name, arr in [("pitch", pitch), ("yaw", yaw), ("roll", roll)]:
        a = np.asarray(arr, dtype=float)
        result[name] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "range": float(np.ptp(a)),
        }
    return result


def detect_gait_cycles(pitch: np.ndarray, fs: int) -> dict:
    """Detect gait cycles from pitch oscillation using peak detection.

    Args:
        pitch: Pitch angle array (degrees)
        fs: Sampling rate in Hz

    Returns:
        Dict with n_cycles, cycle_times, mean_cycle_duration, cadence_spm.
    """
    pitch = np.asarray(pitch, dtype=float)

    if len(pitch) < 4:
        return {"n_cycles": 0, "cycle_times": [], "mean_cycle_duration_s": 0, "cadence_spm": 0}

    pitch_centered = pitch - np.mean(pitch)

    min_distance = max(1, int(0.3 * fs))
    prominence = max(np.std(pitch_centered) * 0.3, 0.5)

    peaks, props = scipy_signal.find_peaks(
        pitch_centered, distance=min_distance, prominence=prominence,
    )

    cycle_times = (peaks / fs).tolist()
    n_cycles = len(peaks)

    if n_cycles >= 2:
        intervals = np.diff(peaks) / fs
        mean_dur = float(np.mean(intervals))
        cadence = 60.0 / mean_dur if mean_dur > 0 else 0
    else:
        mean_dur = 0
        cadence = 0

    return {
        "n_cycles": n_cycles,
        "cycle_times": cycle_times[:20],
        "mean_cycle_duration_s": round(mean_dur, 3),
        "cadence_spm": round(cadence, 1),
        "cycle_variability_cv": round(float(np.std(np.diff(peaks) / fs) / mean_dur * 100), 1) if mean_dur > 0 and n_cycles >= 3 else 0,
    }


def compute_cadence(pitch: np.ndarray, fs: int) -> dict:
    """Compute cadence (steps per minute) from pitch oscillation.

    Args:
        pitch: Pitch angle array (degrees)
        fs: Sampling rate in Hz

    Returns:
        Dict with cadence_spm and step_count.
    """
    gait = detect_gait_cycles(pitch, fs)
    return {
        "cadence_spm": gait["cadence_spm"],
        "step_count": gait["n_cycles"],
        "mean_step_duration_s": gait["mean_cycle_duration_s"],
    }


def compute_orientation_variability(
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
    window_s: float = 2.0,
    fs: int = 10,
) -> dict:
    """Compute windowed variability metrics for IMU orientation.

    Args:
        pitch: Pitch angle array (degrees)
        yaw: Yaw angle array (degrees)
        roll: Roll angle array (degrees)
        window_s: Window size in seconds
        fs: Sampling rate in Hz

    Returns:
        Dict with per-window std for each axis, overall variability score.
    """
    win_samples = max(2, int(window_s * fs))
    result = {}

    for name, arr in [("pitch", pitch), ("yaw", yaw), ("roll", roll)]:
        a = np.asarray(arr, dtype=float)
        n_windows = max(1, len(a) // win_samples)
        stds = []
        for i in range(n_windows):
            chunk = a[i * win_samples:(i + 1) * win_samples]
            if len(chunk) > 1:
                stds.append(float(np.std(chunk)))
        result[name] = {
            "mean_std": round(float(np.mean(stds)), 3) if stds else 0,
            "max_std": round(float(np.max(stds)), 3) if stds else 0,
            "n_windows": len(stds),
        }

    overall = sum(result[a]["mean_std"] for a in ["pitch", "yaw", "roll"]) / 3
    result["overall_variability"] = round(overall, 3)

    return result


def detect_activity_transitions(
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
    fs: int = 10,
) -> dict:
    """Detect sudden orientation changes that indicate activity transitions.

    Args:
        pitch: Pitch angle array (degrees)
        yaw: Yaw angle array (degrees)
        roll: Roll angle array (degrees)
        fs: Sampling rate in Hz

    Returns:
        Dict with transition events (time, magnitude, dominant axis).
    """
    transitions = []
    threshold_deg_per_s = 15.0

    for name, arr in [("pitch", pitch), ("yaw", yaw), ("roll", roll)]:
        a = np.asarray(arr, dtype=float)
        if len(a) < 2:
            continue
        rate = np.abs(np.diff(a)) * fs
        for i, r in enumerate(rate):
            if r > threshold_deg_per_s:
                transitions.append({
                    "time_s": round((i + 0.5) / fs, 3),
                    "rate_deg_per_s": round(float(r), 1),
                    "axis": name,
                })

    transitions.sort(key=lambda t: t["time_s"])

    merged = []
    for t in transitions:
        if merged and abs(t["time_s"] - merged[-1]["time_s"]) < 0.5:
            if t["rate_deg_per_s"] > merged[-1]["rate_deg_per_s"]:
                merged[-1] = t
        else:
            merged.append(t)

    return {
        "n_transitions": len(merged),
        "transitions": merged[:20],
    }


def classify_posture(
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
) -> dict:
    """Heuristic posture classification from mean orientation angles.

    Args:
        pitch: Pitch angle array (degrees)
        yaw: Yaw angle array (degrees)
        roll: Roll angle array (degrees)

    Returns:
        Dict with posture classification and confidence.
    """
    mean_pitch = float(np.mean(pitch))
    mean_roll = float(np.mean(roll))
    pitch_std = float(np.std(pitch))
    roll_std = float(np.std(roll))

    if abs(mean_pitch) > 60 or abs(mean_roll) > 60:
        posture = "lying"
        confidence = 0.7
    elif abs(mean_pitch) > 25 and pitch_std < 5:
        posture = "sitting"
        confidence = 0.65
    elif pitch_std < 3 and roll_std < 3:
        posture = "standing"
        confidence = 0.6
    else:
        posture = "active"
        confidence = 0.5

    return {
        "posture": posture,
        "confidence": round(confidence, 2),
        "mean_pitch": round(mean_pitch, 1),
        "mean_roll": round(mean_roll, 1),
        "pitch_std": round(pitch_std, 2),
        "roll_std": round(roll_std, 2),
    }
