"""
Synthetic biosignal generator for demo sessions.

Produces realistic-ish surface EMG signals with controllable
movement scenarios: walking, resting, exercise, or mixed.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def generate_session(duration_s: float = 15.0, fs: int = 1000,
                     scenario: str = 'walking', seed: int = None) -> dict:
    """
    Generate a synthetic EMG recording session.

    Returns dict with:
        emg:        np.ndarray of the signal
        labels:     list of (start_s, end_s, phase_name) tuples
        fs:         sampling rate
        duration_s: actual duration
        scenario:   scenario name
    """
    if seed is not None:
        np.random.seed(seed)

    n = int(duration_s * fs)

    generators = {
        'walking': _walking,
        'resting': _resting,
        'exercise': _exercise,
        'mixed': _mixed,
    }
    gen_fn = generators.get(scenario, _walking)
    emg, labels = gen_fn(n, fs)

    return {
        'emg': emg,
        'labels': labels,
        'fs': fs,
        'duration_s': duration_s,
        'scenario': scenario,
    }


def _bandpass_noise(n: int, fs: int, low: float = 20, high: float = 450) -> np.ndarray:
    """Bandpass-filtered Gaussian noise mimicking surface EMG frequency content."""
    noise = np.random.randn(n)
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, min(high / nyq, 0.99)], btype='band')
    return filtfilt(b, a, noise)


def _walking(n: int, fs: int) -> tuple:
    """Alternating stance/swing gait cycles with heel-strike bursts."""
    cycle_s = 1.1
    stance_ratio = 0.6
    emg = _bandpass_noise(n, fs) * 0.02
    labels = []
    pos = 0

    while pos < n:
        cycle_n = int(cycle_s * fs * (1 + 0.05 * np.random.randn()))
        stance_n = int(cycle_n * stance_ratio)
        swing_n = cycle_n - stance_n

        # Stance: moderate activation + heel-strike burst
        end = min(pos + stance_n, n)
        seg_len = end - pos
        if seg_len > 0:
            envelope = np.ones(seg_len) * 0.30
            burst = int(0.08 * fs)
            if seg_len > burst:
                envelope[:burst] *= 2.5
                ramp = np.linspace(2.5, 1.0, min(burst, seg_len - burst))
                envelope[burst:burst + len(ramp)] *= ramp[:len(envelope) - burst]
            emg[pos:end] += _bandpass_noise(seg_len, fs) * envelope
            labels.append((pos / fs, end / fs, 'stance'))
        pos = end

        # Swing: low activation
        end = min(pos + swing_n, n)
        seg_len = end - pos
        if seg_len > 0:
            emg[pos:end] += _bandpass_noise(seg_len, fs) * 0.07
            labels.append((pos / fs, end / fs, 'swing'))
        pos = end

    return emg, labels


def _resting(n: int, fs: int) -> tuple:
    """Low-amplitude tonic activity with occasional micro-twitches."""
    emg = _bandpass_noise(n, fs) * 0.015

    for _ in range(np.random.randint(2, 6)):
        start = np.random.randint(0, max(1, n - int(0.08 * fs)))
        length = int((0.03 + 0.05 * np.random.rand()) * fs)
        end = min(start + length, n)
        emg[start:end] += _bandpass_noise(end - start, fs) * 0.08

    return emg, [(0, n / fs, 'resting')]


def _exercise(n: int, fs: int) -> tuple:
    """Sustained contraction with progressive fatigue (rising amplitude, falling MDF)."""
    t_norm = np.linspace(0, 1, n)

    # Fatigue: amplitude rises, more motor unit recruitment
    envelope = 0.40 * (1.0 + 0.6 * t_norm)
    emg = _bandpass_noise(n, fs) * envelope

    # Add occasional high-amplitude bursts during effort
    for _ in range(np.random.randint(3, 8)):
        start = np.random.randint(0, max(1, n - int(0.15 * fs)))
        length = int((0.05 + 0.1 * np.random.rand()) * fs)
        end = min(start + length, n)
        emg[start:end] += _bandpass_noise(end - start, fs) * 0.25

    return emg, [(0, n / fs, 'sustained_contraction')]


def _mixed(n: int, fs: int) -> tuple:
    """Three phases: rest → walking → exercise (equal thirds)."""
    third = n // 3
    emg = np.zeros(n)
    labels = []

    rest_emg, _ = _resting(third, fs)
    emg[:third] = rest_emg
    labels.append((0, third / fs, 'resting'))

    walk_emg, walk_labels = _walking(third, fs)
    emg[third:2 * third] = walk_emg
    for s, e, lbl in walk_labels:
        labels.append((s + third / fs, e + third / fs, lbl))

    remaining = n - 2 * third
    ex_emg, _ = _exercise(remaining, fs)
    emg[2 * third:] = ex_emg
    labels.append((2 * third / fs, n / fs, 'sustained_contraction'))

    return emg, labels
