"""
Domain-agnostic signal processing tools.
Operate on raw np.ndarray + fs, decoupled from any specific dataclass.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, butter, filtfilt
from pathlib import Path
from datetime import datetime
import pywt

OUTPUT_DIR = Path(__file__).parent / "output" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_TOOL_REGISTRY = {}


def _register(fn):
    SIGNAL_TOOL_REGISTRY[fn.__name__] = {
        'fn': fn,
        'description': fn.__doc__.strip().split('\n')[0] if fn.__doc__ else fn.__name__,
    }
    return fn


@_register
def compute_statistics(signal: np.ndarray, name: str = "") -> dict:
    """Compute descriptive statistics for any signal channel."""
    clean = signal[~np.isnan(signal)] if np.any(np.isnan(signal)) else signal
    if len(clean) == 0:
        return {'name': name, 'error': 'all NaN'}
    return {
        'name': name,
        'n_samples': len(clean),
        'mean': float(np.mean(clean)),
        'std': float(np.std(clean)),
        'min': float(np.min(clean)),
        'max': float(np.max(clean)),
        'median': float(np.median(clean)),
        'range': float(np.ptp(clean)),
        'rms': float(np.sqrt(np.mean(clean ** 2))),
        'skewness': float(_skewness(clean)),
        'kurtosis': float(_kurtosis(clean)),
        'nan_pct': float(np.sum(np.isnan(signal)) / len(signal) * 100) if len(signal) > 0 else 0,
    }


def _skewness(x):
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


@_register
def compute_psd(signal: np.ndarray, fs: int, name: str = "") -> dict:
    """Compute Welch PSD: MNF, MDF, total power, peak freq, band powers."""
    clean = signal[~np.isnan(signal)] if np.any(np.isnan(signal)) else signal
    if len(clean) < 16:
        return {'name': name, 'error': 'signal too short for PSD'}

    nperseg = min(256, len(clean))
    freqs, psd = welch(clean, fs=fs, nperseg=nperseg)

    total_power = float(np.sum(psd))
    if total_power == 0:
        return {'name': name, 'total_power': 0.0}

    mnf = float(np.sum(freqs * psd) / np.sum(psd))
    cum_power = np.cumsum(psd)
    mdf_idx = np.searchsorted(cum_power, cum_power[-1] / 2)
    mdf = float(freqs[min(mdf_idx, len(freqs) - 1)])
    peak_freq = float(freqs[np.argmax(psd)])

    nyq = fs / 2.0
    low_mask = freqs < nyq * 0.1
    mid_mask = (freqs >= nyq * 0.1) & (freqs < nyq * 0.5)
    high_mask = freqs >= nyq * 0.5

    return {
        'name': name,
        'MNF': mnf,
        'MDF': mdf,
        'total_power': total_power,
        'peak_freq': peak_freq,
        'low_band_ratio': float(np.sum(psd[low_mask]) / total_power) if np.any(low_mask) else 0.0,
        'mid_band_ratio': float(np.sum(psd[mid_mask]) / total_power) if np.any(mid_mask) else 0.0,
        'high_band_ratio': float(np.sum(psd[high_mask]) / total_power) if np.any(high_mask) else 0.0,
        'freqs': freqs.tolist(),
        'psd': psd.tolist(),
    }


@_register
def compute_wavelet(signal: np.ndarray, wavelet: str = 'db4', level: int = 5, name: str = "") -> dict:
    """Compute DWT decomposition: energy, entropy per subband, SVD features."""
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    actual_level = min(level, max_level)
    if actual_level < 1:
        return {'name': name, 'error': 'signal too short for wavelet decomposition'}

    coeffs = pywt.wavedec(signal, wavelet, level=actual_level)
    features = {'name': name, 'wavelet': wavelet, 'level': actual_level}

    energies = []
    for i, c in enumerate(coeffs):
        label = f'cA{actual_level}' if i == 0 else f'cD{actual_level - i + 1}'
        energy = float(np.sum(c ** 2))
        energies.append(energy)
        p = c ** 2
        p_sum = np.sum(p)
        entropy = 0.0
        if p_sum > 0:
            p_norm = p / p_sum
            p_norm = p_norm[p_norm > 0]
            entropy = float(-np.sum(p_norm * np.log2(p_norm)))
        features[f'{label}_energy'] = energy
        features[f'{label}_entropy'] = entropy

    total = sum(energies)
    if total > 0:
        for i, e in enumerate(energies):
            label = f'cA{actual_level}' if i == 0 else f'cD{actual_level - i + 1}'
            features[f'{label}_energy_ratio'] = float(e / total)

    return features


@_register
def compute_variability(values: np.ndarray, name: str = "") -> dict:
    """Compute variability metrics: CV, RMSSD, SDNN, pNN50, DFA alpha-1."""
    clean = values[~np.isnan(values)] if np.any(np.isnan(values)) else values
    if len(clean) < 4:
        return {'name': name, 'error': 'too few values for variability analysis'}

    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean))
    cv = float(std_val / mean_val) if mean_val != 0 else 0.0

    diffs = np.diff(clean)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = std_val

    result = {
        'name': name,
        'n_values': len(clean),
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'rmssd': rmssd,
        'sdnn': sdnn,
    }

    if mean_val > 0:
        threshold = 0.05 * mean_val
        pnn = float(np.sum(np.abs(diffs) > threshold) / len(diffs) * 100)
        result['pnn_5pct'] = pnn

    try:
        alpha = _dfa_alpha1(clean)
        result['dfa_alpha1'] = alpha
    except Exception:
        pass

    return result


def _dfa_alpha1(x, min_box=4, max_box=None):
    """Detrended fluctuation analysis (alpha-1 short-term exponent)."""
    n = len(x)
    if max_box is None:
        max_box = n // 4
    y = np.cumsum(x - np.mean(x))
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num=10
    ).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box]
    if len(box_sizes) < 2:
        return float('nan')

    fluctuations = []
    for bs in box_sizes:
        n_boxes = n // bs
        if n_boxes < 1:
            continue
        rms_vals = []
        for i in range(n_boxes):
            seg = y[i * bs:(i + 1) * bs]
            t = np.arange(bs)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(rms_vals))

    if len(fluctuations) < 2:
        return float('nan')
    log_n = np.log(box_sizes[:len(fluctuations)])
    log_f = np.log(fluctuations)
    alpha = float(np.polyfit(log_n, log_f, 1)[0])
    return alpha


@_register
def compute_cross_correlation(sig_a: np.ndarray, sig_b: np.ndarray,
                               name_a: str = "", name_b: str = "") -> dict:
    """Compute cross-correlation, max lag, and Pearson correlation between two channels."""
    min_len = min(len(sig_a), len(sig_b))
    a = sig_a[:min_len]
    b = sig_b[:min_len]

    a_clean = (a - np.mean(a))
    b_clean = (b - np.mean(b))
    a_norm = np.std(a)
    b_norm = np.std(b)

    if a_norm == 0 or b_norm == 0:
        return {'name_a': name_a, 'name_b': name_b, 'pearson_r': 0.0, 'error': 'zero variance'}

    pearson_r = float(np.mean(a_clean * b_clean) / (a_norm * b_norm))

    max_lag = min(min_len // 4, 500)
    xcorr = np.correlate(a_clean[:max_lag * 2], b_clean[:max_lag * 2], mode='full')
    xcorr = xcorr / (a_norm * b_norm * min(len(a_clean), max_lag * 2))
    peak_idx = np.argmax(np.abs(xcorr))
    lag_samples = peak_idx - len(xcorr) // 2

    return {
        'name_a': name_a,
        'name_b': name_b,
        'pearson_r': pearson_r,
        'max_xcorr': float(xcorr[peak_idx]),
        'lag_samples': int(lag_samples),
        'n_samples_compared': min_len,
    }


@_register
def compute_symmetry(left: np.ndarray, right: np.ndarray) -> dict:
    """Compute symmetry index between left and right channels."""
    min_len = min(len(left), len(right))
    l, r = left[:min_len], right[:min_len]

    l_mean = float(np.mean(np.abs(l)))
    r_mean = float(np.mean(np.abs(r)))
    denom = max(l_mean, r_mean)
    if denom == 0:
        return {'symmetry_index': 0.0, 'error': 'both channels zero'}

    si = float(2 * abs(l_mean - r_mean) / (l_mean + r_mean) * 100)

    l_rms = float(np.sqrt(np.mean(l ** 2)))
    r_rms = float(np.sqrt(np.mean(r ** 2)))
    ratio = float(l_rms / r_rms) if r_rms > 0 else float('inf')

    return {
        'symmetry_index_pct': si,
        'left_mean': l_mean,
        'right_mean': r_mean,
        'left_rms': l_rms,
        'right_rms': r_rms,
        'ratio_lr': ratio,
        'dominant_side': 'left' if l_mean > r_mean else 'right',
        'clinically_significant': si > 10.0,
    }


@_register
def detect_peaks(signal: np.ndarray, fs: int, name: str = "") -> dict:
    """Adaptive peak detection with rate, regularity, and amplitude statistics."""
    clean = signal[~np.isnan(signal)] if np.any(np.isnan(signal)) else signal
    if len(clean) < 10:
        return {'name': name, 'error': 'signal too short'}

    sig_std = np.std(clean)
    if sig_std == 0:
        return {'name': name, 'n_peaks': 0, 'error': 'zero variance signal'}

    height_threshold = np.mean(clean) + 0.5 * sig_std
    min_distance = max(1, int(0.2 * fs))

    peaks, properties = find_peaks(
        clean, height=height_threshold, distance=min_distance
    )

    if len(peaks) == 0:
        peaks, properties = find_peaks(clean, distance=min_distance, prominence=sig_std * 0.3)

    if len(peaks) < 2:
        return {
            'name': name,
            'n_peaks': len(peaks),
            'peaks_per_second': len(peaks) / (len(clean) / fs) if fs > 0 else 0,
        }

    intervals = np.diff(peaks) / fs
    amplitudes = clean[peaks]

    return {
        'name': name,
        'n_peaks': len(peaks),
        'peaks_per_second': float(len(peaks) / (len(clean) / fs)),
        'mean_interval_s': float(np.mean(intervals)),
        'std_interval_s': float(np.std(intervals)),
        'cv_interval': float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0,
        'mean_amplitude': float(np.mean(amplitudes)),
        'std_amplitude': float(np.std(amplitudes)),
        'min_interval_s': float(np.min(intervals)),
        'max_interval_s': float(np.max(intervals)),
    }


@_register
def segment_signal(signal: np.ndarray, fs: int, window_s: float = 1.0,
                   overlap: float = 0.5, name: str = "") -> dict:
    """Segment a signal into overlapping windows and compute per-window statistics."""
    win_samples = int(window_s * fs)
    step = int(win_samples * (1 - overlap))
    if win_samples < 4 or step < 1:
        return {'name': name, 'error': 'window too small'}

    windows = []
    i = 0
    while i + win_samples <= len(signal):
        w = signal[i:i + win_samples]
        windows.append({
            'start_s': float(i / fs),
            'rms': float(np.sqrt(np.mean(w ** 2))),
            'mean': float(np.mean(w)),
            'std': float(np.std(w)),
        })
        i += step

    return {
        'name': name,
        'n_windows': len(windows),
        'window_s': window_s,
        'overlap': overlap,
        'windows': windows,
    }


@_register
def generate_signal_plot(signals: dict, fs: int, title: str = "",
                         annotations: list = None) -> str:
    """Generate a multi-channel signal plot. Returns saved file path."""
    n_channels = len(signals)
    if n_channels == 0:
        return ''

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3 * n_channels),
                             sharex=True, squeeze=False)

    for idx, (ch_name, data) in enumerate(signals.items()):
        ax = axes[idx, 0]
        if fs > 0:
            t = np.arange(len(data)) / fs
            ax.plot(t, data, linewidth=0.5, alpha=0.8)
            ax.set_xlabel('Time (s)')
        else:
            ax.plot(data, linewidth=0.5, alpha=0.8)
            ax.set_xlabel('Sample')

        ax.set_ylabel(ch_name)
        ax.grid(True, alpha=0.3)

    if title:
        axes[0, 0].set_title(title)

    if annotations:
        for ann in annotations:
            for ax_row in axes:
                ax_row[0].axvline(x=ann.get('x', 0), color='r',
                                  linestyle='--', alpha=0.5)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    safe_title = (title or 'signal').replace(' ', '_').replace('/', '_')
    fname = f"{safe_title}_{timestamp}.png"
    fpath = OUTPUT_DIR / fname
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(fpath)
