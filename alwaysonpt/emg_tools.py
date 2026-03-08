"""
AlwaysOnPT — EMG Tool Functions
REPL-accessible tools for EMG signal analysis, feature extraction,
and clinical reasoning support.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
from datetime import datetime
import pywt
import os


OUTPUT_DIR = Path(__file__).parent / "output" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_segment_overview(segment) -> dict:
    """Summary statistics and data quality metrics for an EMG segment."""
    emg = segment.emg
    gonio = segment.gonio

    rms = np.sqrt(np.mean(emg ** 2))
    snr_estimate = rms / (np.std(emg - np.convolve(emg, np.ones(10)/10, mode='same')) + 1e-10)

    return {
        'subject_id': segment.subject_id,
        'motion_class': segment.motion_class,
        'duration_s': segment.duration_s,
        'n_samples': len(emg),
        'fs': segment.fs,
        'emg_mean_mv': float(np.mean(emg)),
        'emg_std_mv': float(np.std(emg)),
        'emg_rms_mv': float(rms),
        'emg_min_mv': float(np.min(emg)),
        'emg_max_mv': float(np.max(emg)),
        'emg_peak_to_peak': float(np.max(emg) - np.min(emg)),
        'gonio_mean_deg': float(np.mean(gonio)),
        'gonio_range_deg': float(np.max(gonio) - np.min(gonio)),
        'snr_estimate': float(snr_estimate),
        'data_quality': 'good' if snr_estimate > 2.0 else 'marginal' if snr_estimate > 1.0 else 'poor',
    }


def extract_time_features(segment) -> dict:
    """
    Time-domain features from EMG signal.
    Standard features used in EMG classification literature:
    MAV, RMS, iEMG, ZC, SSC, WL, VAR
    """
    emg = segment.emg
    n = len(emg)
    abs_emg = np.abs(emg)

    mav = float(np.mean(abs_emg))
    rms = float(np.sqrt(np.mean(emg ** 2)))
    iemg = float(np.sum(abs_emg))
    var = float(np.var(emg))
    wl = float(np.sum(np.abs(np.diff(emg))))

    threshold = 0.01 * mav if mav > 0 else 1e-6
    zc = 0
    for i in range(n - 1):
        if (emg[i] > 0 and emg[i+1] < 0) or (emg[i] < 0 and emg[i+1] > 0):
            if abs(emg[i] - emg[i+1]) > threshold:
                zc += 1

    ssc = 0
    for i in range(1, n - 1):
        diff1 = emg[i] - emg[i-1]
        diff2 = emg[i] - emg[i+1]
        if diff1 * diff2 > 0:
            if abs(diff1) > threshold or abs(diff2) > threshold:
                ssc += 1

    return {
        'MAV': mav,
        'RMS': rms,
        'iEMG': iemg,
        'VAR': var,
        'WL': wl,
        'ZC': zc,
        'SSC': ssc,
        'MAV_normalized': mav / (segment.duration_s + 1e-10),
    }


def extract_freq_features(segment) -> dict:
    """
    Frequency-domain features from EMG signal.
    MNF, MDF, total power, peak frequency, spectral bandwidth.
    """
    emg = segment.emg
    fs = segment.fs

    nperseg = min(256, len(emg))
    freqs, psd = welch(emg, fs=fs, nperseg=nperseg)

    total_power = float(np.sum(psd))
    if total_power == 0:
        return {
            'MNF': 0.0, 'MDF': 0.0, 'total_power': 0.0,
            'peak_freq': 0.0, 'bandwidth': 0.0,
            'low_band_power': 0.0, 'mid_band_power': 0.0, 'high_band_power': 0.0,
        }

    mnf = float(np.sum(freqs * psd) / np.sum(psd))

    cum_power = np.cumsum(psd)
    mdf_idx = np.searchsorted(cum_power, cum_power[-1] / 2)
    mdf = float(freqs[min(mdf_idx, len(freqs) - 1)])

    peak_freq = float(freqs[np.argmax(psd)])
    mean_sq = np.sum(freqs**2 * psd) / np.sum(psd)
    bandwidth = float(np.sqrt(mean_sq - mnf**2))

    low_mask = freqs < 50
    mid_mask = (freqs >= 50) & (freqs < 150)
    high_mask = freqs >= 150

    return {
        'MNF': mnf,
        'MDF': mdf,
        'total_power': total_power,
        'peak_freq': peak_freq,
        'bandwidth': bandwidth,
        'low_band_power': float(np.sum(psd[low_mask]) / total_power),
        'mid_band_power': float(np.sum(psd[mid_mask]) / total_power),
        'high_band_power': float(np.sum(psd[high_mask]) / total_power),
    }


def extract_wavelet_features(segment, wavelet: str = 'db4',
                              level: int = 5) -> dict:
    """
    Wavelet-domain features via discrete wavelet transform.
    Uses db4 wavelet with 5-level decomposition (matching paper's WT-SVD approach).
    Returns energy, entropy, and statistics for each decomposition level,
    plus SVD-based features of the coefficient matrix.
    """
    emg = segment.emg

    max_level = pywt.dwt_max_level(len(emg), pywt.Wavelet(wavelet).dec_len)
    actual_level = min(level, max_level)

    coeffs = pywt.wavedec(emg, wavelet, level=actual_level)

    features = {
        'wavelet': wavelet,
        'decomposition_level': actual_level,
        'n_subbands': len(coeffs),
    }

    energies = []
    for i, c in enumerate(coeffs):
        name = f'cA{actual_level}' if i == 0 else f'cD{actual_level - i + 1}'
        energy = float(np.sum(c ** 2))
        energies.append(energy)

        entropy = 0.0
        p = c ** 2
        p_sum = np.sum(p)
        if p_sum > 0:
            p_norm = p / p_sum
            p_norm = p_norm[p_norm > 0]
            entropy = float(-np.sum(p_norm * np.log2(p_norm)))

        features[f'{name}_energy'] = energy
        features[f'{name}_entropy'] = entropy
        features[f'{name}_mean'] = float(np.mean(np.abs(c)))
        features[f'{name}_std'] = float(np.std(c))

    total_energy = sum(energies)
    if total_energy > 0:
        for i, (e, c) in enumerate(zip(energies, coeffs)):
            name = f'cA{actual_level}' if i == 0 else f'cD{actual_level - i + 1}'
            features[f'{name}_energy_ratio'] = float(e / total_energy)

    max_len = max(len(c) for c in coeffs)
    coeff_matrix = np.zeros((len(coeffs), max_len))
    for i, c in enumerate(coeffs):
        coeff_matrix[i, :len(c)] = c

    try:
        U, S, Vt = np.linalg.svd(coeff_matrix, full_matrices=False)
        features['svd_singular_values'] = S.tolist()
        features['svd_top3'] = S[:3].tolist() if len(S) >= 3 else S.tolist()
        features['svd_energy_ratio'] = float(S[0] ** 2 / (np.sum(S ** 2) + 1e-10))
        features['svd_condition_number'] = float(S[0] / (S[-1] + 1e-10))
    except np.linalg.LinAlgError:
        features['svd_error'] = 'SVD did not converge'

    return features


def detect_fatigue_pattern(segment, window_size_s: float = 0.5) -> dict:
    """
    Detect fatigue indicators within a segment by analyzing
    MDF trend and amplitude changes across sliding windows.
    Fatigue manifests as decreasing MDF + increasing amplitude.
    """
    emg = segment.emg
    fs = segment.fs
    win = int(window_size_s * fs)

    if len(emg) < 2 * win:
        return {
            'fatigue_detected': False,
            'reason': 'segment too short for fatigue analysis',
            'n_windows': 0,
        }

    n_windows = len(emg) // win
    mdfs = []
    rmss = []

    for i in range(n_windows):
        chunk = emg[i * win: (i + 1) * win]
        nperseg = min(128, len(chunk))
        freqs, psd = welch(chunk, fs=fs, nperseg=nperseg)
        total = np.sum(psd)
        if total > 0:
            cum = np.cumsum(psd)
            idx = np.searchsorted(cum, cum[-1] / 2)
            mdfs.append(float(freqs[min(idx, len(freqs) - 1)]))
        else:
            mdfs.append(0.0)
        rmss.append(float(np.sqrt(np.mean(chunk ** 2))))

    mdfs = np.array(mdfs)
    rmss = np.array(rmss)

    mdf_slope = 0.0
    rms_slope = 0.0
    if len(mdfs) >= 2:
        t = np.arange(len(mdfs))
        mdf_slope = float(np.polyfit(t, mdfs, 1)[0])
        rms_slope = float(np.polyfit(t, rmss, 1)[0])

    fatigue_detected = (mdf_slope < -2.0 and rms_slope > 0)

    return {
        'fatigue_detected': fatigue_detected,
        'mdf_slope_hz_per_window': mdf_slope,
        'rms_slope_mv_per_window': rms_slope,
        'mdf_start': float(mdfs[0]) if len(mdfs) > 0 else 0.0,
        'mdf_end': float(mdfs[-1]) if len(mdfs) > 0 else 0.0,
        'rms_start': float(rmss[0]) if len(rmss) > 0 else 0.0,
        'rms_end': float(rmss[-1]) if len(rmss) > 0 else 0.0,
        'n_windows': n_windows,
        'window_mdfs': mdfs.tolist(),
        'window_rmss': rmss.tolist(),
    }


def compare_to_baseline(segment, baseline_stats: dict = None) -> dict:
    """
    Compare a segment's features against population or session baselines.
    If no baseline provided, uses default normative values.
    """
    time_feats = extract_time_features(segment)
    freq_feats = extract_freq_features(segment)

    if baseline_stats is None:
        baseline_stats = {
            'MAV': 0.05,
            'RMS': 0.07,
            'MNF': 120.0,
            'MDF': 100.0,
            'total_power': 0.001,
        }

    deviations = {}
    for key in ['MAV', 'RMS']:
        if key in baseline_stats and baseline_stats[key] > 0:
            actual = time_feats[key]
            baseline = baseline_stats[key]
            deviations[key] = {
                'actual': actual,
                'baseline': baseline,
                'ratio': float(actual / baseline),
                'z_score': float((actual - baseline) / (baseline * 0.3 + 1e-10)),
            }

    for key in ['MNF', 'MDF', 'total_power']:
        if key in baseline_stats and baseline_stats[key] > 0:
            actual = freq_feats[key]
            baseline = baseline_stats[key]
            deviations[key] = {
                'actual': actual,
                'baseline': baseline,
                'ratio': float(actual / baseline),
                'z_score': float((actual - baseline) / (baseline * 0.2 + 1e-10)),
            }

    abnormal = {k: v for k, v in deviations.items() if abs(v['z_score']) > 2.0}

    return {
        'deviations': deviations,
        'abnormal_features': list(abnormal.keys()),
        'overall_deviation_score': float(
            np.mean([abs(v['z_score']) for v in deviations.values()])
        ) if deviations else 0.0,
    }


def generate_plot(segment, plot_type: str = 'overview',
                  title: str = None, save: bool = True) -> str:
    """
    Generate a visualization of an EMG segment.

    plot_type options:
      - 'overview': EMG + goniometer dual-axis plot
      - 'spectrum': PSD plot
      - 'wavelet': Wavelet decomposition visualization
      - 'features': Feature comparison bar chart

    Returns the file path if saved, empty string otherwise.
    """
    if title is None:
        title = f"S{segment.subject_id} {segment.motion_class}"

    fig, axes = plt.subplots(figsize=(12, 6))

    if plot_type == 'overview':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = np.arange(len(segment.emg)) / segment.fs

        ax1.plot(t, segment.emg, 'b-', linewidth=0.5, alpha=0.8)
        ax1.set_ylabel('EMG (mV)')
        ax1.set_title(f'{title} — EMG + Goniometer')
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, segment.gonio, 'r-', linewidth=1.0)
        ax2.set_ylabel('Knee Angle (°)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        plt.close(axes.figure)

    elif plot_type == 'spectrum':
        nperseg = min(256, len(segment.emg))
        freqs, psd = welch(segment.emg, fs=segment.fs, nperseg=nperseg)
        axes.semilogy(freqs, psd, 'b-')
        axes.set_xlabel('Frequency (Hz)')
        axes.set_ylabel('PSD (mV²/Hz)')
        axes.set_title(f'{title} — Power Spectral Density')
        axes.grid(True, alpha=0.3)
        fig = axes.figure

    elif plot_type == 'wavelet':
        coeffs = pywt.wavedec(segment.emg, 'db4',
                              level=min(5, pywt.dwt_max_level(
                                  len(segment.emg), pywt.Wavelet('db4').dec_len)))
        n_levels = len(coeffs)
        plt.close(fig)
        fig, axes_arr = plt.subplots(n_levels, 1, figsize=(12, 2 * n_levels))
        if n_levels == 1:
            axes_arr = [axes_arr]
        for i, (ax, c) in enumerate(zip(axes_arr, coeffs)):
            name = f'cA{n_levels - 1}' if i == 0 else f'cD{n_levels - i}'
            ax.plot(c, 'g-', linewidth=0.5)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
        axes_arr[0].set_title(f'{title} — Wavelet Decomposition (db4)')
        axes_arr[-1].set_xlabel('Coefficient Index')

    elif plot_type == 'features':
        time_f = extract_time_features(segment)
        labels = ['MAV', 'RMS', 'WL', 'ZC', 'SSC']
        vals = [time_f.get(l, 0) for l in labels]
        max_val = max(vals) if max(vals) > 0 else 1
        vals_norm = [v / max_val for v in vals]
        axes.bar(labels, vals_norm, color='steelblue')
        axes.set_title(f'{title} — Normalized Time Features')
        axes.set_ylabel('Normalized Value')
        axes.grid(True, alpha=0.3, axis='y')
        fig = axes.figure

    else:
        plt.close(fig)
        return ''

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_title = title.replace(' ', '_').replace('/', '_')
        fname = f"{safe_title}_{plot_type}_{timestamp}.png"
        fpath = OUTPUT_DIR / fname
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(fpath)
    else:
        plt.close(fig)
        return ''


def get_feature_vector(segment) -> np.ndarray:
    """
    Extract a full feature vector suitable for classification.
    Combines time-domain, frequency-domain, and wavelet features
    into a single fixed-length vector.
    """
    tf = extract_time_features(segment)
    ff = extract_freq_features(segment)
    wf = extract_wavelet_features(segment)

    time_vec = [tf['MAV'], tf['RMS'], tf['iEMG'], tf['VAR'],
                tf['WL'], tf['ZC'], tf['SSC']]

    freq_vec = [ff['MNF'], ff['MDF'], ff['total_power'],
                ff['peak_freq'], ff['bandwidth'],
                ff['low_band_power'], ff['mid_band_power'], ff['high_band_power']]

    wavelet_vec = []
    for key in sorted(wf.keys()):
        if key.endswith('_energy') and not key.endswith('_energy_ratio'):
            wavelet_vec.append(wf[key])
        elif key.endswith('_energy_ratio'):
            wavelet_vec.append(wf[key])
        elif key.endswith('_entropy'):
            wavelet_vec.append(wf[key])

    svd_vec = wf.get('svd_top3', [0, 0, 0])
    while len(svd_vec) < 3:
        svd_vec.append(0.0)

    feature_vec = time_vec + freq_vec + wavelet_vec + svd_vec
    return np.array(feature_vec, dtype=np.float64)


TOOL_REGISTRY = {
    'get_segment_overview': {
        'fn': get_segment_overview,
        'description': 'Summary statistics and data quality for an EMG segment',
        'args': ['segment'],
    },
    'extract_time_features': {
        'fn': extract_time_features,
        'description': 'Time-domain features: MAV, RMS, iEMG, ZC, SSC, WL, VAR',
        'args': ['segment'],
    },
    'extract_freq_features': {
        'fn': extract_freq_features,
        'description': 'Frequency-domain features: MNF, MDF, PSD, spectral bands',
        'args': ['segment'],
    },
    'extract_wavelet_features': {
        'fn': extract_wavelet_features,
        'description': 'Wavelet features: db4 5-level DWT + SVD of coefficient matrix',
        'args': ['segment', 'wavelet', 'level'],
    },
    'detect_fatigue_pattern': {
        'fn': detect_fatigue_pattern,
        'description': 'Detect fatigue via MDF trend + amplitude changes',
        'args': ['segment', 'window_size_s'],
    },
    'compare_to_baseline': {
        'fn': compare_to_baseline,
        'description': 'Compare segment features to population/session baseline',
        'args': ['segment', 'baseline_stats'],
    },
    'generate_plot': {
        'fn': generate_plot,
        'description': 'Generate EMG visualization (overview/spectrum/wavelet/features)',
        'args': ['segment', 'plot_type', 'title', 'save'],
    },
    'get_feature_vector': {
        'fn': get_feature_vector,
        'description': 'Extract fixed-length feature vector for classification',
        'args': ['segment'],
    },
}


def extract_fast_features(emg: np.ndarray, fs: int = 1000) -> dict:
    """Lightweight feature extraction for streaming windows. No EMGSegment needed."""
    seg = type('Seg', (), {
        'emg': emg, 'gonio': np.zeros_like(emg),
        'fs': fs, 'duration_s': len(emg) / fs,
        'subject_id': 0, 'motion_class': 'unknown',
    })()
    time_f = extract_time_features(seg)
    freq_f = extract_freq_features(seg)
    return {**time_f, **freq_f}


if __name__ == "__main__":
    from alwaysonpt.data_loader import load_dataset

    dataset = load_dataset()
    seg = dataset.segments[0]

    print("=== Segment Overview ===")
    overview = get_segment_overview(seg)
    for k, v in overview.items():
        print(f"  {k}: {v}")

    print("\n=== Time Features ===")
    tf = extract_time_features(seg)
    for k, v in tf.items():
        print(f"  {k}: {v}")

    print("\n=== Frequency Features ===")
    ff = extract_freq_features(seg)
    for k, v in ff.items():
        print(f"  {k}: {v}")

    print("\n=== Wavelet Features ===")
    wf = extract_wavelet_features(seg)
    for k, v in list(wf.items())[:10]:
        print(f"  {k}: {v}")
    print(f"  ... ({len(wf)} total features)")

    print("\n=== Fatigue Analysis ===")
    fat = detect_fatigue_pattern(seg)
    for k, v in fat.items():
        if k not in ('window_mdfs', 'window_rmss'):
            print(f"  {k}: {v}")

    print("\n=== Feature Vector ===")
    fv = get_feature_vector(seg)
    print(f"  Length: {len(fv)}")
    print(f"  First 10: {fv[:10]}")

    print("\n=== Generating Plots ===")
    for ptype in ['overview', 'spectrum', 'wavelet', 'features']:
        path = generate_plot(seg, plot_type=ptype)
        print(f"  {ptype}: {path}")
