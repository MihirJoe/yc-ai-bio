"""
Always On PT — Gait Analyzer.

Two-stage pipeline for live EMG gait analysis:
  1. analyze_activation() — deterministic signal processing (burst detection,
     stride timing, fatigue trends)
  2. get_gait_assessment() — streams a single Claude call with a gait-specific
     system prompt for clinical interpretation

Used by the Live Sensor tab in the dashboard.
"""

import json
import os
import anthropic
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import AsyncGenerator

from alwaysonpt.task_prompts import get_task_config

PLOTS_DIR = Path(__file__).parent / "static" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_activation(values: list, duration_s: float) -> dict:
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    fs = n / duration_s if duration_s > 0 else 1.0

    if n < 3:
        return {"error": "Signal too short"}

    baseline = float(np.percentile(arr, 10))
    peak = float(np.max(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    dynamic_range = peak - float(np.min(arr))

    thresh = baseline + std_val
    active = arr > thresh
    active_pct = float(np.mean(active) * 100)

    changes = np.diff(active.astype(int))
    burst_starts = np.where(changes == 1)[0]
    burst_ends = np.where(changes == -1)[0]

    if active[0]:
        burst_starts = np.concatenate([[0], burst_starts])
    if active[-1]:
        burst_ends = np.concatenate([burst_ends, [n - 1]])

    bursts = []
    for i in range(min(len(burst_starts), len(burst_ends))):
        s, e = burst_starts[i], burst_ends[i]
        dur = float((e - s) / fs)
        if dur > 0.1:
            segment = arr[s:e + 1]
            bursts.append({
                "start_s": round(float(s / fs), 3),
                "end_s": round(float(e / fs), 3),
                "duration_s": round(dur, 3),
                "peak_value": int(np.max(segment)),
                "mean_value": round(float(np.mean(segment)), 1),
            })

    active_time = float(np.sum(active) / fs)
    rest_time = float(duration_s - active_time)
    wr_ratio = active_time / rest_time if rest_time > 0 else float("inf")

    n_windows = min(n, 8)
    win_size = max(n // n_windows, 1)
    window_means = []
    for i in range(n_windows):
        seg_arr = arr[i * win_size:(i + 1) * win_size]
        if len(seg_arr) > 0:
            window_means.append(float(np.mean(seg_arr)))

    if len(window_means) >= 3:
        t = np.arange(len(window_means))
        slope = float(np.polyfit(t, window_means, 1)[0])
        pct_change = (
            (window_means[-1] - window_means[0]) / window_means[0] * 100
            if window_means[0] > 0 else 0
        )
        if slope < -0.5:
            fatigue_interp = (
                f"Declining activation trend ({pct_change:+.1f}%) — "
                f"may indicate fatigue or reduced effort over time"
            )
        elif slope > 0.5:
            fatigue_interp = (
                f"Increasing activation trend ({pct_change:+.1f}%) — "
                f"muscle engagement increasing over time"
            )
        else:
            fatigue_interp = f"Stable activation level ({pct_change:+.1f}%)"
    else:
        slope, pct_change = 0, 0
        fatigue_interp = "Insufficient data for trend analysis"

    if len(bursts) >= 2:
        onsets = [b["start_s"] for b in bursts]
        stride_times = [round(onsets[i + 1] - onsets[i], 3) for i in range(len(onsets) - 1)]
        stride_mean = float(np.mean(stride_times))
        stride_std = float(np.std(stride_times))
        stride_cv = stride_std / stride_mean * 100 if stride_mean > 0 else 0
        cadence = 60.0 / stride_mean if stride_mean > 0 else 0
    else:
        stride_times = []
        stride_mean = stride_std = stride_cv = cadence = 0

    if bursts:
        b_peaks = [b["peak_value"] for b in bursts]
        b_durs = [b["duration_s"] for b in bursts]
        peak_cv = float(np.std(b_peaks) / np.mean(b_peaks) * 100) if np.mean(b_peaks) > 0 else 0
        dur_cv = float(np.std(b_durs) / np.mean(b_durs) * 100) if np.mean(b_durs) > 0 else 0
        gaps = []
        for i in range(1, len(bursts)):
            gaps.append(round(bursts[i]["start_s"] - bursts[i - 1]["end_s"], 3))
    else:
        b_peaks, b_durs = [], []
        peak_cv = dur_cv = 0
        gaps = []

    return {
        "signal_info": {
            "n_samples": n,
            "duration_s": round(duration_s, 2),
            "effective_sampling_rate_hz": round(fs, 2),
        },
        "activation_level": {
            "baseline": round(baseline, 1),
            "peak": round(peak, 1),
            "mean": round(mean_val, 1),
            "std": round(std_val, 2),
            "dynamic_range": round(dynamic_range, 1),
            "coefficient_of_variation_pct": round(std_val / mean_val * 100, 2) if mean_val > 0 else 0,
        },
        "contraction_detection": {
            "threshold": round(thresh, 1),
            "active_time_pct": round(active_pct, 1),
            "burst_count": len(bursts),
            "bursts": bursts,
        },
        "stride_analysis": {
            "stride_times_s": stride_times,
            "stride_time_mean_s": round(stride_mean, 3),
            "stride_time_std_s": round(stride_std, 3),
            "stride_time_cv_pct": round(stride_cv, 1),
            "cadence_strides_per_min": round(cadence, 1),
        },
        "burst_variability": {
            "peak_values": b_peaks,
            "peak_cv_pct": round(peak_cv, 1),
            "durations_s": b_durs,
            "duration_cv_pct": round(dur_cv, 1),
            "inter_burst_gaps_s": gaps,
        },
        "work_rest": {
            "active_time_s": round(active_time, 2),
            "rest_time_s": round(rest_time, 2),
            "work_rest_ratio": round(wr_ratio, 2),
        },
        "fatigue_trend": {
            "window_means": [round(v, 1) for v in window_means],
            "slope_per_window": round(slope, 4),
            "pct_change": round(pct_change, 1),
            "interpretation": fatigue_interp,
        },
    }


def generate_gait_plot(emg_signal: list, result: dict, record_id: str = "live") -> str:
    """
    Generate a 3-panel gait analysis figure from the result metrics + raw EMG.
    Returns the URL path to the saved plot (relative to static root).
    """
    ma = result.get("muscle_activation", {})
    if not ma or "error" in ma:
        return ""

    si = ma["signal_info"]
    al = ma["activation_level"]
    cd = ma["contraction_detection"]
    sa = ma["stride_analysis"]
    bv = ma["burst_variability"]
    ft = ma["fatigue_trend"]

    arr = np.array(emg_signal, dtype=np.float64)
    duration = si["duration_s"]
    fs = si["effective_sampling_rate_hz"]
    n = len(arr)
    t = np.linspace(0, duration, n)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.patch.set_facecolor("#0b0f14")
    for ax in axes:
        ax.set_facecolor("#0e1218")
        ax.tick_params(colors="#6b7a8d", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#232d3d")

    # --- Panel 1: EMG signal with burst highlighting ---
    ax1 = axes[0]
    ax1.plot(t, arr, color="#50c8c8", linewidth=0.6, alpha=0.9)
    ax1.axhline(al["baseline"], color="#6b7a8d", linewidth=0.5, linestyle="--", alpha=0.6)
    ax1.axhline(cd["threshold"], color="#e6983a", linewidth=0.5, linestyle="--", alpha=0.6)
    ax1.text(duration * 0.01, al["baseline"], "baseline", fontsize=7, color="#6b7a8d", va="bottom")
    ax1.text(duration * 0.01, cd["threshold"], "threshold", fontsize=7, color="#e6983a", va="bottom")

    bursts = cd.get("bursts", [])
    for b in bursts:
        ax1.axvspan(b["start_s"], b["end_s"], alpha=0.15, color="#3b9eff")
        ax1.plot(
            [b["start_s"] + (b["end_s"] - b["start_s"]) / 2],
            [b["peak_value"]],
            "v", color="#3b9eff", markersize=4, alpha=0.8,
        )

    ax1.set_ylabel("Activation", fontsize=9, color="#c8ced8")
    ax1.set_title(
        f"EMG Gait Recording — {len(bursts)} bursts, "
        f"cadence {sa['cadence_strides_per_min']:.0f}/min",
        fontsize=10, color="#c8ced8", pad=8,
    )
    ax1.set_xlim(0, duration)

    # --- Panel 2: Stride intervals ---
    ax2 = axes[1]
    stride_times = sa.get("stride_times_s", [])
    if len(stride_times) >= 1:
        stride_x = [bursts[i]["start_s"] for i in range(1, min(len(bursts), len(stride_times) + 1))]
        mean_st = sa["stride_time_mean_s"]
        std_st = sa["stride_time_std_s"]

        ax2.bar(stride_x, stride_times, width=mean_st * 0.4, color="#3b9eff", alpha=0.7, edgecolor="#3b9eff")
        ax2.axhline(mean_st, color="#2ecc71", linewidth=1, linestyle="-", alpha=0.8)
        ax2.fill_between(
            [0, duration], mean_st - std_st, mean_st + std_st,
            alpha=0.08, color="#2ecc71",
        )
        ax2.text(
            duration * 0.98, mean_st,
            f"mean={mean_st:.2f}s  CV={sa['stride_time_cv_pct']:.1f}%",
            fontsize=7, color="#2ecc71", ha="right", va="bottom",
        )
        ax2.set_xlim(0, duration)
    else:
        ax2.text(0.5, 0.5, "Not enough bursts for stride analysis",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9, color="#6b7a8d")

    ax2.set_ylabel("Stride (s)", fontsize=9, color="#c8ced8")

    # --- Panel 3: Fatigue trend (windowed means) ---
    ax3 = axes[2]
    window_means = ft.get("window_means", [])
    if len(window_means) >= 2:
        wx = np.linspace(0, duration, len(window_means))
        ax3.plot(wx, window_means, "o-", color="#e6983a", markersize=4, linewidth=1.5, alpha=0.9)
        z = np.polyfit(np.arange(len(window_means)), window_means, 1)
        trend_y = np.polyval(z, np.arange(len(window_means)))
        ax3.plot(wx, trend_y, "--", color="#e85454" if z[0] < -0.5 else "#2ecc71", linewidth=1, alpha=0.7)
        ax3.text(
            duration * 0.98, window_means[-1],
            f"{ft['pct_change']:+.1f}%",
            fontsize=8, color="#e6983a", ha="right", va="bottom",
        )
        ax3.set_xlim(0, duration)
    else:
        ax3.text(0.5, 0.5, "Insufficient data for trend",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=9, color="#6b7a8d")

    ax3.set_ylabel("Mean Act.", fontsize=9, color="#c8ced8")
    ax3.set_xlabel("Time (s)", fontsize=9, color="#c8ced8")

    plt.tight_layout(h_pad=1.0)

    safe_id = record_id.replace("/", "_")
    filename = f"gait_{safe_id}.png"
    filepath = PLOTS_DIR / filename
    fig.savefig(str(filepath), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return f"/plots/{filename}"


def process_recording(data: dict) -> dict:
    """Run analyze_activation on an in-memory recording dict."""
    duration = data.get("duration", 0)
    emg_signal = data.get("emgSignal", [])

    if not emg_signal:
        return {"error": "No emgSignal found in recording"}

    return {
        "recording_info": {
            "source_file": data.get("filename", "live_recording"),
            "duration_s": duration,
            "recorded_at": data.get("recordedAt"),
            "stated_sampling_rate": data.get("samplingRate"),
            "emg_samples": len(emg_signal),
        },
        "muscle_activation": analyze_activation(emg_signal, duration),
    }


def get_gait_assessment_sync(result: dict) -> str:
    """Non-streaming Claude call — returns the full assessment text."""
    config = get_task_config("live_gait_analysis")
    system_prompt = config["system_prompt"]

    user_message = (
        "Walking recording. Assess gait quality.\n\n"
        + json.dumps(result, indent=2, default=str)
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.3,
    )

    return response.content[0].text


def stream_gait_assessment(result: dict):
    """
    Streaming Claude call — yields text chunks as they arrive.
    Runs synchronously (meant to be called from a thread).
    """
    config = get_task_config("live_gait_analysis")
    system_prompt = config["system_prompt"]

    user_message = (
        "Walking recording. Assess gait quality.\n\n"
        + json.dumps(result, indent=2, default=str)
    )

    client = anthropic.Anthropic()
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.3,
    ) as stream:
        for text in stream.text_stream:
            yield text
