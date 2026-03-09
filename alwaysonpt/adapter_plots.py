"""
Clinical adapter visualizations for the static demo.

Four plot types, one per adapter:
1. Fatigue Over Time (line chart with shaded zones)
2. Effort & Peak Events (gauge + scatter)
3. Joint Angle Radar Chart (spider plot)
4. Activity Timeline (horizontal Gantt)
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

PLOT_DIR = Path(__file__).parent / "static" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

_STYLE = {
    "figure.facecolor": "#0f1923",
    "axes.facecolor": "#162230",
    "axes.edgecolor": "#2a3a4a",
    "axes.labelcolor": "#c0c8d0",
    "text.color": "#c0c8d0",
    "xtick.color": "#8090a0",
    "ytick.color": "#8090a0",
    "grid.color": "#1e2e3e",
    "grid.alpha": 0.5,
    "font.size": 10,
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# ── Plot 1: Fatigue Over Time ──────────────────────────────────────

def plot_fatigue(result: dict, output_path: str | None = None) -> str:
    """Line chart: fatigue score + MDF trend with shaded severity zones."""
    _apply_style()

    segments = result.get("fatigue_segments", [])
    evidence = result.get("evidence_summary", result.get("evidence", {}))

    if not segments:
        n_win = evidence.get("n_windows", 10)
        score = result.get("fatigue_score", 0.5)
        segments = [{"window_idx": i, "fatigue_proxy": score, "median_freq_proxy": 100 - i * 2}
                    for i in range(n_win)]

    x = [s["window_idx"] for s in segments]
    fatigue = [s.get("fatigue_proxy", 0) for s in segments]
    mdf = [s.get("median_freq_proxy", 0) for s in segments]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.axhspan(0, 0.3, color="#1a4a2a", alpha=0.3)
    ax1.axhspan(0.3, 0.6, color="#4a4a1a", alpha=0.3)
    ax1.axhspan(0.6, 1.0, color="#4a1a1a", alpha=0.3)

    ax1.plot(x, fatigue, color="#ef5350", linewidth=2, marker="o", markersize=3, label="Fatigue Score")
    ax1.set_xlabel("Window Index")
    ax1.set_ylabel("Fatigue Score (0-1)", color="#ef5350")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="#ef5350")

    ax2 = ax1.twinx()
    ax2.plot(x, mdf, color="#42a5f5", linewidth=2, linestyle="--", marker="s", markersize=3, label="Median Freq")
    ax2.set_ylabel("Median Frequency (proxy)", color="#42a5f5")
    ax2.tick_params(axis="y", labelcolor="#42a5f5")

    legend_patches = [
        mpatches.Patch(color="#1a4a2a", alpha=0.5, label="Normal (0-0.3)"),
        mpatches.Patch(color="#4a4a1a", alpha=0.5, label="Watch (0.3-0.6)"),
        mpatches.Patch(color="#4a1a1a", alpha=0.5, label="Concerning (0.6-1.0)"),
    ]
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + legend_patches, labels1 + labels2 + [p.get_label() for p in legend_patches],
               loc="upper left", fontsize=8, framealpha=0.7)

    ax1.set_title("Fatigue Progression", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()

    out = output_path or str(PLOT_DIR / "fatigue.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Plot 2: Effort & Peak Events ──────────────────────────────────

def plot_effort(result: dict, output_path: str | None = None) -> str:
    """Semi-circular gauge for effort score + stem plot of peak events."""
    _apply_style()

    effort_score = result.get("effort_score", 0.5)
    peak_events = result.get("peak_events", [])
    n_segments = result.get("n_contraction_segments", 0)
    act_summary = result.get("activation_summary", {})

    fig, (ax_gauge, ax_peaks) = plt.subplots(2, 1, figsize=(8, 5),
                                              gridspec_kw={"height_ratios": [1, 1]})

    # Gauge (semi-circle)
    theta_range = np.linspace(np.pi, 0, 300)
    for t in theta_range:
        frac = 1 - (t / np.pi)
        if frac < 0.3:
            c = "#2e7d32"
        elif frac < 0.7:
            c = "#f9a825"
        else:
            c = "#c62828"
        ax_gauge.plot([0, 1.0 * np.cos(t)], [0, 1.0 * np.sin(t)], color=c, alpha=0.15, linewidth=4)

    needle_angle = np.pi * (1 - effort_score)
    ax_gauge.annotate("", xy=(0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle)),
                       xytext=(0, 0),
                       arrowprops=dict(arrowstyle="-|>", color="#e0e0e0", lw=2.5))
    ax_gauge.plot(0, 0, "o", color="#e0e0e0", markersize=6)

    ax_gauge.text(0, -0.15, f"{effort_score:.2f}", ha="center", va="top",
                  fontsize=20, fontweight="bold", color="#ffffff")
    ax_gauge.text(-1.05, -0.05, "0", ha="center", fontsize=9, color="#8090a0")
    ax_gauge.text(1.05, -0.05, "1", ha="center", fontsize=9, color="#8090a0")
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.35, 1.15)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")
    ax_gauge.set_title("Effort Score", fontsize=13, fontweight="bold", pad=5)

    badge_text = f"Segments: {n_segments}  |  Mean RMS: {act_summary.get('mean', 0):.4f}"
    ax_gauge.text(0, -0.30, badge_text, ha="center", fontsize=8, color="#8090a0")

    # Peak events stem plot
    if peak_events:
        times = [p.get("time_s", 0) for p in peak_events]
        amps = [p.get("amplitude", 0) for p in peak_events]
        markerline, stemlines, baseline = ax_peaks.stem(times, amps, basefmt=" ")
        plt.setp(stemlines, color="#42a5f5", linewidth=1.5)
        plt.setp(markerline, color="#42a5f5", markersize=6)

        if len(times) > 1:
            z = np.polyfit(times, amps, 1)
            trend_x = np.linspace(min(times), max(times), 50)
            ax_peaks.plot(trend_x, np.polyval(z, trend_x), "--", color="#ef5350", linewidth=1.5, label="Trend")
            ax_peaks.legend(fontsize=8, loc="upper right")

        ax_peaks.set_xlabel("Time (s)")
        ax_peaks.set_ylabel("Peak Amplitude")
    else:
        ax_peaks.text(0.5, 0.5, "No peaks detected", ha="center", va="center",
                      transform=ax_peaks.transAxes, color="#8090a0")

    ax_peaks.set_title("Peak Events", fontsize=11, fontweight="bold", pad=5)
    fig.tight_layout(pad=1.5)

    out = output_path or str(PLOT_DIR / "effort.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Plot 3: Joint Angle Radar Chart ───────────────────────────────

_DEFAULT_JOINT_LABELS = [
    "wrist_flex", "wrist_ext", "wrist_dev_r", "wrist_dev_u",
    "thumb_abd", "thumb_flex", "thumb_opp",
    "index_flex", "index_ext", "index_abd",
    "middle_flex", "middle_ext",
    "ring_flex", "ring_ext",
    "pinky_flex", "pinky_ext", "pinky_abd",
    "grip", "spread", "rotate",
]


def plot_pose_radar(
    result: dict,
    expected_ranges: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Radar/spider chart for 20 joint angles with expected range overlay."""
    _apply_style()

    pose_features = result.get("pose_features", [])
    n_joints = result.get("joints", len(pose_features)) or len(pose_features)

    if not pose_features:
        pose_features = [0.0] * 20
        n_joints = 20

    labels = _DEFAULT_JOINT_LABELS[:n_joints]
    if len(labels) < n_joints:
        labels += [f"j{i}" for i in range(len(labels), n_joints)]

    values = np.array(pose_features[:n_joints], dtype=float)
    v_min, v_max = values.min(), values.max()
    span = v_max - v_min if v_max > v_min else 1.0
    values_norm = (values - v_min) / span

    angles = np.linspace(0, 2 * np.pi, n_joints, endpoint=False).tolist()
    angles += angles[:1]
    values_plot = values_norm.tolist() + [values_norm[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#162230")
    fig.patch.set_facecolor("#0f1923")

    ax.plot(angles, values_plot, color="#42a5f5", linewidth=2)
    ax.fill(angles, values_plot, color="#42a5f5", alpha=0.25)

    if expected_ranges:
        exp_low = [expected_ranges.get(l, {}).get("low", 0.2) for l in labels]
        exp_high = [expected_ranges.get(l, {}).get("high", 0.8) for l in labels]
        exp_low_plot = exp_low + [exp_low[0]]
        exp_high_plot = exp_high + [exp_high[0]]
        ax.plot(angles, exp_high_plot, color="#8090a0", linewidth=1, linestyle="--")
        ax.plot(angles, exp_low_plot, color="#8090a0", linewidth=1, linestyle="--")
        ax.fill_between(angles, exp_low_plot, exp_high_plot, color="#8090a0", alpha=0.08)

        for i, l in enumerate(labels):
            rng = expected_ranges.get(l)
            if rng and (values_norm[i] < rng.get("low", 0) or values_norm[i] > rng.get("high", 1)):
                ax.plot(angles[i], values_plot[i], "o", color="#ef5350", markersize=10, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7, color="#c0c8d0")
    ax.set_yticklabels([])
    ax.set_title("Joint Angle Profile", fontsize=13, fontweight="bold", pad=20, color="#c0c8d0")
    ax.spines["polar"].set_color("#2a3a4a")
    ax.grid(color="#1e2e3e", alpha=0.5)

    fig.tight_layout()
    out = output_path or str(PLOT_DIR / "pose_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Plot 4: Activity Timeline (Gantt) ─────────────────────────────

_INTENT_COLORS = {
    "rest": "#f9a825",
    "idle": "#f9a825",
    "correct": "#2e7d32",
    "target": "#2e7d32",
    "wrong": "#c62828",
    "compensatory": "#c62828",
    "error": "#c62828",
    "transition": "#546e7a",
}


def plot_activity_timeline(
    result: dict,
    total_duration_s: float = 10.0,
    output_path: str | None = None,
) -> str:
    """Horizontal timeline with colored blocks for intent labels."""
    _apply_style()

    intent_labels = result.get("intent_labels", [])
    onsets = result.get("onset_timestamps", [])
    stability = result.get("stability_score", 0.0)

    fig, ax = plt.subplots(figsize=(10, 2.5))

    if not intent_labels:
        ax.barh(0, total_duration_s, left=0, height=0.6, color="#546e7a", edgecolor="#2a3a4a")
        ax.text(total_duration_s / 2, 0, "No activity detected", ha="center", va="center",
                fontsize=10, color="#c0c8d0")
    elif not onsets or len(onsets) == 0:
        label = intent_labels[0] if intent_labels else "unknown"
        color = _INTENT_COLORS.get(label.lower(), "#42a5f5")
        ax.barh(0, total_duration_s, left=0, height=0.6, color=color, edgecolor="#2a3a4a", alpha=0.8)
        ax.text(total_duration_s / 2, 0, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#ffffff")
    else:
        times = sorted(onsets)
        all_labels = intent_labels if len(intent_labels) >= len(times) else intent_labels + ["unknown"] * (len(times) - len(intent_labels) + 1)

        boundaries = [0.0] + times + [total_duration_s]
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            width = boundaries[i + 1] - start
            label = all_labels[i] if i < len(all_labels) else "unknown"
            color = _INTENT_COLORS.get(label.lower(), "#42a5f5")
            ax.barh(0, width, left=start, height=0.6, color=color, edgecolor="#1a2a3a", alpha=0.85)
            if width > total_duration_s * 0.05:
                ax.text(start + width / 2, 0, label, ha="center", va="center",
                        fontsize=8, fontweight="bold", color="#ffffff")

    ax.set_xlim(0, total_duration_s)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])

    tick_interval = 5 if total_duration_s > 20 else (2 if total_duration_s > 5 else 1)
    ax.set_xticks(np.arange(0, total_duration_s + 0.1, tick_interval))

    stab_text = f"Stability: {stability:.2f}" if stability else ""
    ax.text(0.98, 0.95, stab_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#8090a0", bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f1923", edgecolor="#2a3a4a"))

    ax.set_title("Activity Timeline", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()

    out = output_path or str(PLOT_DIR / "activity_timeline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Convenience: generate all plots for adapter results ────────────

def generate_adapter_plots(
    adapter_results: dict[str, dict[str, Any]],
    total_duration_s: float = 10.0,
    expected_pose_ranges: dict | None = None,
) -> dict[str, str]:
    """Generate all relevant plots and return {adapter_name: plot_path}."""
    plots = {}

    if "fatigue" in adapter_results and adapter_results["fatigue"].get("status") == "ok":
        plots["fatigue"] = plot_fatigue(adapter_results["fatigue"])

    if "effort" in adapter_results and adapter_results["effort"].get("status") == "ok":
        plots["effort"] = plot_effort(adapter_results["effort"])

    if "pose" in adapter_results and adapter_results["pose"].get("status") == "ok":
        plots["pose"] = plot_pose_radar(adapter_results["pose"], expected_pose_ranges)

    if "intent" in adapter_results and adapter_results["intent"].get("status") == "ok":
        plots["intent"] = plot_activity_timeline(adapter_results["intent"], total_duration_s)

    return plots
