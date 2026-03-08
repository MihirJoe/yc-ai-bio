"""
Fatigue expert using feature-based estimator.

Uses RMS, MAV, waveform length, zero crossing, spectral features,
median frequency proxy, and window trend analysis.
Works well in single-channel mode (e.g. M40).
"""

import logging
import time
from typing import Any

import numpy as np

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.preprocessing.pipeline import (
    window_signal,
    extract_features,
    rms,
    mav,
    waveform_length,
    median_frequency_proxy,
)
from emg_model_server.types import FatiguePrediction

logger = logging.getLogger(__name__)


class FatigueAdapter(BaseEMGExpert):
    """Feature-based fatigue estimator."""

    name = "fatigue_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_fatigue", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["single_channel", "multi_channel"]

    def is_available(self) -> bool:
        return True

    def validate_input(self, payload: dict[str, Any]) -> None:
        super().validate_input(payload)
        data = payload["emg_data"]
        if data.size < 100:
            raise ValueError("emg_data too short for fatigue estimation (need >= 100 samples)")

    def predict(self, payload: dict[str, Any]) -> FatiguePrediction:
        t0 = time.perf_counter()
        try:
            self.validate_input(payload)
        except ValueError as e:
            return FatiguePrediction(
                expert_name=self.name,
                task_type="fatigue",
                status="error",
                implementation_status="baseline",
                confidence=0.0,
                latency_ms=0,
                evidence={"error": str(e)},
            )

        data = payload["emg_data"]
        sample_rate = payload.get("sample_rate", 1000)
        channel = payload.get("channel", 0)
        if data.ndim > 1:
            sig = data[:, channel]
        else:
            sig = data.flatten()

        windows = window_signal(sig, sample_rate, window_size_ms=200, overlap_ratio=0.5)
        if not windows:
            return FatiguePrediction(
                expert_name=self.name,
                task_type="fatigue",
                status="ok",
                implementation_status="baseline",
                confidence=0.3,
                fatigue_score=0.0,
                fatigue_trend="unknown",
                fatigue_segments=[],
                evidence={"n_windows": 0},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        feats_per_window: list[dict[str, float]] = []
        for w in windows:
            feats_per_window.append(extract_features(w, sample_rate, 0))

        # Aggregate features across windows
        rms_vals = [f["rms"] for f in feats_per_window]
        mav_vals = [f["mav"] for f in feats_per_window]
        mf_vals = [f["median_freq_proxy"] for f in feats_per_window]
        wl_vals = [f["waveform_length"] for f in feats_per_window]
        zc_vals = [f["zero_crossing_rate"] for f in feats_per_window]

        # Fatigue heuristics:
        # - Increasing RMS/MAV over time -> compensatory activation
        # - Decreasing median freq -> spectral shift
        # - Decreasing zero crossing -> fewer fast motor units
        rms_trend = np.polyfit(np.arange(len(rms_vals)), rms_vals, 1)[0] if len(rms_vals) > 1 else 0
        mf_trend = np.polyfit(np.arange(len(mf_vals)), mf_vals, 1)[0] if len(mf_vals) > 1 else 0
        zc_trend = np.polyfit(np.arange(len(zc_vals)), zc_vals, 1)[0] if len(zc_vals) > 1 else 0

        # Combined fatigue score: higher when RMS rises, MF drops, ZC drops
        rms_norm = (rms_trend + 0.01) / 0.02 if abs(rms_trend) > 1e-6 else 0
        mf_norm = (-mf_trend) / max(np.std(mf_vals) + 1e-6, 0.01)
        zc_norm = (-zc_trend) / max(np.std(zc_vals) + 1e-6, 0.001)
        raw_score = np.clip(0.3 * np.tanh(rms_norm) + 0.4 * np.tanh(mf_norm) + 0.3 * np.tanh(zc_norm), -1, 1)
        fatigue_score = float((raw_score + 1) / 2)

        if rms_trend > 0.001 and mf_trend < -0.1:
            fatigue_trend = "increasing"
        elif rms_trend < -0.001 and mf_trend > 0.1:
            fatigue_trend = "decreasing"
        else:
            fatigue_trend = "stable"

        fatigue_segments = [
            {
                "window_idx": i,
                "rms": rms_vals[i],
                "median_freq_proxy": mf_vals[i],
                "fatigue_proxy": float(np.clip((1 - mf_vals[i] / (max(mf_vals) + 1e-6)), 0, 1)),
            }
            for i in range(min(5, len(feats_per_window)))
        ]

        evidence = {
            "n_windows": len(windows),
            "rms_trend": float(rms_trend),
            "median_freq_trend": float(mf_trend),
            "zero_crossing_trend": float(zc_trend),
            "rms_mean": float(np.mean(rms_vals)),
            "median_freq_mean": float(np.mean(mf_vals)),
        }

        latency_ms = (time.perf_counter() - t0) * 1000
        confidence = min(0.95, 0.5 + 0.15 * len(windows))

        return FatiguePrediction(
            expert_name=self.name,
            task_type="fatigue",
            status="ok",
            implementation_status="baseline",
            confidence=confidence,
            latency_ms=latency_ms,
            fatigue_score=fatigue_score,
            fatigue_trend=fatigue_trend,
            fatigue_segments=fatigue_segments,
            evidence=evidence,
        )
