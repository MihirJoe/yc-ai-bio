"""
Effort / activation expert using signal amplitude and time-frequency features.

Estimates activation magnitude, contraction peaks, contraction segments.
Works well in single-channel mode (e.g. M40).
"""

import logging
import time
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.preprocessing.pipeline import (
    window_signal,
    extract_features,
    rms,
    mav,
)
from emg_model_server.types import EffortPrediction

logger = logging.getLogger(__name__)


class EffortAdapter(BaseEMGExpert):
    """Feature-based effort / activation estimator."""

    name = "effort_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_effort", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["single_channel", "multi_channel"]

    def is_available(self) -> bool:
        return True

    def validate_input(self, payload: dict[str, Any]) -> None:
        super().validate_input(payload)
        data = payload["emg_data"]
        if data.size < 50:
            raise ValueError("emg_data too short for effort estimation (need >= 50 samples)")

    def predict(self, payload: dict[str, Any]) -> EffortPrediction:
        t0 = time.perf_counter()
        try:
            self.validate_input(payload)
        except ValueError as e:
            return EffortPrediction(
                expert_name=self.name,
                task_type="effort",
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
        sig = np.asarray(sig, dtype=float)

        # Envelope: rectified + lowpass
        rectified = np.abs(sig)
        b, a = scipy_signal.butter(2, 5 / (sample_rate / 2), btype="low")
        envelope = scipy_signal.filtfilt(b, a, rectified)

        # Effort score: normalized envelope mean
        effort_score = float(np.clip(np.mean(envelope) / (np.std(envelope) + 1e-6) * 0.3, 0, 1))
        effort_score = min(1.0, effort_score)

        # Find peaks in envelope
        min_prom = np.std(envelope) * 0.3
        peaks, props = scipy_signal.find_peaks(envelope, prominence=max(min_prom, 1e-6))
        peak_events = [
            {
                "idx": int(p),
                "time_s": float(p / sample_rate),
                "amplitude": float(envelope[p]),
                "prominence": float(props["prominences"][i]) if "prominences" in props else 0,
            }
            for i, p in enumerate(peaks[:20])
        ]

        # Contraction segments: contiguous regions above threshold
        thresh = np.median(envelope) + 0.5 * np.std(envelope)
        above = envelope > thresh
        segments = []
        in_seg = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_seg:
                start = i
                in_seg = True
            elif not above[i] and in_seg:
                if i - start > int(0.1 * sample_rate):
                    segments.append({
                        "start_idx": start,
                        "end_idx": i,
                        "start_s": start / sample_rate,
                        "end_s": i / sample_rate,
                        "duration_s": (i - start) / sample_rate,
                        "mean_amplitude": float(np.mean(envelope[start:i])),
                    })
                in_seg = False
        if in_seg and len(envelope) - start > int(0.1 * sample_rate):
            segments.append({
                "start_idx": start,
                "end_idx": len(envelope),
                "start_s": start / sample_rate,
                "end_s": len(envelope) / sample_rate,
                "duration_s": (len(envelope) - start) / sample_rate,
                "mean_amplitude": float(np.mean(envelope[start:])),
            })
        contraction_segments = segments[:10]

        # Activation trace summary
        windows = window_signal(sig, sample_rate, window_size_ms=100, overlap_ratio=0.5)
        act_trace = [float(rms(w)) for w in windows] if windows else [float(rms(sig))]
        activation_trace_summary = {
            "mean": float(np.mean(act_trace)),
            "std": float(np.std(act_trace)),
            "max": float(np.max(act_trace)),
            "n_windows": len(windows),
        }

        evidence = {
            "n_peaks": len(peaks),
            "n_segments": len(segments),
            "envelope_mean": float(np.mean(envelope)),
            "envelope_std": float(np.std(envelope)),
        }

        latency_ms = (time.perf_counter() - t0) * 1000
        confidence = min(0.95, 0.6 + 0.1 * min(len(segments), 3))

        return EffortPrediction(
            expert_name=self.name,
            task_type="effort",
            status="ok",
            implementation_status="baseline",
            confidence=confidence,
            latency_ms=latency_ms,
            effort_score=effort_score,
            peak_events=peak_events,
            contraction_segments=contraction_segments,
            activation_trace_summary=activation_trace_summary,
            evidence=evidence,
        )
