"""
ReactEMG adapter - Intent/state expert.

Wraps ReactEMG (https://reactemg.github.io/) when available.
Falls back to mock or lightweight baseline classifier on hand-engineered EMG features.

TODO (real integration):
- Install: pip install reactemg (or clone https://github.com/roamlab/reactemg)
- Model path: set REACTEMG_MODEL_PATH env
- Expected input: multi-channel forearm EMG
- Output: intent labels, onset timestamps, stability
"""

import logging
import os
import time
from typing import Any

import numpy as np

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.preprocessing.pipeline import (
    extract_features,
    window_signal,
    rms,
    mav,
)
from emg_model_server.types import IntentPrediction

logger = logging.getLogger(__name__)

_reactemg_available = False
try:
    # import reactemg
    # _reactemg_available = True
    pass
except Exception:
    pass


class ReactEMGAdapter(BaseEMGExpert):
    """Adapter for ReactEMG intent/state model."""

    name = "reactemg_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_intent", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["multi_channel", "single_channel"]  # baseline works single-channel

    def is_available(self) -> bool:
        return True  # Baseline always available

    def validate_input(self, payload: dict[str, Any]) -> None:
        super().validate_input(payload)

    def _predict_baseline(self, payload: dict[str, Any]) -> IntentPrediction:
        """Lightweight baseline: threshold-based activation segments."""
        data = payload["emg_data"]
        sample_rate = payload.get("sample_rate", 1000)
        channel = 0
        if data.ndim > 1:
            sig = data[:, channel]
        else:
            sig = data.flatten()

        windows = window_signal(sig, sample_rate, window_size_ms=100, overlap_ratio=0.5)
        if not windows:
            return IntentPrediction(
                expert_name=self.name,
                task_type="intent",
                status="ok",
                implementation_status="baseline",
                confidence=0.4,
                intent_labels=["rest"],
                intent_labels_over_time=[],
                onset_timestamps=[],
                segment_confidence=[],
                stability_score=0.5,
                metadata={"mode": "baseline"},
            )

        rms_vals = [float(rms(w)) for w in windows]
        thresh = np.median(rms_vals) + 0.5 * np.std(rms_vals)
        labels = ["active" if r > thresh else "rest" for r in rms_vals]
        step_s = 0.1 * (1 - 0.5)
        intent_labels_over_time = [
            (i * step_s, labels[i]) for i in range(len(labels))
        ]

        # Onset timestamps
        onsets = []
        for i in range(1, len(labels)):
            if labels[i] == "active" and labels[i - 1] == "rest":
                onsets.append(i * step_s)
        onset_timestamps = onsets[:10]
        segment_confidence = [0.6 + 0.2 * (r > thresh) for r in rms_vals[:10]]
        stability_score = 0.6

        return IntentPrediction(
            expert_name=self.name,
            task_type="intent",
            status="ok",
            implementation_status="baseline",
            confidence=0.6,
            intent_labels=list(set(labels)),
            intent_labels_over_time=intent_labels_over_time[:50],
            onset_timestamps=onset_timestamps,
            segment_confidence=segment_confidence,
            stability_score=stability_score,
            metadata={"mode": "baseline"},
        )

    def _predict_real(self, payload: dict[str, Any]) -> IntentPrediction | None:
        """Run real ReactEMG. Returns None if not implemented."""
        return None

    def predict(self, payload: dict[str, Any]) -> IntentPrediction:
        t0 = time.perf_counter()
        mode = payload.get("mode", "auto")
        is_single_channel = (
            payload["emg_data"].ndim == 1 or payload["emg_data"].shape[1] == 1
        )

        if _reactemg_available and not is_single_channel and mode != "live_lite":
            out = self._predict_real(payload)
            if out is not None:
                out.latency_ms = (time.perf_counter() - t0) * 1000
                return out

        out = self._predict_baseline(payload)
        out.latency_ms = (time.perf_counter() - t0) * 1000
        return out
