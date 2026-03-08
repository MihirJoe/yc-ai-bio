"""
emg_gesture_adapter - Intent/gesture expert using SilvaUnCompte model.

Uses the pretrained MLP/sklearn pipeline from:
https://github.com/SilvaUnCompte/EMG-Gesture-Recognition-System

Requirements:
- 8-channel EMG (Myo armband format)
- EMG_GESTURE_MODEL_DIR: path to model dir (contains pipeline.joblib, config.json)
- Example: clone repo, use models-example/gesture_cls/1.0.0_20250821T094534Z/
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.preprocessing.pipeline import window_signal
from emg_model_server.types import IntentPrediction

logger = logging.getLogger(__name__)

REQUIRED_CHANNELS = 8

_pipeline = None
_config = None
_import_error: str | None = None


def _load_model() -> tuple[Any, dict] | None:
    """Load joblib pipeline and config. Returns (pipe, cfg) or None."""
    global _pipeline, _config, _import_error
    if _pipeline is not None:
        return _pipeline, _config

    model_dir = os.environ.get("EMG_GESTURE_MODEL_DIR")
    if not model_dir:
        _import_error = "EMG_GESTURE_MODEL_DIR not set"
        return None

    path = Path(model_dir).expanduser().resolve()
    pipe_file = path / "pipeline.joblib"
    config_file = path / "config.json"
    if not pipe_file.exists() or not config_file.exists():
        _import_error = f"Model files not found in {path}"
        return None

    try:
        import joblib
        import json

        _pipeline = joblib.load(pipe_file)
        with open(config_file) as f:
            _config = json.load(f)
        _import_error = None
        return _pipeline, _config
    except Exception as e:
        _import_error = str(e)
        return None


class EMGGestureAdapter(BaseEMGExpert):
    """
    Real intent/gesture adapter using SilvaUnCompte pretrained model.
    Requires 8-channel EMG, EMG_GESTURE_MODEL_DIR.
    """

    name = "emg_gesture_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_intent", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["multi_channel"]

    def is_available(self) -> bool:
        return _load_model() is not None

    def get_capability_info(self) -> dict[str, Any]:
        loaded = _load_model()
        model_dir = os.environ.get("EMG_GESTURE_MODEL_DIR")
        return {
            "expert_name": self.name,
            "implementation_status": "real" if loaded else "unimplemented",
            "requires_checkpoint": True,
            "model_dir": model_dir,
            "requires_multichannel": True,
            "required_channels": REQUIRED_CHANNELS,
            "tested_on": "Myo armband 8-channel (SilvaUnCompte)",
            "output_schema": "IntentPrediction",
            "notes": _import_error,
        }

    def validate_input(self, payload: dict[str, Any]) -> None:
        super().validate_input(payload)
        data = payload["emg_data"]
        n_ch = data.shape[1] if data.ndim == 2 else 1
        if n_ch < REQUIRED_CHANNELS:
            raise ValueError(
                f"emg_gesture_adapter requires >= {REQUIRED_CHANNELS} channels, got {n_ch}"
            )

    def predict(self, payload: dict[str, Any]) -> IntentPrediction:
        t0 = time.perf_counter()

        loaded = _load_model()
        if loaded is None:
            return IntentPrediction(
                expert_name=self.name,
                task_type="intent",
                status="unavailable",
                implementation_status="unimplemented",
                reason=(
                    f"emg_gesture_adapter not available. {_import_error}. "
                    "Set EMG_GESTURE_MODEL_DIR to model dir (pipeline.joblib, config.json). "
                    "See https://github.com/SilvaUnCompte/EMG-Gesture-Recognition-System"
                ),
                metadata={"blocker": "model"},
            )

        pipe, cfg = loaded
        data = payload["emg_data"]
        sample_rate = payload.get("sample_rate", 1000)
        if data.ndim == 1:
            return IntentPrediction(
                expert_name=self.name,
                task_type="intent",
                status="unavailable",
                implementation_status="real",
                reason=f"Requires {REQUIRED_CHANNELS}-channel EMG, got 1",
                metadata={"blocker": "input"},
            )

        n_ch = data.shape[1]
        if n_ch < REQUIRED_CHANNELS:
            return IntentPrediction(
                expert_name=self.name,
                task_type="intent",
                status="unavailable",
                implementation_status="real",
                reason=f"Requires >= {REQUIRED_CHANNELS} channels, got {n_ch}",
                metadata={"blocker": "input"},
            )

        # Use first 8 channels when input has more (e.g. 16-channel emg2pose data)
        data = data[:, :REQUIRED_CHANNELS]

        try:
            import pandas as pd
        except ImportError:
            return IntentPrediction(
                expert_name=self.name,
                task_type="intent",
                status="error",
                implementation_status="real",
                reason="pandas required for emg_gesture_adapter",
                metadata={"blocker": "import"},
            )

        feature_names = cfg.get("feature_names", [f"EMG{i+1}" for i in range(8)])
        class_names = cfg.get("class_names", [])
        abstain_threshold = cfg.get("abstain_threshold", 0.6)

        # Window the signal, extract mean per channel per window
        windows = window_signal(
            data,
            sample_rate,
            window_size_ms=200,
            overlap_ratio=0.5,
        )
        if not windows:
            windows = [data[-1:]] if data.ndim == 2 else [data[-1:].reshape(1, -1)]

        intent_labels_over_time: list[tuple[float, str]] = []
        segment_confidence: list[float] = []
        all_labels: set[str] = set()

        for i, w in enumerate(windows):
            if w.ndim == 1:
                continue
            if w.shape[1] < REQUIRED_CHANNELS:
                continue
            w = w[:, :REQUIRED_CHANNELS]
            features = np.mean(w, axis=0).astype(np.float64)
            features_df = pd.DataFrame([features], columns=feature_names)
            probs = pipe.predict_proba(features_df)[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            label = class_names[top_idx] if top_idx < len(class_names) else "unknown"
            if top_prob < abstain_threshold:
                label = "unknown"
            t_s = i * 0.2 * 0.5  # approximate time
            intent_labels_over_time.append((t_s, label))
            segment_confidence.append(top_prob)
            all_labels.add(label)

        # Onset timestamps (transitions)
        onset_timestamps: list[float] = []
        for i in range(1, len(intent_labels_over_time)):
            if intent_labels_over_time[i][1] != intent_labels_over_time[i - 1][1]:
                onset_timestamps.append(intent_labels_over_time[i][0])

        stability_score = (
            float(np.mean(segment_confidence)) if segment_confidence else 0.0
        )
        confidence = stability_score

        return IntentPrediction(
            expert_name=self.name,
            task_type="intent",
            status="ok",
            implementation_status="real",
            confidence=confidence,
            latency_ms=(time.perf_counter() - t0) * 1000,
            intent_labels=list(all_labels) if all_labels else ["unknown"],
            intent_labels_over_time=intent_labels_over_time[:100],
            onset_timestamps=onset_timestamps[:20],
            segment_confidence=segment_confidence[:20],
            stability_score=stability_score,
            evidence={
                "n_windows": len(windows),
                "class_names": class_names,
                "channels_used": f"0:{REQUIRED_CHANNELS}" if n_ch > REQUIRED_CHANNELS else "all",
            },
            metadata={"model_dir": os.environ.get("EMG_GESTURE_MODEL_DIR")},
        )
