"""
emg2pose adapter - Pose/kinematics expert.

Integrates the real emg2pose model (https://github.com/facebookresearch/emg2pose).
Uses the existing project at CoolProjectName/emg2pose/api.py.

Model requirements:
- Input: (batch, 16, time) or (16, time), time >= 11790 samples
- Sample rate: 2000 Hz (resampled if needed)
- Checkpoint: ~/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt
  or EMG2POSE_CHECKPOINT_PATH env var

Setup:
  pip install -e /path/to/CoolProjectName/emg2pose
  # Download checkpoint to ~/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt
  # See https://github.com/facebookresearch/emg2pose for checkpoint download
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.preprocessing.pipeline import resample
from emg_model_server.types import PosePrediction

logger = logging.getLogger(__name__)

# emg2pose constants
EMG2POSE_REQUIRED_CHANNELS = 16
EMG2POSE_MIN_SAMPLES = 11790
EMG2POSE_SAMPLE_RATE = 2000

# Try to import emg2pose
_emg2pose_inference = None
_emg2pose_import_error: str | None = None
try:
    from emg2pose.api import emg2pose_inference

    _emg2pose_inference = emg2pose_inference
except ImportError:
    # api.py lives at emg2pose project root (contains api.py), not inside the package
    # Set EMG2POSE_PROJECT_ROOT to enable real inference
    import sys

    _root = os.environ.get("EMG2POSE_PROJECT_ROOT")
    if _root:
        _root = str(Path(_root).expanduser().resolve())
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            from api import emg2pose_inference  # noqa: E402

            _emg2pose_inference = emg2pose_inference
            _emg2pose_import_error = None
        except ImportError as e2:
            _emg2pose_import_error = str(e2)
    else:
        _emg2pose_import_error = (
            "emg2pose.api not found. Set EMG2POSE_PROJECT_ROOT to the emg2pose "
            "project directory (contains api.py)"
        )


def _get_checkpoint_path() -> Path | None:
    """Return checkpoint path if it exists."""
    path = os.environ.get("EMG2POSE_CHECKPOINT_PATH")
    if path:
        p = Path(path).expanduser()
        return p if p.exists() else None
    default = Path.home() / "emg2pose_model_checkpoints" / "tracking_vemg2pose.ckpt"
    return default if default.exists() else None


def _prepare_emg_for_emg2pose(
    data: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Prepare EMG for emg2pose: (16, T) with T >= 11790, 2000 Hz.
    Resamples if needed; truncates or pads to min length.
    """
    if data.ndim == 1:
        raise ValueError("emg2pose requires 16-channel EMG, got 1 channel")
    if data.ndim == 2 and data.shape[1] != 16:
        raise ValueError(f"emg2pose requires 16 channels, got {data.shape[1]}")
    # Ensure (channels, time)
    if data.shape[0] != 16:
        data = data.T  # (time, ch) -> (ch, time) if shape was (T, 16)
    if data.shape[0] != 16:
        raise ValueError(f"emg2pose requires 16 channels, got shape {data.shape}")

    # Resample to 2000 Hz if needed
    if sample_rate != EMG2POSE_SAMPLE_RATE:
        # resample expects (samples, channels)
        data_ct = data.T  # (T, C)
        data_resampled = resample(
            data_ct,
            float(sample_rate),
            float(EMG2POSE_SAMPLE_RATE),
        )
        data = data_resampled.T.astype(np.float32)  # (C, T)
    else:
        data = data.astype(np.float32)

    n_time = data.shape[1]
    if n_time < EMG2POSE_MIN_SAMPLES:
        raise ValueError(
            f"emg2pose requires >= {EMG2POSE_MIN_SAMPLES} time samples "
            f"(at 2000 Hz), got {n_time}"
        )

    return data


class EMG2PoseAdapter(BaseEMGExpert):
    """
    Adapter for real emg2pose pose/kinematics model.

    implementation_status: "real" when model runs, "unimplemented" when blocked.
    Does NOT return mock outputs. Returns status="unavailable" with reason when
    input is incompatible (single-channel, insufficient samples) or model not set up.
    """

    name = "emg2pose_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_pose", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["multi_channel"]

    def is_available(self) -> bool:
        """True only if emg2pose is importable AND checkpoint exists."""
        if _emg2pose_inference is None:
            return False
        return _get_checkpoint_path() is not None

    def get_capability_info(self) -> dict[str, Any]:
        """Capability table for documentation."""
        ckpt = _get_checkpoint_path()
        return {
            "expert_name": self.name,
            "implementation_status": "real" if self.is_available() else "unimplemented",
            "requires_checkpoint": True,
            "checkpoint_path": str(ckpt) if ckpt else None,
            "requires_multichannel": True,
            "required_channels": EMG2POSE_REQUIRED_CHANNELS,
            "min_samples": EMG2POSE_MIN_SAMPLES,
            "required_sample_rate_hz": EMG2POSE_SAMPLE_RATE,
            "tested_on": "emg2pose dataset (16-ch forearm EMG)",
            "output_schema": "PosePrediction",
            "notes": (
                "Import failed" if _emg2pose_import_error else
                ("Checkpoint missing" if not ckpt else None)
            ),
            "import_error": _emg2pose_import_error,
        }

    def validate_input(self, payload: dict[str, Any]) -> None:
        super().validate_input(payload)
        data = payload["emg_data"]
        if data.ndim == 1 or data.shape[-1] == 1:
            raise ValueError(
                "emg2pose requires 16-channel EMG. Single-channel input is incompatible. "
                "Use benchmark mode with multi-channel data."
            )
        n_ch = data.shape[1] if data.ndim == 2 else data.shape[0]
        if n_ch != EMG2POSE_REQUIRED_CHANNELS:
            raise ValueError(
                f"emg2pose requires exactly {EMG2POSE_REQUIRED_CHANNELS} channels, got {n_ch}"
            )
        sr = payload.get("sample_rate", 1000)
        n_samples = data.shape[0] if data.ndim == 2 else data.shape[1]
        if n_samples < EMG2POSE_MIN_SAMPLES:
            raise ValueError(
                f"emg2pose requires >= {EMG2POSE_MIN_SAMPLES} samples (at model input rate), "
                f"got {n_samples} at {sr} Hz"
            )

    def predict(self, payload: dict[str, Any]) -> PosePrediction:
        t0 = time.perf_counter()

        # Block 1: Model not integrated
        if _emg2pose_inference is None:
            return PosePrediction(
                expert_name=self.name,
                task_type="pose",
                status="unavailable",
                implementation_status="unimplemented",
                reason=(
                    "emg2pose not importable. Install with: "
                    "pip install -e /path/to/CoolProjectName/emg2pose. "
                    f"Import error: {_emg2pose_import_error}"
                ),
                metadata={"blocker": "import", "import_error": _emg2pose_import_error},
            )

        # Block 2: Checkpoint missing
        ckpt = _get_checkpoint_path()
        if ckpt is None:
            return PosePrediction(
                expert_name=self.name,
                task_type="pose",
                status="unavailable",
                implementation_status="unimplemented",
                reason=(
                    "emg2pose checkpoint not found. "
                    "Download to ~/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt "
                    "or set EMG2POSE_CHECKPOINT_PATH. "
                    "See https://github.com/facebookresearch/emg2pose"
                ),
                metadata={"blocker": "checkpoint"},
            )

        # Block 3: Input incompatible (single-channel, wrong shape)
        data = payload["emg_data"]
        sample_rate = payload.get("sample_rate", 1000)
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] < EMG2POSE_REQUIRED_CHANNELS):
            return PosePrediction(
                expert_name=self.name,
                task_type="pose",
                status="unavailable",
                implementation_status="real",
                reason=(
                    f"emg2pose requires 16-channel EMG and >= {EMG2POSE_MIN_SAMPLES} samples. "
                    f"Got shape {data.shape}, {sample_rate} Hz. Use benchmark data."
                ),
                metadata={"blocker": "input_incompatible", "shape": list(data.shape)},
            )

        # Prepare input
        try:
            emg_prepared = _prepare_emg_for_emg2pose(data, sample_rate)
        except ValueError as e:
            return PosePrediction(
                expert_name=self.name,
                task_type="pose",
                status="unavailable",
                implementation_status="real",
                reason=str(e),
                metadata={"blocker": "input_validation"},
            )

        # Run real inference
        try:
            joint_angles = _emg2pose_inference(
                emg_prepared,
                checkpoint_path=str(ckpt),
            )
        except Exception as e:
            logger.exception("emg2pose inference failed: %s", e)
            return PosePrediction(
                expert_name=self.name,
                task_type="pose",
                status="error",
                implementation_status="real",
                reason=str(e),
                metadata={"blocker": "inference", "error": str(e)},
            )

        # Normalize to schema: joint_angles (batch, 20, T)
        if joint_angles.ndim == 3:
            joint_angles = joint_angles[0]
        # pose_features: flatten or summary (mean per joint over time)
        pose_features = np.mean(joint_angles, axis=1).tolist()
        temporal_trajectory_summary = {
            "shape": list(joint_angles.shape),
            "joints": 20,
            "time_samples": int(joint_angles.shape[1]),
            "sample_rate_hz": EMG2POSE_SAMPLE_RATE,
            "units": "radians",
        }

        latency_ms = (time.perf_counter() - t0) * 1000
        return PosePrediction(
            expert_name=self.name,
            task_type="pose",
            status="ok",
            implementation_status="real",
            confidence=1.0,
            latency_ms=latency_ms,
            pose_features=pose_features,
            movement_phase=None,
            temporal_trajectory_summary=temporal_trajectory_summary,
            evidence={"joint_angles_shape": list(joint_angles.shape)},
            metadata={"checkpoint": str(ckpt)},
        )
