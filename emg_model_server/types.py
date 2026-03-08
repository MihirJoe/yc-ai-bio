"""Pydantic schemas for structured inputs and outputs."""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# --- Input schemas ---


class EMGInput(BaseModel):
    """Structured EMG signal input supporting single- or multi-channel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray | None = Field(
        default=None,
        description="Raw EMG data as (samples,) or (samples, channels)",
    )
    file_path: str | None = Field(
        default=None,
        description="Path to CSV, NPY, or NPZ file",
    )
    sample_rate: int = Field(default=1000, ge=1, description="Samples per second")
    channel_names: list[str] | None = Field(
        default=None,
        description="Names for each channel, e.g. ['ch0'] or ['flexor', 'extensor']",
    )
    timestamps: np.ndarray | None = Field(
        default=None,
        description="Optional timestamps for each sample",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_single_channel(self) -> bool:
        """True if input is single-channel (e.g. M40)."""
        if self.data is not None:
            return self.data.ndim == 1 or self.data.shape[1] == 1
        return True  # assume single if only path given

    @property
    def num_channels(self) -> int:
        """Number of EMG channels."""
        if self.data is not None:
            return 1 if self.data.ndim == 1 else self.data.shape[1]
        return 1


class EMGWindow(BaseModel):
    """A windowed segment of EMG data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray = Field(description="Windowed EMG (window_len,) or (window_len, ch)")
    start_idx: int = 0
    end_idx: int = 0
    sample_rate: int = 1000
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiModalInput(BaseModel):
    """Combined multi-modal input for experts."""

    emg: EMGInput | EMGWindow
    optional_modalities: dict[str, Any] = Field(default_factory=dict)


# --- Base expert output ---


class ExpertPrediction(BaseModel):
    """Base schema for expert predictions."""

    expert_name: str = ""
    task_type: str = ""
    status: str = Field(
        ...,
        description="ok | error | unavailable",
    )
    implementation_status: str = Field(
        default="unimplemented",
        description="real | baseline | unimplemented",
    )
    confidence: float = Field(default=0.0, ge=0, le=1)
    latency_ms: float = 0.0
    reason: str | None = Field(
        default=None,
        description="Blocker/setup message when status=unavailable",
    )
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExpertError(BaseModel):
    """Error information when an expert fails."""

    expert_name: str = ""
    error_type: str = ""
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# --- Specialized prediction schemas ---


class PosePrediction(ExpertPrediction):
    """Pose / kinematics expert output."""

    task_type: str = "pose"
    pose_features: list[float] = Field(default_factory=list)
    movement_phase: str | None = None
    temporal_trajectory_summary: dict[str, Any] = Field(default_factory=dict)

    def format_output(self) -> dict[str, Any]:
        """Clean output for agent consumption."""
        return {
            "adapter": "pose",
            "status": self.status,
            "implementation_status": self.implementation_status,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "pose_features": self.pose_features,
            "joints": self.temporal_trajectory_summary.get("joints"),
            "trajectory": self.temporal_trajectory_summary,
            "reason": self.reason,
        }


class IntentPrediction(ExpertPrediction):
    """Intent / state expert output."""

    task_type: str = "intent"
    intent_labels: list[str] = Field(default_factory=list)
    intent_labels_over_time: list[tuple[float, str]] = Field(default_factory=list)
    onset_timestamps: list[float] = Field(default_factory=list)
    segment_confidence: list[float] = Field(default_factory=list)
    stability_score: float = 0.0

    def format_output(self) -> dict[str, Any]:
        """Clean output for agent consumption."""
        return {
            "adapter": "intent",
            "status": self.status,
            "implementation_status": self.implementation_status,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "intent_labels": self.intent_labels,
            "onset_timestamps": self.onset_timestamps[:10],
            "stability_score": self.stability_score,
            "reason": self.reason,
        }


class FatiguePrediction(ExpertPrediction):
    """Fatigue / drift expert output."""

    task_type: str = "fatigue"
    fatigue_score: float = 0.0
    fatigue_trend: str = ""  # e.g. "increasing", "stable", "decreasing"
    fatigue_segments: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    def format_output(self) -> dict[str, Any]:
        """Clean output for agent consumption."""
        return {
            "adapter": "fatigue",
            "status": self.status,
            "implementation_status": self.implementation_status,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "fatigue_score": self.fatigue_score,
            "fatigue_trend": self.fatigue_trend,
            "evidence_summary": {k: v for k, v in self.evidence.items() if k in ("n_windows", "rms_trend", "median_freq_trend")},
            "reason": self.reason,
        }


class EffortPrediction(ExpertPrediction):
    """Effort / activation expert output."""

    task_type: str = "effort"
    effort_score: float = 0.0
    peak_events: list[dict[str, Any]] = Field(default_factory=list)
    contraction_segments: list[dict[str, Any]] = Field(default_factory=list)
    activation_trace_summary: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)

    def format_output(self) -> dict[str, Any]:
        """Clean output for agent consumption."""
        return {
            "adapter": "effort",
            "status": self.status,
            "implementation_status": self.implementation_status,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "effort_score": self.effort_score,
            "n_peaks": len(self.peak_events),
            "peak_events": [{"time_s": p.get("time_s"), "amplitude": p.get("amplitude")} for p in self.peak_events[:5]],
            "n_contraction_segments": len(self.contraction_segments),
            "activation_summary": self.activation_trace_summary,
            "reason": self.reason,
        }


# --- Aggregated results ---


class AggregatedExpertResult(BaseModel):
    """Aggregated output from multiple experts."""

    predictions: list[ExpertPrediction | PosePrediction | IntentPrediction | FatiguePrediction | EffortPrediction] = Field(
        default_factory=list
    )
    errors: list[ExpertError] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class RunExpertsRequest(BaseModel):
    """Request to run EMG experts."""

    task: str = Field(..., description="Task identifier")
    emg_input: EMGInput = Field(..., description="EMG input data")
    optional_modalities: dict[str, Any] | None = None
    preferred_experts: list[str] | None = None
    mode: str = "auto"


class RunExpertsResponse(BaseModel):
    """Response from run_emg_experts."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    task: str = ""
    mode: str = ""
    selected_experts: list[str] = Field(default_factory=list)
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def model_dump_json_serializable(self) -> dict:
        """Produce JSON-serializable dict for external consumers."""
        return {
            "task": self.task,
            "mode": self.mode,
            "selected_experts": self.selected_experts,
            "predictions": self.predictions,
            "errors": self.errors,
            "meta": self.meta,
        }
