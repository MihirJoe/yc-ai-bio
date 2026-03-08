"""
Unified API for the EMG Model Server.

Provides individual adapter functions as the primary API. Each returns a typed
prediction schema. The agentic layer calls these per adapter as needed.
"""

import logging
from typing import Any

from emg_model_server.service import (
    run_task,
    run_single_expert as _run_single_expert,
    run_experts,
    auto_select_experts,
)
from emg_model_server.registry import list_experts, get_expert
from emg_model_server.config import load_config
from emg_model_server.types import (
    EMGInput,
    RunExpertsResponse,
    ExpertPrediction,
    FatiguePrediction,
    EffortPrediction,
    PosePrediction,
    IntentPrediction,
)
from emg_model_server.experts import register_default_experts

logger = logging.getLogger(__name__)

# Ensure default experts are registered on first import
_experts_initialized = False


def _ensure_experts() -> None:
    global _experts_initialized
    if not _experts_initialized:
        register_default_experts()
        _experts_initialized = True


# -----------------------------------------------------------------------------
# Individual adapter APIs (primary interface for agentic layer)
# -----------------------------------------------------------------------------


def run_fatigue(
    emg_input: EMGInput,
    mode: str = "auto",
    config_path: str | None = None,
) -> FatiguePrediction:
    """
    Run fatigue estimation on EMG. Returns FatiguePrediction with
    fatigue_score, fatigue_trend, fatigue_segments, evidence.
    """
    _ensure_experts()
    pred = _run_single_expert("fatigue_adapter", emg_input, mode=mode, config_path=config_path)
    if pred is None or not isinstance(pred, FatiguePrediction):
        return FatiguePrediction(
            expert_name="fatigue_adapter",
            task_type="fatigue",
            status="error",
            implementation_status="baseline",
            reason="Fatigue adapter failed or not found",
        )
    return pred


def run_effort(
    emg_input: EMGInput,
    mode: str = "auto",
    config_path: str | None = None,
) -> EffortPrediction:
    """
    Run effort/activation estimation on EMG. Returns EffortPrediction with
    effort_score, peak_events, contraction_segments, activation_trace_summary.
    """
    _ensure_experts()
    pred = _run_single_expert("effort_adapter", emg_input, mode=mode, config_path=config_path)
    if pred is None or not isinstance(pred, EffortPrediction):
        return EffortPrediction(
            expert_name="effort_adapter",
            task_type="effort",
            status="error",
            implementation_status="baseline",
            reason="Effort adapter failed or not found",
        )
    return pred


def run_pose(
    emg_input: EMGInput,
    mode: str = "auto",
    config_path: str | None = None,
) -> PosePrediction:
    """
    Run pose/kinematics estimation (emg2pose). Returns PosePrediction with
    pose_features (20 joint angles), temporal_trajectory_summary.
    Requires 16-channel EMG, >=11790 samples.
    """
    _ensure_experts()
    pred = _run_single_expert("emg2pose_adapter", emg_input, mode=mode, config_path=config_path)
    if pred is None or not isinstance(pred, PosePrediction):
        return PosePrediction(
            expert_name="emg2pose_adapter",
            task_type="pose",
            status="error",
            implementation_status="unimplemented",
            reason="Pose adapter failed or not found",
        )
    return pred


def run_intent(
    emg_input: EMGInput,
    mode: str = "auto",
    config_path: str | None = None,
) -> IntentPrediction:
    """
    Run intent/gesture estimation (emg_gesture). Returns IntentPrediction with
    intent_labels, intent_labels_over_time, onset_timestamps, stability_score.
    Uses first 8 channels when input has >=8 channels.
    """
    _ensure_experts()
    pred = _run_single_expert("emg_gesture_adapter", emg_input, mode=mode, config_path=config_path)
    if pred is None or not isinstance(pred, IntentPrediction):
        return IntentPrediction(
            expert_name="emg_gesture_adapter",
            task_type="intent",
            status="error",
            implementation_status="unimplemented",
            reason="Intent adapter failed or not found",
        )
    return pred


# -----------------------------------------------------------------------------
# Generic / batch APIs (optional)
# -----------------------------------------------------------------------------


def run_emg_experts(
    task: str,
    emg_input: EMGInput,
    optional_modalities: dict[str, Any] | None = None,
    preferred_experts: list[str] | None = None,
    mode: str = "auto",
    config_path: str | None = None,
) -> RunExpertsResponse:
    """
    Main API: run EMG experts for a task and return structured outputs.

    Args:
        task: Task identifier (e.g. "generate_physio_report", "estimate_fatigue")
        emg_input: EMGInput with data or file_path
        optional_modalities: Optional additional inputs
        preferred_experts: Override auto-selected experts
        mode: "auto" | "benchmark" | "live_lite"
        config_path: Optional path to config YAML

    Returns:
        RunExpertsResponse with predictions and errors
    """
    _ensure_experts()
    return run_task(
        task=task,
        emg_input=emg_input,
        optional_modalities=optional_modalities,
        preferred_experts=preferred_experts,
        mode=mode,
        config_path=config_path,
    )


def run_single_expert(
    name: str,
    emg_input: EMGInput,
    mode: str = "auto",
    optional_modalities: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> ExpertPrediction | None:
    """
    Run a single expert by name.
    """
    _ensure_experts()
    return _run_single_expert(
        name=name,
        emg_input=emg_input,
        mode=mode,
        optional_modalities=optional_modalities,
        config_path=config_path,
    )


def list_available_experts() -> list[str]:
    """List names of all registered experts."""
    _ensure_experts()
    return list_experts()


def get_capability_map() -> dict[str, Any]:
    """
    Return a map of expert capabilities for discovery.

    Returns:
        Dict with expert names -> tasks, modalities, input modes, implementation_status
    """
    _ensure_experts()
    result: dict[str, Any] = {}
    for name in list_experts():
        expert = get_expert(name)
        if expert:
            base = {
                "supported_tasks": expert.supported_tasks(),
                "required_modalities": expert.required_modalities(),
                "supported_input_modes": expert.supported_input_modes(),
                "is_available": expert.is_available(),
            }
            if hasattr(expert, "get_capability_info"):
                base.update(expert.get_capability_info())
            result[name] = base
    return result
