"""Service layer: run experts, aggregate results, task selection."""

import logging
from typing import Any

import numpy as np

from emg_model_server.config import load_config, get_mode_from_env
from emg_model_server.io.loader import load_emg
from emg_model_server.preprocessing.pipeline import preprocess_emg
from emg_model_server.registry import get_expert, get_experts_for_task, list_experts
from emg_model_server.types import (
    EMGInput,
    ExpertPrediction,
    ExpertError,
    RunExpertsResponse,
)

logger = logging.getLogger(__name__)


def _resolve_mode(mode: str, config_path: str | None = None) -> str:
    """Resolve mode: auto -> use config/env, else use given mode."""
    if mode == "auto":
        cfg = load_config(config_path)
        return cfg.mode if cfg.mode != "auto" else get_mode_from_env()
    return mode


def _prepare_emg(emg_input: EMGInput, config_path: str | None = None) -> tuple[np.ndarray, int]:
    """Load and preprocess EMG from EMGInput."""
    cfg = load_config(config_path)
    prep = cfg.preprocessing

    if emg_input.data is not None:
        data = np.asarray(emg_input.data, dtype=np.float64)
        sr = emg_input.sample_rate
    elif emg_input.file_path:
        data, sr = load_emg(emg_input.file_path, emg_input.sample_rate)
    else:
        raise ValueError("EMGInput must have data or file_path")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    data = preprocess_emg(
        data,
        sr,
        target_sample_rate=prep.target_sample_rate,
        bandpass_low=prep.bandpass_low,
        bandpass_high=prep.bandpass_high,
        notch_freq=prep.notch_freq,
        normalize=prep.normalize,
    )
    return data, prep.target_sample_rate


def _prediction_to_dict(p: ExpertPrediction) -> dict[str, Any]:
    """Convert ExpertPrediction to JSON-serializable dict."""
    return p.model_dump(mode="json")


def run_single_expert(
    name: str,
    emg_input: EMGInput,
    mode: str = "auto",
    optional_modalities: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> ExpertPrediction | None:
    """
    Run a single expert by name.

    Returns None if expert not found or input validation fails.
    """
    expert = get_expert(name)
    if expert is None:
        logger.warning("Expert not found: %s", name)
        return None

    mode = _resolve_mode(mode, config_path)
    try:
        data, sr = _prepare_emg(emg_input, config_path)
    except Exception as e:
        logger.exception("Failed to prepare EMG: %s", e)
        return None

    payload: dict[str, Any] = {
        "emg_data": data,
        "sample_rate": sr,
        "mode": mode,
        "channel": 0,
    }
    if optional_modalities:
        payload.update(optional_modalities)

    try:
        return expert.predict(payload)
    except Exception as e:
        logger.exception("Expert %s failed: %s", name, e)
        return None


def run_experts(
    expert_names: list[str],
    emg_input: EMGInput,
    mode: str = "auto",
    optional_modalities: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> RunExpertsResponse:
    """
    Run multiple experts and aggregate results.
    Partial outputs on expert failure; does not crash.
    """
    mode = _resolve_mode(mode, config_path)
    try:
        data, sr = _prepare_emg(emg_input, config_path)
    except Exception as e:
        logger.exception("Failed to prepare EMG: %s", e)
        return RunExpertsResponse(
            task="",
            mode=mode,
            selected_experts=expert_names,
            predictions=[],
            errors=[{"expert_name": "pipeline", "message": str(e), "details": {}}],
            meta={"input_channels": 0, "sample_rate": 0},
        )

    is_single_channel = data.ndim == 1 or data.shape[1] == 1
    cfg = load_config(config_path)
    single_ok = set(cfg.single_channel_compatible or [])
    benchmark_only = set(cfg.benchmark_only or [])

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    selected: list[str] = []

    payload: dict[str, Any] = {
        "emg_data": data,
        "sample_rate": sr,
        "mode": mode,
        "channel": 0,
    }
    if optional_modalities:
        payload.update(optional_modalities)

    for name in expert_names:
        expert = get_expert(name)
        if expert is None:
            errors.append({
                "expert_name": name,
                "message": "Expert not found",
                "details": {},
            })
            continue

        try:
            pred = expert.predict(payload)
            predictions.append(_prediction_to_dict(pred))
            selected.append(name)
        except Exception as e:
            logger.exception("Expert %s failed: %s", name, e)
            errors.append({
                "expert_name": name,
                "message": str(e),
                "details": {},
            })

    return RunExpertsResponse(
        task="",
        mode=mode,
        selected_experts=selected,
        predictions=predictions,
        errors=errors,
        meta={
            "input_channels": int(data.shape[1]) if data.ndim > 1 else 1,
            "sample_rate": sr,
        },
    )


def auto_select_experts(
    task: str,
    emg_input: EMGInput,
    preferred_experts: list[str] | None = None,
    mode: str = "auto",
    config_path: str | None = None,
) -> list[str]:
    """
    Select experts for a task based on mapping and mode.
    If preferred_experts is given, use it as override (may include non-mapped experts).
    """
    cfg = load_config(config_path)
    mapping = cfg.task_expert_mapping
    experts = get_experts_for_task(task, mapping)
    if preferred_experts:
        return list(preferred_experts)
    return experts


def run_task(
    task: str,
    emg_input: EMGInput,
    optional_modalities: dict[str, Any] | None = None,
    preferred_experts: list[str] | None = None,
    mode: str = "auto",
    config_path: str | None = None,
) -> RunExpertsResponse:
    """
    Run experts for a task: auto-select, then run.
    """
    experts = auto_select_experts(task, emg_input, preferred_experts, mode, config_path)
    if not experts:
        return RunExpertsResponse(
            task=task,
            mode=_resolve_mode(mode, config_path),
            selected_experts=[],
            predictions=[],
            errors=[{"message": f"No experts found for task: {task}", "details": {}}],
            meta={},
        )
    resp = run_experts(experts, emg_input, mode, optional_modalities, config_path)
    resp.task = task
    return resp
