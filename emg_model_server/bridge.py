"""
Bridging interface between EMG Model Server and the agentic layer.

Use this module from your agentic project:

    from emg_model_server.bridge import (
        get_fatigue,
        get_effort,
        get_pose,
        get_intent,
        load_emg_input,
    )

    emg_input = load_emg_input(data=emg_array, sample_rate=1000)
    fatigue_out = get_fatigue(emg_input)
    effort_out = get_effort(emg_input)
"""

from typing import Any

import numpy as np

from emg_model_server.types import EMGInput
from emg_model_server.api import run_fatigue, run_effort, run_pose, run_intent


def load_emg_input(
    *,
    data: np.ndarray | None = None,
    file_path: str | None = None,
    sample_rate: int = 1000,
    channel_names: list[str] | None = None,
) -> EMGInput:
    """
    Load EMG input for the bridge functions.

    Args:
        data: Raw EMG array (samples,) or (samples, channels)
        file_path: Path to CSV, NPY, or NPZ file (alternative to data)
        sample_rate: Samples per second
        channel_names: Optional channel names

    Returns:
        EMGInput for use with get_fatigue, get_effort, get_pose, get_intent
    """
    if data is not None:
        return EMGInput(data=np.asarray(data, dtype=np.float64), sample_rate=sample_rate, channel_names=channel_names)
    if file_path:
        return EMGInput(file_path=file_path, sample_rate=sample_rate, channel_names=channel_names)
    raise ValueError("Provide data or file_path")


def get_fatigue(emg_input: EMGInput, mode: str = "auto") -> dict[str, Any]:
    """
    Run fatigue estimation. Returns format_output() dict for agent.

    Works with single- or multi-channel EMG.
    """
    pred = run_fatigue(emg_input, mode=mode)
    return pred.format_output()


def get_effort(emg_input: EMGInput, mode: str = "auto") -> dict[str, Any]:
    """
    Run effort/activation estimation. Returns format_output() dict for agent.

    Works with single- or multi-channel EMG.
    """
    pred = run_effort(emg_input, mode=mode)
    return pred.format_output()


def get_pose(emg_input: EMGInput, mode: str = "benchmark") -> dict[str, Any]:
    """
    Run pose/kinematics (emg2pose). Returns format_output() dict for agent.

    Requires 16-channel EMG, >=11790 samples. Returns unavailable if input incompatible.
    """
    pred = run_pose(emg_input, mode=mode)
    return pred.format_output()


def get_intent(emg_input: EMGInput, mode: str = "benchmark") -> dict[str, Any]:
    """
    Run intent/gesture (emg_gesture). Returns format_output() dict for agent.

    Requires >=8 channels; uses first 8 when input has more.
    """
    pred = run_intent(emg_input, mode=mode)
    return pred.format_output()


def get_all(
    emg_input: EMGInput,
    mode: str = "auto",
    *,
    include_pose: bool = True,
    include_intent: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Run all compatible adapters and return a dict of {adapter_name: format_output}.

    Pose and intent return status=unavailable if input incompatible.
    """
    result: dict[str, dict[str, Any]] = {}
    result["fatigue"] = get_fatigue(emg_input, mode=mode)
    result["effort"] = get_effort(emg_input, mode=mode)
    m = "benchmark" if mode == "auto" else mode
    if include_pose:
        result["pose"] = get_pose(emg_input, mode=m)
    if include_intent:
        result["intent"] = get_intent(emg_input, mode=m)
    return result
