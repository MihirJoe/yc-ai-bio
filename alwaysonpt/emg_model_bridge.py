"""
EMG Model Server bridge — interface to EMG expert models.

Call EMG model server adapters (fatigue, effort, pose, intent) from the agentic layer.
emg_model_server is vendored in this repo; pip install -e . to use.

Usage:
    from alwaysonpt.emg_model_bridge import (
        load_emg_input,
        get_fatigue,
        get_effort,
        get_pose,
        get_intent,
        get_all,
    )

    emg_input = load_emg_input(data=emg_array, sample_rate=1000)
    fatigue_out = get_fatigue(emg_input)   # dict
    effort_out = get_effort(emg_input)     # dict
    pose_out = get_pose(emg_input)         # dict (16-channel)
    intent_out = get_intent(emg_input)     # dict (8+ channel)
    all_out = get_all(emg_input)           # {"fatigue": {...}, "effort": {...}, ...}
"""

from typing import Any

from emg_model_server.bridge import (
    load_emg_input,
    get_fatigue,
    get_effort,
    get_pose,
    get_intent,
    get_all,
)


def is_emg_model_server_available() -> bool:
    """Always True — emg_model_server is vendored in this repo."""
    return True


__all__ = [
    "load_emg_input",
    "get_fatigue",
    "get_effort",
    "get_pose",
    "get_intent",
    "get_all",
    "is_emg_model_server_available",
]
