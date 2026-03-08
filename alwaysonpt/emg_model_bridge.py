"""
EMG Model Server bridge — interface to EMG expert models.

Call EMG model server adapters (fatigue, effort, pose, intent) from the agentic layer.
Requires emg-model-server installed: pip install -e /path/to/emg-model-server

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

try:
    from emg_model_server.bridge import (
        load_emg_input,
        get_fatigue,
        get_effort,
        get_pose,
        get_intent,
        get_all,
    )
    _EMG_SERVER_AVAILABLE = True
except ImportError as e:
    _EMG_SERVER_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    def load_emg_input(*, data=None, file_path=None, sample_rate=1000, channel_names=None):
        raise ImportError(
            "emg_model_server not installed. "
            "Install: pip install -e /path/to/emg-model-server. "
            f"Error: {_IMPORT_ERROR}"
        )

    def get_fatigue(*args, **kwargs):
        raise ImportError("emg_model_server not installed. pip install -e /path/to/emg-model-server")

    def get_effort(*args, **kwargs):
        raise ImportError("emg_model_server not installed. pip install -e /path/to/emg-model-server")

    def get_pose(*args, **kwargs):
        raise ImportError("emg_model_server not installed. pip install -e /path/to/emg-model-server")

    def get_intent(*args, **kwargs):
        raise ImportError("emg_model_server not installed. pip install -e /path/to/emg-model-server")

    def get_all(*args, **kwargs):
        raise ImportError("emg_model_server not installed. pip install -e /path/to/emg-model-server")


def is_emg_model_server_available() -> bool:
    """Check if emg-model-server is installed and usable."""
    return _EMG_SERVER_AVAILABLE


__all__ = [
    "load_emg_input",
    "get_fatigue",
    "get_effort",
    "get_pose",
    "get_intent",
    "get_all",
    "is_emg_model_server_available",
]
