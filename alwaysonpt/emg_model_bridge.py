"""
EMG Model Server bridge — interface to EMG expert models.

Call EMG model server adapters (fatigue, effort, pose, intent) from the agentic layer.
emg_model_server is vendored in this repo; pip install -e . to use.

Model paths are auto-detected when not set: gesture model (submodule), emg2pose (submodule + models/).
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

import os
from pathlib import Path
from typing import Any


def _bootstrap_model_paths() -> None:
    """Set default model paths when env vars not set (self-contained repo)."""
    try:
        import emg_model_server
        pkg_dir = Path(emg_model_server.__file__).resolve().parent
        repo_root = pkg_dir.parent  # emg_model_server/ -> repo root
    except Exception:
        return

    # Gesture model (submodule)
    if not os.environ.get("EMG_GESTURE_MODEL_DIR"):
        gesture_dir = repo_root / "EMG-Gesture-Recognition-System" / "models-example" / "gesture_cls" / "1.0.0_20250821T094534Z"
        if (gesture_dir / "pipeline.joblib").exists():
            os.environ["EMG_GESTURE_MODEL_DIR"] = str(gesture_dir)

    # emg2pose project root (submodule)
    if not os.environ.get("EMG2POSE_PROJECT_ROOT"):
        emg2pose_root = repo_root / "emg2pose"
        if emg2pose_root.exists() and ((emg2pose_root / "api.py").exists() or (emg2pose_root / "emg2pose").is_dir()):
            os.environ["EMG2POSE_PROJECT_ROOT"] = str(emg2pose_root)

    # emg2pose checkpoint (models/ after download_models.sh)
    if not os.environ.get("EMG2POSE_CHECKPOINT_PATH"):
        ckpt = repo_root / "models" / "emg2pose_model_checkpoints" / "tracking_vemg2pose.ckpt"
        if ckpt.exists():
            os.environ["EMG2POSE_CHECKPOINT_PATH"] = str(ckpt)


_bootstrap_model_paths()

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
