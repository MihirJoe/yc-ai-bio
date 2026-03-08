"""EMG expert model adapters."""

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.experts.fatigue_adapter import FatigueAdapter
from emg_model_server.experts.effort_adapter import EffortAdapter
from emg_model_server.experts.emg2pose_adapter import EMG2PoseAdapter
from emg_model_server.experts.reactemg_adapter import ReactEMGAdapter
from emg_model_server.experts.emg_gesture_adapter import EMGGestureAdapter
from emg_model_server.experts.mock_pose_adapter import MockPoseAdapter
from emg_model_server.experts.mock_intent_adapter import MockIntentAdapter

from emg_model_server.registry import register_expert


def register_default_experts() -> None:
    """Register all built-in experts with the registry."""
    register_expert(FatigueAdapter())
    register_expert(EffortAdapter())
    register_expert(EMG2PoseAdapter())
    register_expert(ReactEMGAdapter())
    register_expert(EMGGestureAdapter())
    register_expert(MockPoseAdapter())
    register_expert(MockIntentAdapter())


__all__ = [
    "BaseEMGExpert",
    "FatigueAdapter",
    "EffortAdapter",
    "EMG2PoseAdapter",
    "ReactEMGAdapter",
    "EMGGestureAdapter",
    "MockPoseAdapter",
    "MockIntentAdapter",
    "register_default_experts",
]
