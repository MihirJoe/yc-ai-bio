"""
EMG Model Server - Model-serving and normalization layer for EMG expert models.

Primary API: individual adapter functions returning typed outputs.
  - run_fatigue(emg_input) -> FatiguePrediction
  - run_effort(emg_input) -> EffortPrediction
  - run_pose(emg_input) -> PosePrediction
  - run_intent(emg_input) -> IntentPrediction

Each returns a Pydantic schema; use .model_dump() for JSON.
"""

from emg_model_server.api import (
    run_fatigue,
    run_effort,
    run_pose,
    run_intent,
    run_single_expert,
    run_emg_experts,
    list_available_experts,
    get_capability_map,
)
from emg_model_server.types import (
    EMGInput,
    FatiguePrediction,
    EffortPrediction,
    PosePrediction,
    IntentPrediction,
    RunExpertsRequest,
    RunExpertsResponse,
)

__version__ = "0.1.0"
__all__ = [
    "run_fatigue",
    "run_effort",
    "run_pose",
    "run_intent",
    "run_single_expert",
    "run_emg_experts",
    "list_available_experts",
    "get_capability_map",
    "EMGInput",
    "FatiguePrediction",
    "EffortPrediction",
    "PosePrediction",
    "IntentPrediction",
    "RunExpertsRequest",
    "RunExpertsResponse",
]
