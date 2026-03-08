"""Placeholder pose adapter - NOT IMPLEMENTED.

Kept for interface compatibility. Returns status=unavailable.
Use emg2pose_adapter for real pose inference when 16-channel data is available.
"""

import time
from typing import Any

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.types import PosePrediction


class MockPoseAdapter(BaseEMGExpert):
    """Placeholder. Returns unavailable - use emg2pose_adapter for real pose."""

    name = "mock_pose_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_pose", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["single_channel", "multi_channel"]

    def is_available(self) -> bool:
        return False

    def predict(self, payload: dict[str, Any]) -> PosePrediction:
        return PosePrediction(
            expert_name=self.name,
            task_type="pose",
            status="unavailable",
            implementation_status="unimplemented",
            reason=(
                "mock_pose_adapter is a placeholder. Use emg2pose_adapter for real pose "
                "inference with 16-channel EMG."
            ),
            metadata={"blocker": "placeholder_adapter"},
        )
