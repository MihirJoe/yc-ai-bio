"""Placeholder intent adapter - NOT IMPLEMENTED.

Kept for interface compatibility. Returns status=unavailable.
Use reactemg_adapter for baseline intent (threshold-based) or real ReactEMG when integrated.
"""

import time
from typing import Any

from emg_model_server.experts.base import BaseEMGExpert
from emg_model_server.types import IntentPrediction


class MockIntentAdapter(BaseEMGExpert):
    """Placeholder. Returns unavailable - use reactemg_adapter for baseline intent."""

    name = "mock_intent_adapter"

    def supported_tasks(self) -> list[str]:
        return ["estimate_intent", "generate_physio_report", "generate_physio_signals"]

    def required_modalities(self) -> list[str]:
        return ["emg"]

    def supported_input_modes(self) -> list[str]:
        return ["single_channel", "multi_channel"]

    def is_available(self) -> bool:
        return False

    def predict(self, payload: dict[str, Any]) -> IntentPrediction:
        return IntentPrediction(
            expert_name=self.name,
            task_type="intent",
            status="unavailable",
            implementation_status="unimplemented",
            reason=(
                "mock_intent_adapter is a placeholder. Use reactemg_adapter for baseline "
                "intent (threshold-based) or real ReactEMG when integrated."
            ),
            metadata={"blocker": "placeholder_adapter"},
        )
