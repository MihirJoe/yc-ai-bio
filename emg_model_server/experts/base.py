"""Base expert interface for EMG model adapters."""

from abc import ABC, abstractmethod
from typing import Any

from emg_model_server.types import ExpertPrediction


class BaseEMGExpert(ABC):
    """Abstract base class for all EMG expert model wrappers."""

    name: str = "base"

    @abstractmethod
    def supported_tasks(self) -> list[str]:
        """Return task identifiers this expert supports."""
        ...

    @abstractmethod
    def required_modalities(self) -> list[str]:
        """Return required input modalities (e.g. 'emg')."""
        ...

    def supported_input_modes(self) -> list[str]:
        """
        Return supported input modes.

        Examples: ["single_channel", "multi_channel", "benchmark_only"]
        """
        return ["single_channel", "multi_channel"]

    def is_available(self) -> bool:
        """Return True if the expert can run (model loaded, etc)."""
        return True

    def validate_input(self, payload: dict[str, Any]) -> None:
        """
        Validate input payload. Raise ValueError if invalid.
        """
        if "emg_data" not in payload:
            raise ValueError("payload must contain 'emg_data'")
        data = payload["emg_data"]
        import numpy as np
        if not isinstance(data, np.ndarray):
            raise ValueError("emg_data must be numpy.ndarray")
        if data.size == 0:
            raise ValueError("emg_data cannot be empty")

    @abstractmethod
    def predict(self, payload: dict[str, Any]) -> ExpertPrediction:
        """Run inference and return structured prediction."""
        ...
