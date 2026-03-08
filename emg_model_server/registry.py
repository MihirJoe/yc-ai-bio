"""Expert registry for discovery and lookup."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emg_model_server.experts.base import BaseEMGExpert

logger = logging.getLogger(__name__)

_registry: dict[str, "BaseEMGExpert"] = {}


def register_expert(expert: "BaseEMGExpert") -> None:
    """
    Register an expert by name.

    Args:
        expert: Expert instance implementing BaseEMGExpert
    """
    name = expert.name
    if name in _registry:
        logger.warning("Overwriting existing expert: %s", name)
    _registry[name] = expert
    logger.debug("Registered expert: %s", name)


def unregister_expert(name: str) -> None:
    """Remove an expert from the registry."""
    if name in _registry:
        del _registry[name]
        logger.debug("Unregistered expert: %s", name)


def list_experts() -> list[str]:
    """List names of all registered experts."""
    return list(_registry.keys())


def get_expert(name: str) -> "BaseEMGExpert | None":
    """Get expert by name."""
    return _registry.get(name)


def get_experts_for_task(task: str, mapping: dict[str, list[str]]) -> list[str]:
    """
    Get expert names for a given task using the mapping.

    Args:
        task: Task identifier (e.g. 'estimate_fatigue')
        mapping: task -> list of expert names

    Returns:
        List of expert names; empty if task unknown; all experts if 'full_benchmark_bundle'
    """
    if task in mapping:
        experts = mapping[task]
        if not experts and task == "full_benchmark_bundle":
            return list(_registry.keys())
        return [e for e in experts if e in _registry]
    return []


def clear_registry() -> None:
    """Clear all registered experts (mainly for tests)."""
    _registry.clear()
