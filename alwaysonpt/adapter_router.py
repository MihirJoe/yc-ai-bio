"""
LLM-based query router for EMG adapter selection.

Maps physician queries to the relevant EMG expert adapters
(fatigue, effort, pose, intent) and runs them in parallel
via the emg_model_bridge.
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import anthropic

from alwaysonpt.emg_model_bridge import (
    load_emg_input,
    get_fatigue,
    get_effort,
    get_pose,
    get_intent,
)

logger = logging.getLogger(__name__)

ADAPTER_NAMES = ["fatigue", "effort", "pose", "intent"]

_ROUTER_PROMPT = """\
You are a clinical query router. Given a physician's question about EMG data,
return a JSON list of which adapters to invoke.

Available adapters:
- "fatigue" — muscle fatigue estimation (MDF trend, RMS changes)
- "effort" — effort/activation level (peak events, contraction intensity)
- "pose" — joint angle estimation from 16-channel EMG (emg2pose)
- "intent" — gesture/intent classification from EMG

Rules:
- If the query is about tiredness, fatigue, endurance, sustained effort, or MDF: include "fatigue"
- If the query is about force, effort, intensity, activation, strength, contraction: include "effort"
- If the query is about pose, joint angles, movement, kinematics, range of motion: include "pose"
- If the query is about gesture, intent, grasp, hand movement, finger, action recognition: include "intent"
- If the query asks for a full summary, report, overview, or is broad: include ALL adapters
- Return ONLY a JSON array of adapter name strings. No explanation.

Examples:
- "How fatigued is the patient?" -> ["fatigue"]
- "Show me effort level and peak contractions" -> ["effort"]
- "Give me a full physio report" -> ["fatigue", "effort", "pose", "intent"]
- "What gesture is the patient performing?" -> ["intent"]
- "Joint angles and fatigue assessment" -> ["fatigue", "pose"]
"""

_ADAPTER_FNS = {
    "fatigue": get_fatigue,
    "effort": get_effort,
    "pose": get_pose,
    "intent": get_intent,
}


def route_query(query: str) -> list[str]:
    """Classify a physician query into adapter names using Claude."""
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=_ROUTER_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        text = response.content[0].text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            adapters = json.loads(text[start:end])
            return [a for a in adapters if a in ADAPTER_NAMES]
    except Exception as e:
        logger.warning("Router LLM call failed, defaulting to all: %s", e)

    return list(ADAPTER_NAMES)


def run_adapters(
    adapter_names: list[str],
    emg_data: np.ndarray,
    sample_rate: int = 1000,
    channel_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run selected adapters in parallel and collect results."""
    emg_input = load_emg_input(
        data=emg_data, sample_rate=sample_rate, channel_names=channel_names,
    )

    results: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for name in adapter_names:
            fn = _ADAPTER_FNS.get(name)
            if fn is None:
                results[name] = {"adapter": name, "status": "error", "reason": f"Unknown adapter: {name}"}
                continue
            futures[pool.submit(fn, emg_input)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"adapter": name, "status": "error", "reason": str(e)}

    return results


def run_routed_analysis(
    query: str,
    emg_data: np.ndarray,
    sample_rate: int = 1000,
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Full pipeline: route query -> run adapters -> return results."""
    adapter_names = route_query(query)
    logger.info("Routed query to adapters: %s", adapter_names)

    adapter_results = run_adapters(
        adapter_names, emg_data, sample_rate, channel_names,
    )

    return {
        "query": query,
        "routed_adapters": adapter_names,
        "adapter_results": adapter_results,
    }
