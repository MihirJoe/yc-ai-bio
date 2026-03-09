#!/usr/bin/env python3
"""
Verify each EMG model pipeline. Run after setup: python scripts/verify_models.py
"""
import sys
from pathlib import Path

# Ensure package root is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from alwaysonpt.emg_model_bridge import (
    load_emg_input,
    get_fatigue,
    get_effort,
    get_pose,
    get_intent,
    get_all,
    is_emg_model_server_available,
)


def main() -> int:
    print("=" * 60)
    print("EMG Model Pipeline Verification")
    print("=" * 60)

    if not is_emg_model_server_available():
        print("ERROR: emg_model_server not available")
        return 1

    # Single-channel for fatigue/effort
    emg_1ch = np.random.randn(2000, 1).astype(np.float64) * 0.001
    inp_1ch = load_emg_input(data=emg_1ch, sample_rate=1000)

    # Fatigue
    f = get_fatigue(inp_1ch)
    ok_f = f.get("status") == "ok" or "fatigue_score" in f
    print(f"Fatigue: {'OK' if ok_f else 'FAIL'} - {f.get('status', '?')} {f.get('fatigue_score', '')}")

    # Effort
    e = get_effort(inp_1ch)
    ok_e = e.get("status") == "ok" or "effort_score" in e
    print(f"Effort:  {'OK' if ok_e else 'FAIL'} - {e.get('status', '?')} {e.get('effort_score', '')}")

    # Intent (8ch)
    emg_8ch = np.random.randn(2000, 8).astype(np.float64) * 0.001
    inp_8ch = load_emg_input(data=emg_8ch, sample_rate=1000)
    i = get_intent(inp_8ch)
    ok_i = i.get("status") == "ok"
    print(f"Intent:  {'OK' if ok_i else 'unavailable (need EMG_GESTURE_MODEL_DIR)'} - {i.get('status', '?')}")

    # Pose (16ch)
    emg_16ch = np.random.randn(12000, 16).astype(np.float64) * 0.001
    inp_16ch = load_emg_input(data=emg_16ch, sample_rate=2000)
    p = get_pose(inp_16ch)
    ok_p = p.get("status") == "ok"
    print(f"Pose:    {'OK' if ok_p else 'unavailable (need emg2pose env + checkpoint)'} - {p.get('status', '?')}")

    print("=" * 60)
    working = sum([ok_f, ok_e, ok_i, ok_p])
    print(f"Working: {working}/4 (fatigue, effort, intent, pose)")
    return 0 if ok_f and ok_e else 1


if __name__ == "__main__":
    sys.exit(main())
