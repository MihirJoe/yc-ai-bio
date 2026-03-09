"""
emg2pose api.py - Inference wrapper for emg2pose model.

Copied to emg2pose/api.py by setup.sh. Provides emg2pose_inference() for use by
emg_model_server pose adapter. Meta's emg2pose repo does not include this file.

Usage:
    from api import emg2pose_inference
    joint_angles = emg2pose_inference(emg, checkpoint_path="path/to/tracking_vemg2pose.ckpt")
"""
from __future__ import annotations

import numpy as np
import torch

from emg2pose.constants import NUM_JOINTS
from emg2pose.lightning import Emg2PoseModule


_module_cache: Emg2PoseModule | None = None
_module_checkpoint: str | None = None


def emg2pose_inference(
    emg: np.ndarray,
    checkpoint_path: str,
) -> np.ndarray:
    """
    Run emg2pose inference on EMG data.

    Args:
        emg: EMG array (16, T) or (B, 16, T), float32, 2000 Hz, T >= 11790.
        checkpoint_path: Path to tracking_vemg2pose.ckpt.

    Returns:
        joint_angles: (B, 20, T_out) numpy array, radians.
    """
    global _module_cache, _module_checkpoint

    # Lazy load and cache model
    if _module_cache is None or _module_checkpoint != checkpoint_path:
        _module_cache = Emg2PoseModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        _module_cache.eval()
        _module_checkpoint = checkpoint_path

    model = _module_cache

    # Ensure (B, 16, T)
    if emg.ndim == 2:
        emg = emg[np.newaxis, ...]  # (1, 16, T)
    emg_t = torch.from_numpy(emg.astype(np.float32))

    # Build dummy batch for BasePoseModule.forward (needs joint_angles, no_ik_failure)
    B, C, T = emg_t.shape
    joint_angles = torch.zeros(B, NUM_JOINTS, T, dtype=torch.float32)
    no_ik_failure = torch.ones(B, T, dtype=torch.bool)
    batch = {"emg": emg_t, "joint_angles": joint_angles, "no_ik_failure": no_ik_failure}

    # Use zeros for initial pos when no ground truth available
    orig_provide = model.provide_initial_pos
    model.provide_initial_pos = False
    try:
        with torch.no_grad():
            pred, _, _ = model.forward(batch)
    finally:
        model.provide_initial_pos = orig_provide

    return pred.numpy()
