"""Checkpoint save / load."""
from __future__ import annotations

from pathlib import Path

import torch


def save_model(path, *, actor, critic, rms=None, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"actor": actor, "critic": critic}
    if rms is not None:
        payload["rms"] = rms
    payload.update(extra)
    torch.save(payload, str(path))
    return path


def load_model(target_module, key: str, ckpt_path) -> None:
    """Load `key` from a checkpoint file into `target_module` (in-place)."""
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if key not in state:
        raise KeyError(f"Checkpoint at {ckpt_path} has no key '{key}' (keys: {list(state)}).")
    if hasattr(target_module, "load_state_dict"):
        target_module.load_state_dict(state[key])
    else:
        # RunningMeanStd-style — exposes load_state_dict that takes a tuple.
        target_module.load_state_dict(state[key])
