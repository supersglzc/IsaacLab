"""
Common utilities — Tracker, set_random_seed, handle_timeout, load_class_from_path,
list_class_names. Adapted from PQL/DDiffPG with no extra dependencies.
"""
from __future__ import annotations

import ast
import importlib.util
import random
import sys
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int | None = None) -> int:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


def load_class_from_path(cls_name: str, path):
    """Load a class by name from an arbitrary .py file path."""
    mod_name = f"MOD{cls_name}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def list_class_names(dir_path) -> dict:
    """Return mapping {class_name: file_path} for every class defined in any .py
    file under dir_path (excluding __init__.py)."""
    dir_path = Path(dir_path)
    out: dict[str, Path] = {}
    for py_file in dir_path.rglob("*.py"):
        if not py_file.is_file() or py_file.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out[node.name] = py_file
    return out


class Tracker:
    """Fixed-window moving statistic over recent episode metrics.

    The deque starts empty — `mean()` returns 0.0 until the first `update()`,
    after which it averages over actual observations only. (The previous
    implementation pre-filled with zeros, which biased early-training metrics
    downward by a factor of `n_observed / max_len`.)
    """

    def __init__(self, max_len: int):
        self.moving_average: deque = deque(maxlen=max_len)
        self.max_len = max_len

    def update(self, value):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self) -> float:
        if not self.moving_average:
            return 0.0
        return float(np.mean(self.moving_average))

    def std(self) -> float:
        if not self.moving_average:
            return 0.0
        return float(np.std(self.moving_average))


def handle_timeout(dones, info):
    """Mask `dones` so that envs which died because of TimeLimit don't bootstrap as terminal.

    Looks for either `TimeLimit.truncated` (gym) or `time_outs` (IsaacGymEnvs) keys.
    """
    timeout = info.get("TimeLimit.truncated") if isinstance(info, dict) else None
    if timeout is None and isinstance(info, dict):
        timeout = info.get("time_outs")
    if timeout is None:
        return dones
    if isinstance(timeout, torch.Tensor) and isinstance(dones, torch.Tensor):
        return dones * (~timeout.bool())
    return dones


def aggregate_traj_info(infos: list[dict], key: str):
    out = []
    for info in infos:
        if isinstance(info, dict) and key in info:
            out.append(info[key])
    if not out:
        return None
    if isinstance(out[0], torch.Tensor):
        return torch.stack(out, dim=0)
    return np.stack(out, axis=0)
