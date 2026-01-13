from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def safe_torch_load(path: str | Path, *, map_location: str | torch.device = "cpu", weights_only: bool = True):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state_dict-like dict, got {type(obj)}")
    return obj


def align_module_prefix(state_dict: Dict[str, torch.Tensor], model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    model_has = any(k.startswith("module.") for k in model_state_dict)
    ckpt_has = any(k.startswith("module.") for k in state_dict)
    if model_has and not ckpt_has:
        return {f"module.{k}": v for k, v in state_dict.items()}
    if (not model_has) and ckpt_has:
        return {k[len("module.") :]: v for k, v in state_dict.items() if k.startswith("module.")}
    return state_dict


def load_model_state(model: torch.nn.Module, path: str | Path, *, strict: bool = False):
    ckpt = safe_torch_load(path, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(ckpt)
    state = align_module_prefix(state, model.state_dict())
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


def load_stream_ckpt(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor] | None]:
    """Load a per-stream checkpoint.

    Supported formats:
      1) {"detector": <state_dict>, "proj": <state_dict or None>, ...}
      2) <state_dict>  (assumed to be detector weights)
    """
    ckpt = safe_torch_load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "detector" in ckpt:
        det = ckpt["detector"]
        proj = ckpt.get("proj")
        if not isinstance(det, dict):
            raise TypeError("Stream checkpoint key 'detector' must be a state_dict dict")
        if proj is not None and not isinstance(proj, dict):
            raise TypeError("Stream checkpoint key 'proj' must be a state_dict dict or None")
        return det, proj
    state = unwrap_state_dict(ckpt)
    return state, None

