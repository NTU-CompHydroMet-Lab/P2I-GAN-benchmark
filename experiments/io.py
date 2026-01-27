from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np


def ensure_thw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {arr.shape}")
    return arr


def crop_center(arr: np.ndarray, size: int) -> np.ndarray:
    arr = ensure_thw(arr)
    t, h, w = arr.shape
    if size > min(h, w):
        raise ValueError(f"crop size {size} exceeds input {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[:, top:top + size, left:left + size]


def load_mask(path: str) -> np.ndarray:
    mask = np.loadtxt(path)
    return mask.astype(bool)


def _open_zarr_xarray(path: str, key: Optional[str] = None) -> np.ndarray:
    import xarray as xr

    ds = xr.open_zarr(path)
    if hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
        if key is None:
            key = list(ds.data_vars.keys())[0]
        arr = ds[key].values
    else:
        arr = ds.values
    return np.asarray(arr)


def _open_zarr_native(path: str) -> np.ndarray:
    import zarr

    z = zarr.open(path, mode="r")
    if hasattr(z, "array_keys"):
        keys = list(z.array_keys())
        if keys:
            return np.asarray(z[keys[0]])
    return np.asarray(z)


def _load_zarr_events_xarray(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        import xarray as xr
    except Exception:
        return None

    ds = xr.open_zarr(path)
    if hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
        return {k: np.asarray(ds[k].values) for k in ds.data_vars.keys()}
    return None


def _load_zarr_events_native(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        import zarr
    except Exception:
        return None

    z = zarr.open(path, mode="r")
    if hasattr(z, "group_keys"):
        keys = list(z.group_keys())
        if keys:
            return {k: np.asarray(z[k]) for k in keys}
    if hasattr(z, "array_keys"):
        keys = list(z.array_keys())
        if keys:
            return {k: np.asarray(z[k]) for k in keys}
    return None


def load_zarr_array(path: str,
                    key: Optional[str] = None,
                    return_events: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if return_events:
        events = _load_zarr_events_xarray(path)
        if events is None:
            events = _load_zarr_events_native(path)
        if events:
            return events
    try:
        arr = _open_zarr_xarray(path, key=key)
    except Exception:
        arr = _open_zarr_native(path)
    return np.asarray(arr)


def align_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = ensure_thw(a)
    b = ensure_thw(b)
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def select_by_mask(arr: np.ndarray, mask: np.ndarray, invert: bool = False) -> np.ndarray:
    arr = ensure_thw(arr)
    mask = mask.astype(bool)
    if mask.shape != arr.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} != data shape {arr.shape[1:]}")
    if invert:
        mask = ~mask
    flat = arr.reshape(arr.shape[0], -1)
    return flat[:, mask.ravel()]


def mask_for_input(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = ensure_thw(arr)
    mask = mask.astype(bool)
    if mask.shape != arr.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} != data shape {arr.shape[1:]}")
    out = arr.copy()
    out[:, mask] = 0.0
    return out


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config_snapshot(path: str, cfg) -> None:
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = cfg.__dict__
    else:
        payload = cfg
    save_json(path, payload)


def save_text(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
