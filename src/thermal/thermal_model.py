#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/thermal/thermal_model.py

Core thermal model used by all scripts:
- macros (name, x_um, y_um, power_W) -> power grid
- Gaussian diffusion -> temperature field
- gradient -> grad-norm field
- metrics (mean / p90 / max)
"""

from typing import List, Tuple, Dict, Any

import numpy as np
from scipy import ndimage


Macro = Tuple[str, float, float, float]  # (name, x_um, y_um, power_W)


def build_power_grid(
    macros: List[Macro],
    chip_w_um: float,
    chip_h_um: float,
    nx: int = 128,
    ny: int = 128,
):
    """
    Build a discrete power grid from a list of macros.

    Args:
        macros: list of (name, x_um, y_um, power_W)
        chip_w_um: chip width in microns
        chip_h_um: chip height in microns
        nx, ny: grid resolution

    Returns:
        power_grid: (ny, nx) array of power values
        meta: dict with grid / geometry info
    """
    power_grid = np.zeros((ny, nx), dtype=np.float32)

    for name, x_um, y_um, p_w in macros:
        # map physical coordinates to grid indices
        ix = int(x_um / chip_w_um * nx)
        iy = int(y_um / chip_h_um * ny)
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        power_grid[iy, ix] += p_w

    meta = {
        "chip_w_um": float(chip_w_um),
        "chip_h_um": float(chip_h_um),
        "nx": int(nx),
        "ny": int(ny),
        "dx_um": float(chip_w_um) / nx,
        "dy_um": float(chip_h_um) / ny,
    }
    return power_grid, meta


def power_to_temperature(
    power_grid: np.ndarray,
    meta: Dict[str, Any],
    sigma_um: float = 80.0,
) -> np.ndarray:
    """
    Approximate steady-state temperature field via Gaussian diffusion.

    Args:
        power_grid: (ny, nx) power grid
        meta: dict from build_power_grid
        sigma_um: Gaussian kernel std in microns

    Returns:
        temp_grid: (ny, nx) temperature field (relative units)
    """
    nx, ny = meta["nx"], meta["ny"]
    chip_w_um = meta["chip_w_um"]
    chip_h_um = meta["chip_h_um"]

    # convert physical sigma to grid units
    sigma_cells_x = sigma_um / chip_w_um * nx
    sigma_cells_y = sigma_um / chip_h_um * ny

    temp_grid = ndimage.gaussian_filter(
        power_grid,
        sigma=(sigma_cells_y, sigma_cells_x),
        mode="nearest",
    )
    return temp_grid


def temperature_gradient(
    temp_grid: np.ndarray,
    meta: Dict[str, Any],
):
    """
    Compute temperature gradient and its norm.

    Returns:
        dTx, dTy, grad_norm: arrays of shape (ny, nx)
    """
    dx_um = meta["dx_um"]
    dy_um = meta["dy_um"]

    # np.gradient order: (d/dy, d/dx)
    dTy, dTx = np.gradient(temp_grid, dy_um, dx_um)
    grad_norm = np.sqrt(dTx ** 2 + dTy ** 2)
    return dTx, dTy, grad_norm


def _stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def thermal_metrics(
    temp_grid: np.ndarray,
    grad_norm: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute scalar metrics from temperature / grad-norm fields.
    """
    return {
        "temperature": _stats(temp_grid),
        "grad_norm": _stats(grad_norm),
    }


def compute_thermal_from_macros(
    macros: List[Macro],
    chip_w_um: float,
    chip_h_um: float,
    nx: int = 128,
    ny: int = 128,
    sigma_um: float = 80.0,
) -> Dict[str, Any]:
    """
    One-shot convenience API:
    macros -> power grid -> temp -> grad -> metrics.

    Returns:
        {
          "power_grid": ...,
          "temp_grid": ...,
          "dTx": ...,
          "dTy": ...,
          "grad_norm": ...,
          "meta": meta,
          "metrics": metrics_dict,
        }
    """
    power_grid, meta = build_power_grid(macros, chip_w_um, chip_h_um, nx, ny)
    temp_grid = power_to_temperature(power_grid, meta, sigma_um)
    dTx, dTy, grad_norm = temperature_gradient(temp_grid, meta)
    metrics = thermal_metrics(temp_grid, grad_norm)

    return {
        "power_grid": power_grid,
        "temp_grid": temp_grid,
        "dTx": dTx,
        "dTy": dTy,
        "grad_norm": grad_norm,
        "meta": meta,
        "metrics": metrics,
    }
