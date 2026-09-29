"""Cephalometric evaluation metrics (MRE and SDR).

Standard definitions:
    - Radial Error (RE): Euclidean distance in millimeters between predicted and ground-truth landmark.
    - Mean Radial Error (MRE): Average RE over all valid landmarks.
    - Successful Detection Rate (SDR_z): Percentage of valid landmarks whose RE <= z mm.
      Standard thresholds: 2.0 mm, 2.5 mm, 3.0 mm, 4.0 mm.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def radial_error_mm(
    pred_orig: np.ndarray,
    gt_orig: np.ndarray,
    mask: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    """
    Compute per-landmark radial error in millimetres.

    Args:
        pred_orig: (K, 2) predicted coordinates in original image space.
        gt_orig: (K, 2) ground truth coordinates in original image space.
        mask: (K,) binary array (1 = valid, 0 = missing/ignore).
        pixel_size: Image resolution in mm/pixel.

    Returns:
        err: (K,) array of radial errors in mm; np.nan where mask == 0.
    """
    diff = pred_orig - gt_orig
    err = np.linalg.norm(diff, axis=1) * pixel_size
    err = err.astype(np.float32)
    err[mask == 0] = np.nan
    return err


def compute_sdr(
    errors_mm: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] = (2.0, 2.5, 3.0, 4.0),
) -> Dict[str, float]:
    """
    Compute Successful Detection Rate (%) for given thresholds in millimetres.
    """
    arr = np.asarray(errors_mm, dtype=np.float32)
    valid = arr[~np.isnan(arr)]
    total = len(valid)

    results = {}
    for th in thresholds:
        key = f"SDR_{th}mm"
        if total == 0:
            results[key] = 0.0
        else:
            sdr = float(np.mean(valid <= th) * 100.0)
            results[key] = round(sdr, 2)
    return results


def summarize_metrics(
    errors_mm: Sequence[float] | np.ndarray,
    sdr_thresholds: Sequence[float] = (2.0, 2.5, 3.0, 4.0),
) -> Dict[str, float]:
    """
    Compute full summary: MRE, Median, Std, and SDR at each threshold.
    """
    arr = np.asarray(errors_mm, dtype=np.float32)
    valid = arr[~np.isnan(arr)]

    if len(valid) == 0:
        summary = {
            "mre": 999.0,
            "median": 999.0,
            "std": 0.0,
            "count": 0,
        }
        for th in sdr_thresholds:
            summary[f"SDR_{th}mm"] = 0.0
        return summary

    summary = {
        "mre": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "count": int(len(valid)),
    }
    summary.update(compute_sdr(valid, sdr_thresholds))
    return summary
