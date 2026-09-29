"""Cephalometric visualization utilities.

Draws predicted landmarks, error vectors, anatomical labels, and standard
cephalometric planes on lateral skull radiographs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


# Standard Cephalometric anatomical planes (connecting point pairs)
STANDARD_CEPH_PLANES = [
    ("S", "N", (255, 200, 0)),     # Sella-Nasion plane (Cranial base)
    ("Po", "Or", (200, 255, 0)),   # Frankfort Horizontal (FH) plane
    ("ANS", "PNS", (0, 255, 255)), # Palatal plane
    ("Go", "Me", (0, 200, 255)),   # Mandibular plane
    ("N", "A", (255, 100, 255)),   # NA line
    ("N", "B", (255, 100, 200)),   # NB line
    ("N", "Pog", (100, 255, 100)), # Facial plane
]


def draw_landmarks_on_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    symbols: Optional[Sequence[str]] = None,
    mask: Optional[Sequence[float | int | bool]] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 5,
    draw_labels: bool = True,
    font_scale: float = 0.5,
    thickness: int = -1,
) -> np.ndarray:
    """
    Overlay landmarks and text symbols onto a grayscale or BGR image.

    Args:
        image: HxW grayscale or HxWx3 BGR image.
        landmarks: (K, 2) array of (x, y) coordinates.
        symbols: Optional list of K landmark names.
        mask: Optional (K,) array indicating valid landmarks.
        color: Circle BGR color.
        radius: Landmark circle radius.
        draw_labels: Whether to render landmark symbol text.

    Returns:
        Annotated BGR image.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    k_total = len(landmarks)
    for i in range(k_total):
        if mask is not None and not bool(mask[i]):
            continue

        x, y = landmarks[i]
        if np.isnan(x) or np.isnan(y):
            continue

        px, py = int(round(float(x))), int(round(float(y)))
        cv2.circle(vis, (px, py), radius, color, thickness)

        if draw_labels and symbols is not None and i < len(symbols):
            sym = symbols[i]
            cv2.putText(
                vis,
                sym,
                (px + radius + 3, py + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return vis


def draw_comparison_overlay(
    image: np.ndarray,
    pred_landmarks: np.ndarray,
    gt_landmarks: np.ndarray,
    symbols: Optional[Sequence[str]] = None,
    mask: Optional[Sequence[float | int | bool]] = None,
    draw_planes: bool = True,
) -> np.ndarray:
    """
    Produce a side-by-side or combined error visualization:
        - Ground truth: Green circles
        - Prediction: Red circles
        - Error vector: Yellow line connecting GT to Pred
        - Ceph Planes: Thin colored lines
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    lm_dict: Dict[str, Tuple[float, float]] = {}

    k_total = len(pred_landmarks)
    for i in range(k_total):
        if mask is not None and not bool(mask[i]):
            continue

        pred_pt = pred_landmarks[i]
        gt_pt = gt_landmarks[i]

        if not np.isnan(pred_pt[0]) and not np.isnan(pred_pt[1]):
            px_p, py_p = int(round(float(pred_pt[0]))), int(round(float(pred_pt[1])))
            # Red for prediction
            cv2.circle(vis, (px_p, py_p), 4, (0, 0, 255), -1)

            if symbols and i < len(symbols):
                lm_dict[symbols[i]] = (float(pred_pt[0]), float(pred_pt[1]))
                cv2.putText(
                    vis,
                    symbols[i],
                    (px_p + 6, py_p - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if not np.isnan(gt_pt[0]) and not np.isnan(gt_pt[1]):
            px_g, py_g = int(round(float(gt_pt[0]))), int(round(float(gt_pt[1])))
            # Green for GT
            cv2.circle(vis, (px_g, py_g), 4, (0, 255, 0), -1)

        # Draw error vector
        if not np.isnan(pred_pt).any() and not np.isnan(gt_pt).any():
            cv2.line(
                vis,
                (int(round(gt_pt[0])), int(round(gt_pt[1]))),
                (int(round(pred_pt[0])), int(round(pred_pt[1]))),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Optional: draw anatomical planes
    if draw_planes:
        for p1, p2, plane_color in STANDARD_CEPH_PLANES:
            if p1 in lm_dict and p2 in lm_dict:
                pt1 = (int(round(lm_dict[p1][0])), int(round(lm_dict[p1][1])))
                pt2 = (int(round(lm_dict[p2][0])), int(round(lm_dict[p2][1])))
                cv2.line(vis, pt1, pt2, plane_color, 1, cv2.LINE_AA)

    return vis
