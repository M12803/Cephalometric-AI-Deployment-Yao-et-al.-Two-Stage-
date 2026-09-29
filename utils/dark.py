"""Distribution-Aware coordinate Representation of Keypoint (DARK) decoding.

Reference:
    Zhang et al., "Distribution-Aware Coordinate Representation for Human Pose Estimation",
    CVPR 2020. https://arxiv.org/abs/1910.06278

Decodes sub-pixel landmark coordinates from predicted Gaussian heatmaps via:
    1. 2D Gaussian smoothing to filter noise.
    2. Modulation in the log domain: G = ln(H).
    3. Taylor expansion around the discrete maximum using 1st & 2nd derivatives (Hessian).
    4. Sub-pixel shift: Delta = - H^(-1) * g.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def _gaussian_blur(hm: np.ndarray, kernel_size: int = 11, sigma: Optional[float] = None) -> np.ndarray:
    """Apply 2D Gaussian blur to a single 2D heatmap or batch of 2D heatmaps."""
    if kernel_size <= 1:
        return hm
    if kernel_size % 2 == 0:
        kernel_size += 1

    if sigma is None:
        sigma = (kernel_size - 1) / 6.0

    return cv2.GaussianBlur(hm, (kernel_size, kernel_size), sigma)


def dark_decode_heatmaps(
    heatmaps: torch.Tensor | np.ndarray,
    stride: int = 4,
    blur_kernel: int = 11,
) -> Tuple[np.ndarray, dict]:
    """
    Sub-pixel landmark coordinate decoding using the DARK algorithm.

    Args:
        heatmaps: Tensor or array of shape [B, K, H, W] containing predicted heatmaps.
        stride: Heatmap stride factor (e.g., 4 means 800x800 input -> 200x200 heatmap).
        blur_kernel: Kernel size for pre-decoding Gaussian blur (must be odd).

    Returns:
        canvas_pred: Array of shape [B, K, 2] with (x, y) coordinates in canvas space.
        aux: Dictionary containing max scores [B, K] for each landmark.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    batch_size, num_landmarks, h, w = heatmaps.shape
    canvas_pred = np.zeros((batch_size, num_landmarks, 2), dtype=np.float32)
    scores = np.zeros((batch_size, num_landmarks), dtype=np.float32)

    for b in range(batch_size):
        for k in range(num_landmarks):
            hm = heatmaps[b, k]

            # 1. Gaussian blur
            if blur_kernel > 1:
                hm_smooth = _gaussian_blur(hm, kernel_size=blur_kernel)
            else:
                hm_smooth = hm

            # 2. Integer maximum
            iy, ix = np.unravel_index(np.argmax(hm_smooth), hm_smooth.shape)
            val = float(hm_smooth[iy, ix])
            scores[b, k] = val

            # Check boundary conditions for Taylor expansion
            if 1 < ix < w - 2 and 1 < iy < h - 2 and val > 1e-4:
                # Log-domain heatmap for Taylor expansion
                # Clip to prevent log(0)
                patch = np.maximum(hm_smooth[iy - 1 : iy + 2, ix - 1 : ix + 2], 1e-6)
                log_patch = np.log(patch)

                # Center is at [1, 1]
                # First derivatives (gradient)
                dx = 0.5 * (log_patch[1, 2] - log_patch[1, 0])
                dy = 0.5 * (log_patch[2, 1] - log_patch[0, 1])
                g = np.array([dx, dy], dtype=np.float64)

                # Second derivatives (Hessian matrix)
                dxx = log_patch[1, 2] - 2.0 * log_patch[1, 1] + log_patch[1, 0]
                dyy = log_patch[2, 1] - 2.0 * log_patch[1, 1] + log_patch[0, 1]
                dxy = 0.25 * (
                    log_patch[2, 2]
                    - log_patch[2, 0]
                    - log_patch[0, 2]
                    + log_patch[0, 0]
                )

                hessian = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)

                det = dxx * dyy - dxy * dxy
                # Hessian must be negative definite for a local maximum (det > 0 and dxx < 0)
                if det > 1e-8 and dxx < 0:
                    try:
                        inv_hessian = np.linalg.inv(hessian)
                        offset = -inv_hessian @ g
                        # Clip offset within [-0.5, 0.5] pixel
                        offset_x = float(np.clip(offset[0], -0.5, 0.5))
                        offset_y = float(np.clip(offset[1], -0.5, 0.5))
                    except np.linalg.LinAlgError:
                        offset_x = 0.25 * float(np.sign(hm_smooth[iy, ix + 1] - hm_smooth[iy, ix - 1]))
                        offset_y = 0.25 * float(np.sign(hm_smooth[iy + 1, ix] - hm_smooth[iy - 1, ix]))
                else:
                    # Fallback to quarter-pixel shift
                    offset_x = 0.25 * float(np.sign(hm_smooth[iy, ix + 1] - hm_smooth[iy, ix - 1]))
                    offset_y = 0.25 * float(np.sign(hm_smooth[iy + 1, ix] - hm_smooth[iy - 1, ix]))
            elif 0 < ix < w - 1 and 0 < iy < h - 1:
                # Quarter-pixel shift near borders
                offset_x = 0.25 * float(np.sign(hm_smooth[iy, ix + 1] - hm_smooth[iy, ix - 1]))
                offset_y = 0.25 * float(np.sign(hm_smooth[iy + 1, ix] - hm_smooth[iy - 1, ix]))
            else:
                offset_x = 0.0
                offset_y = 0.0

            refined_x = (float(ix) + offset_x) * stride
            refined_y = (float(iy) + offset_y) * stride

            canvas_pred[b, k, 0] = refined_x
            canvas_pred[b, k, 1] = refined_y

    return canvas_pred, {"scores": scores}
