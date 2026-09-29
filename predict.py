"""
predict.py
============================================================
Inference Engine for Cephalometric Landmark Localization.
Supports both Single-Stage (HRNet-W48) and Two-Stage (Yao et al. Coarse-to-Fine) Modes.

Given an input lateral cephalometric radiograph (PNG/JPG/BMP):
    1. Preprocesses & scales to standard physical resolution (0.3 mm/px) with CLAHE.
    2. Stage 1 (Global Detection): HRNet-W48 predicts coarse heatmaps + DARK sub-pixel decoding.
    3. Stage 2 (Local Refinement - Yao et al.): Crops high-resolution ROI patches around coarse
       landmarks from the original radiograph and predicts sub-pixel residual displacement (ΔX, ΔY).
    4. Computes clinical cephalometric planes and metrics.
    5. Generates:
        - Structured JSON report with coarse & refined coordinates, offsets, and confidences.
        - High-resolution annotated image with landmarks, refinement vectors, and planes.

Usage:
    # Two-Stage Coarse-to-Fine (Yao et al. mode - Recommended):
    python3 predict.py --image path/to/xray.png --mode two-stage --output-dir results/

    # Single-Stage Global mode:
    python3 predict.py --image path/to/xray.png --mode global --output-dir results/
============================================================
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from config import Config, get_config
from data_loader.create_training_data import CANONICAL_29_SYMBOLS
from data_loader.dataset import IMAGENET_MEAN, IMAGENET_STD, canvas_to_original
from models.hrnet import HRNetW48
from models.two_stage_model import TwoStageCephNet, build_two_stage_cephnet
from utils.dark import dark_decode_heatmaps
from utils.visualizer import draw_landmarks_on_image, STANDARD_CEPH_PLANES


def get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CephalometricPredictor:
    """Production inference engine for cephalometric landmark localization."""

    def __init__(
        self,
        checkpoint_path: str | Path = "saved_models/best_hrnet_w48_clean.pth",
        mode: str = "two-stage",
        device: Optional[torch.device] = None,
        config: Optional[Config] = None,
    ):
        self.device = device or get_inference_device()
        self.ckpt_path = Path(checkpoint_path)
        self.mode = mode.lower()

        if not self.ckpt_path.exists():
            # Fallback to alternate checkpoint locations if available
            alt_paths = [
                Path("saved_models/best_two_stage_cephnet.pth"),
                Path("saved_models/best_hrnet_w48.pth"),
                Path("saved_models/best_hrnet_w48_clean.pth"),
            ]
            found = False
            for p in alt_paths:
                if p.exists():
                    self.ckpt_path = p
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # Config
        if config is not None:
            self.cfg = config
        else:
            cfg_used_path = self.ckpt_path.parent / "config_used.json"
            if cfg_used_path.exists():
                try:
                    self.cfg = Config.load(cfg_used_path)
                except Exception:
                    self.cfg = get_config()
            else:
                self.cfg = get_config()

        self.symbols = CANONICAL_29_SYMBOLS

        # Initialize model based on mode
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = None
        if isinstance(checkpoint, dict):
            if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
                state_dict = checkpoint["ema_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if self.mode == "two-stage":
            self.two_stage_model = TwoStageCephNet(
                num_landmarks=self.cfg.data.num_landmarks,
                patch_size=512,
                local_heatmap_size=128,
                in_channels=self.cfg.data.channels,
            )
            # If checkpoint has two_stage keys directly:
            has_two_stage_keys = any("global_module" in k for k in state_dict.keys())
            if has_two_stage_keys:
                self.two_stage_model.load_state_dict(state_dict, strict=False)
            else:
                # Load weights into global module
                clean_state = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
                self.two_stage_model.global_module.load_state_dict(clean_state, strict=False)
            self.two_stage_model.to(self.device)
            self.two_stage_model.eval()
            self.model = self.two_stage_model.global_module
        else:
            # Single-stage global HRNet-W48
            self.model = HRNetW48(
                num_landmarks=self.cfg.data.num_landmarks,
                in_channels=self.cfg.data.channels,
            )
            clean_state = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(clean_state, strict=False)
            self.model.to(self.device)
            self.model.eval()

    def preprocess_image(
        self,
        img_gray: np.ndarray,
        pixel_size_mm: float = 0.1,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Resize & pad image into square canvas matching training geometry.
        Returns:
            tensor: [1, C, S, S] float32 tensor
            M_full: 3x3 affine matrix mapping original pixels -> canvas pixels
        """
        h, w = img_gray.shape
        S = self.cfg.data.input_size
        target = self.cfg.data.target_mm_per_px

        # Physical scale
        scale = min(pixel_size_mm / target, S / w, S / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        sx, sy = new_w / w, new_h / h

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img_gray, (new_w, new_h), interpolation=interp)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resized = clahe.apply(resized)

        # Centre on canvas
        T = np.eye(3)
        T[0, 2] = (S - new_w) / 2.0
        T[1, 2] = (S - new_h) / 2.0

        canvas = cv2.warpAffine(
            resized,
            T[:2],
            (S, S),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        R = np.array([
            [sx, 0.0, 0.5 * sx - 0.5],
            [0.0, sy, 0.5 * sy - 0.5],
            [0.0, 0.0, 1.0],
        ])
        M_full = T @ R

        # Normalization
        x = canvas.astype(np.float32) / 255.0
        if self.cfg.data.channels == 3:
            x = (np.repeat(x[None], 3, axis=0) - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        else:
            x = (x[None] - 0.5) / 0.5

        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
        return tensor, M_full

    def predict(
        self,
        image: Union[str, Path, np.ndarray] = None,
        image_path: Optional[Union[str, Path]] = None,
        pixel_size_mm: float = 0.1,
        pixel_size: Optional[float] = None,
        mode: Optional[str] = None,
        draw_overlay: bool = True,
    ) -> Dict[str, object]:
        """
        Run inference on an input cephalometric image (file path or in-memory array).
        """
        t_start = time.perf_counter()
        ps = pixel_size if pixel_size is not None else pixel_size_mm
        active_mode = (mode or self.mode).lower()

        img_input = image if image is not None else image_path
        if img_input is None:
            raise ValueError("Must provide either 'image' or 'image_path'.")

        img_label = "in_memory_image"
        if isinstance(img_input, (str, Path)):
            img_path = Path(img_input)
            img_label = str(img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            orig_bgr = cv2.imread(str(img_path))
            if orig_bgr is None:
                raise ValueError(f"Could not decode image at: {img_path}")
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                orig_bgr = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
            elif img_input.ndim == 3:
                orig_bgr = img_input.copy()
            else:
                raise ValueError(f"Unsupported image array shape: {img_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = orig_gray.shape

        tensor, M_full = self.preprocess_image(orig_gray, pixel_size_mm=ps)
        tensor = tensor.to(self.device)

        t_stage1_start = time.perf_counter()
        with torch.no_grad():
            heatmaps = self.model(tensor)
            canvas_pred, aux = dark_decode_heatmaps(
                heatmaps,
                stride=self.cfg.model.dark_stride,
                blur_kernel=self.cfg.model.dark_blur_kernel,
            )

        # Invert affine transform to get original coordinates
        coarse_coords = canvas_to_original(canvas_pred[0], M_full)  # [K, 2]
        scores = aux["scores"][0]
        t_stage1 = time.perf_counter() - t_stage1_start

        t_stage2 = 0.0
        refined_coords = coarse_coords.copy()
        offsets_px = np.zeros_like(coarse_coords)

        # Stage 2: Local Refinement (Yao et al.)
        if active_mode == "two-stage" and hasattr(self, "two_stage_model"):
            t_stage2_start = time.perf_counter()
            with torch.no_grad():
                # Prepare high-res patch source tensor [1, 3, H_orig, W_orig]
                orig_norm = orig_bgr.astype(np.float32) / 255.0
                orig_norm = (orig_norm.transpose(2, 0, 1) - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
                orig_tensor = torch.from_numpy(np.ascontiguousarray(orig_norm, dtype=np.float32)).unsqueeze(0).to(self.device)

                coarse_tensor = torch.from_numpy(coarse_coords.astype(np.float32)).unsqueeze(0).to(self.device)
                patches = self.two_stage_model.extract_roi_patches(orig_tensor, coarse_tensor)
                l_ids = torch.arange(len(self.symbols), device=self.device)
                local_heatmaps = self.two_stage_model.local_module(patches, l_ids)
                local_heatmaps_reshaped = local_heatmaps.view(
                    1, len(self.symbols), 1, self.two_stage_model.local_heatmap_size, self.two_stage_model.local_heatmap_size
                )
                refined_coords_t, offsets_t, local_conf_t = self.two_stage_model.decode_local_heatmaps(
                    local_heatmaps_reshaped, coarse_tensor
                )
                refined_coords = refined_coords_t[0].cpu().numpy()
                offsets_px = offsets_t[0].cpu().numpy()
                scores = scores * (0.8 + 0.2 * local_conf_t[0].cpu().numpy())

            t_stage2 = time.perf_counter() - t_stage2_start

        landmarks = []
        for i, sym in enumerate(self.symbols):
            cx, cy = float(coarse_coords[i, 0]), float(coarse_coords[i, 1])
            rx, ry = float(refined_coords[i, 0]), float(refined_coords[i, 1])
            dx, dy = float(offsets_px[i, 0]), float(offsets_px[i, 1])
            dist_mm = math.sqrt(dx * dx + dy * dy) * ps
            sc = float(scores[i])

            landmarks.append({
                "symbol": sym,
                "x": round(rx, 2),
                "y": round(ry, 2),
                "coarse_x": round(cx, 2),
                "coarse_y": round(cy, 2),
                "offset_dx_px": round(dx, 2),
                "offset_dy_px": round(dy, 2),
                "refinement_delta_mm": round(dist_mm, 3),
                "confidence": round(sc, 4),
            })

        total_time = time.perf_counter() - t_start

        out = {
            "image_path": img_label,
            "mode": active_mode,
            "architecture": "Yao et al. Two-Stage Coarse-to-Fine" if active_mode == "two-stage" else "HRNet-W48 Global",
            "original_shape": (h_orig, w_orig),
            "pixel_size_mm": ps,
            "execution_time_seconds": round(total_time, 4),
            "stage1_time_seconds": round(t_stage1, 4),
            "stage2_time_seconds": round(t_stage2, 4),
            "landmarks": landmarks,
            "image_bgr": orig_bgr,
        }

        if draw_overlay:
            annotated = orig_bgr.copy()
            # Draw standard cephalometric planes
            coords_dict = {sym: (refined_coords[i, 0], refined_coords[i, 1]) for i, sym in enumerate(self.symbols)}
            for sym1, sym2, col in STANDARD_CEPH_PLANES:
                if sym1 in coords_dict and sym2 in coords_dict:
                    p1 = (int(round(coords_dict[sym1][0])), int(round(coords_dict[sym1][1])))
                    p2 = (int(round(coords_dict[sym2][0])), int(round(coords_dict[sym2][1])))
                    cv2.line(annotated, p1, p2, col, 2, cv2.LINE_AA)

            # If two-stage, draw subtle vectors from coarse to refined
            if active_mode == "two-stage":
                for i in range(len(self.symbols)):
                    p_c = (int(round(coarse_coords[i, 0])), int(round(coarse_coords[i, 1])))
                    p_r = (int(round(refined_coords[i, 0])), int(round(refined_coords[i, 1])))
                    cv2.line(annotated, p_c, p_r, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(annotated, p_c, 3, (0, 255, 255), -1, cv2.LINE_AA)

            # Draw final landmarks with labels
            annotated = draw_landmarks_on_image(
                annotated,
                refined_coords,
                symbols=self.symbols,
                color=(0, 0, 255),
                radius=6,
                draw_labels=True,
            )
            out["annotated_image"] = annotated

        return out


def main():
    parser = argparse.ArgumentParser(description="Predict landmarks on lateral cephalometric radiograph.")
    parser.add_argument("--image", type=str, required=True, help="Path to input X-ray image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/best_two_stage_cephnet.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="two-stage",
        choices=["two-stage", "global"],
        help="Inference mode: 'two-stage' (Yao et al. Coarse-to-Fine) or 'global' (HRNet-W48).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.1,
        help="Image resolution in mm/pixel (default: 0.1 mm/px).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save visual and JSON results.",
    )
    parser.add_argument("--no-vis", action="store_true", help="Disable generating visualization image.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = CephalometricPredictor(checkpoint_path=args.checkpoint, mode=args.mode)
    res = predictor.predict(
        image_path=args.image,
        pixel_size_mm=args.pixel_size,
        mode=args.mode,
        draw_overlay=not args.no_vis,
    )

    stem = Path(args.image).stem
    json_path = out_dir / f"{stem}_landmarks.json"

    # Clean non-serializable fields for json
    res_clean = {k: v for k, v in res.items() if k not in ("image_bgr", "annotated_image")}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_clean, f, indent=2)
    print(f"[OK] Landmark coordinates saved: {json_path}")

    if "annotated_image" in res:
        img_out_path = out_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(img_out_path), res["annotated_image"])
        print(f"[OK] Annotated image overlay saved: {img_out_path}")


if __name__ == "__main__":
    main()
