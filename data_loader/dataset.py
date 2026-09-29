"""
data_loader/dataset.py

PyTorch dataset for the unified cephalometric index.

Pipeline per sample
  1. Resample to a common physical scale (target_mm_per_px). If the result does not fit the square
     canvas, the scale is reduced just enough to fit, so nothing is cropped.
  2. Optional CLAHE (deterministic, applied to every split).
  3. Pad to a square canvas + (train only) random rotation / scale / translation.
  4. (train only) gamma, contrast, brightness, noise. No horizontal flip in v1.
  5. Gaussian heatmaps at 1/stride resolution; landmarks that are invalid OR pushed outside the
     canvas by augmentation get mask = 0 (their heatmap is all zeros and the loss ignores them).

Every sample also carries the 3x3 matrix M (original px -> canvas px) and mm_per_canvas_px, so
predictions can be mapped back and errors reported in real millimetres.

Smoke test (writes a picture you should eyeball):
    python3 data_loader/dataset.py --root training_data --split train
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_heatmaps(pts_hm: np.ndarray, mask: np.ndarray, h: int, w: int, sigma: float) -> np.ndarray:
    """pts_hm: (K,2) in heatmap coordinates, mask: (K,) -> (K,h,w) float32. Masked rows are all zero."""
    xs = np.arange(w, dtype=np.float32)[None, None, :]
    ys = np.arange(h, dtype=np.float32)[None, :, None]
    mx = pts_hm[:, 0][:, None, None]
    my = pts_hm[:, 1][:, None, None]
    hm = np.exp(-((xs - mx) ** 2 + (ys - my) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    return hm * mask[:, None, None]


def decode_heatmaps(hm: torch.Tensor, stride: int) -> np.ndarray:
    """(B,K,H,W) -> (B,K,2) canvas coordinates. Argmax + quarter-pixel shift toward the higher neighbour.
    (Swap for DARK decoding later if you want a bit more precision.)"""
    hm = hm.detach().cpu().numpy()
    B, K, H, W = hm.shape
    out = np.zeros((B, K, 2), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            m = hm[b, k]
            iy, ix = np.unravel_index(int(m.argmax()), m.shape)
            x, y = float(ix), float(iy)
            if 0 < ix < W - 1:
                x += 0.25 * np.sign(m[iy, ix + 1] - m[iy, ix - 1])
            if 0 < iy < H - 1:
                y += 0.25 * np.sign(m[iy + 1, ix] - m[iy - 1, ix])
            out[b, k] = (x * stride, y * stride)
    return out


def masked_heatmap_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target (B,K,H,W), mask (B,K). Missing landmarks never contribute."""
    se = ((pred - target) ** 2) * mask[:, :, None, None]
    denom = mask.sum().clamp(min=1.0) * pred.shape[-1] * pred.shape[-2]
    return se.sum() / denom


def canvas_to_original(pts_canvas: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(K,2) canvas coords + 3x3 M (orig->canvas) -> (K,2) original-image pixel coords."""
    Minv = np.linalg.inv(M)
    ones = np.ones((pts_canvas.shape[0], 1), dtype=np.float64)
    p = np.hstack([pts_canvas.astype(np.float64), ones]) @ Minv.T
    return p[:, :2]


def radial_error_mm(pred_orig: np.ndarray, gt_orig: np.ndarray, mask: np.ndarray, pixel_size: float) -> np.ndarray:
    """Per-landmark radial error in mm; NaN where mask == 0."""
    err = np.linalg.norm(pred_orig - gt_orig, axis=1) * pixel_size
    err[mask == 0] = np.nan
    return err


def _to3x3(m2x3: np.ndarray) -> np.ndarray:
    return np.vstack([m2x3, [0.0, 0.0, 1.0]]).astype(np.float64)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class CephDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        sources: Optional[Sequence[str]] = None,
        input_size: int = 1024,
        stride: int = 4,
        sigma: float = 2.5,
        target_mm_per_px: float = 0.3,
        augment: Optional[bool] = None,
        use_clahe: bool = True,
        channels: int = 3,
        rot_deg: float = 10.0,
        scale_range: Sequence[float] = (0.9, 1.1),
        trans_frac: float = 0.05,
    ):
        assert split in {"train", "valid", "test"}
        assert channels in {1, 3}
        assert input_size % stride == 0

        self.root = Path(root)
        with open(self.root / "master_dataset_index.json", "r", encoding="utf-8") as f:
            index = json.load(f)
        self.symbols: List[str] = index["canonical_landmarks_29"]
        self.samples = [s for s in index["samples"]
                        if s["split"] == split and (sources is None or s["source_dataset"] in sources)]
        if not self.samples:
            raise ValueError(f"No samples for split={split}, sources={sources}")

        self.S = input_size
        self.stride = stride
        self.hm_size = input_size // stride
        self.sigma = sigma
        self.target = target_mm_per_px
        self.augment = (split == "train") if augment is None else augment
        self.use_clahe = use_clahe
        self.channels = channels
        self.rot_deg, self.scale_range, self.trans_frac = rot_deg, scale_range, trans_frac

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        S = self.S
        img = cv2.imread(str(self.root / s["image_relative_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(s["image_relative_path"])
        h, w = img.shape

        pts = np.array([[lm["x"], lm["y"]] for lm in s["landmarks"]], dtype=np.float64)
        valid = np.array([lm["valid"] for lm in s["landmarks"]], dtype=np.float32)

        # 1. physical rescale (never larger than what fits the canvas)
        scale = min(s["pixel_size"] / self.target, S / w, S / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        sx, sy = new_w / w, new_h / h
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        pts[:, 0] = (pts[:, 0] + 0.5) * sx - 0.5
        pts[:, 1] = (pts[:, 1] + 0.5) * sy - 0.5

        # 2. CLAHE
        if self.use_clahe:
            resized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)

        # 3. centre on canvas (+ geometric augmentation)
        T = np.eye(3)
        T[0, 2], T[1, 2] = (S - new_w) / 2.0, (S - new_h) / 2.0
        A = np.eye(3)
        if self.augment:
            angle = random.uniform(-self.rot_deg, self.rot_deg)
            zoom = random.uniform(*self.scale_range)
            A = _to3x3(cv2.getRotationMatrix2D((S / 2.0, S / 2.0), angle, zoom))
            A[0, 2] += random.uniform(-self.trans_frac, self.trans_frac) * S
            A[1, 2] += random.uniform(-self.trans_frac, self.trans_frac) * S
        M = A @ T  # original px -> canvas px (includes the rescale via pts above, see below)

        canvas = cv2.warpAffine(resized, M[:2], (S, S), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pts_c = (M[:2, :2] @ pts.T).T + M[:2, 2]

        # Full original->canvas matrix (rescale first, then M) for mapping predictions back.
        R = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])
        M_full = M @ R

        # 4. photometric augmentation
        x = canvas.astype(np.float32) / 255.0
        if self.augment:
            x = np.power(np.clip(x, 0, 1), random.uniform(0.8, 1.25))
            x = (x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.08, 0.08)
            if random.random() < 0.3:
                x = x + np.random.normal(0.0, 0.01, x.shape).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)

        # 5. visibility + heatmaps
        inside = (pts_c[:, 0] >= 0) & (pts_c[:, 0] < S) & (pts_c[:, 1] >= 0) & (pts_c[:, 1] < S)
        mask = (valid * inside).astype(np.float32)
        hms = make_heatmaps((pts_c / self.stride).astype(np.float32), mask, self.hm_size, self.hm_size, self.sigma)

        if self.channels == 3:
            x = (np.repeat(x[None], 3, axis=0) - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        else:
            x = (x[None] - 0.5) / 0.5

        px_per_orig = math.sqrt(abs(np.linalg.det(M_full[:2, :2])))
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "heatmaps": torch.from_numpy(hms),
            "mask": torch.from_numpy(mask),
            "landmarks_canvas": torch.from_numpy(pts_c.astype(np.float32)),
            "landmarks_orig": torch.tensor([[lm["x"], lm["y"]] for lm in s["landmarks"]], dtype=torch.float32),
            "M": torch.from_numpy(M_full.astype(np.float32)),
            "pixel_size": torch.tensor(s["pixel_size"], dtype=torch.float32),
            "mm_per_canvas_px": torch.tensor(s["pixel_size"] / px_per_orig, dtype=torch.float32),
            "unique_id": s["unique_id"],
            "source": s["source_dataset"],
            "machine": s["machine"],
        }


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
def _smoke_test(root: str, split: str, n: int = 3) -> None:
    ds = CephDataset(root, split, augment=True)
    print(f"{len(ds)} samples | canvas {ds.S} | heatmap {ds.hm_size} | {len(ds.symbols)} landmarks")
    out = Path(root) / "qc_overlays"
    out.mkdir(exist_ok=True)
    for i in random.sample(range(len(ds)), min(n, len(ds))):
        it = ds[i]
        img = it["image"].numpy()
        img = img[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0] if ds.channels == 3 else img[0] * 0.5 + 0.5
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        hm = it["heatmaps"].numpy().max(axis=0)
        hm = cv2.resize((hm * 255).astype(np.uint8), (ds.S, ds.S))
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[..., 2] = np.maximum(vis[..., 2], hm)  # heatmap in red channel
        for (px, py), m, sym in zip(it["landmarks_canvas"].numpy(), it["mask"].numpy(), ds.symbols):
            if m:
                cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 0), -1)
        cv2.imwrite(str(out / f"dataset_check_{it['unique_id']}.jpg"), cv2.resize(vis, None, fx=0.6, fy=0.6))

        # round-trip: canvas -> original must reproduce the original annotation
        back = canvas_to_original(it["landmarks_canvas"].numpy(), it["M"].numpy().astype(np.float64))
        err = radial_error_mm(back, it["landmarks_orig"].numpy(), it["mask"].numpy(), float(it["pixel_size"]))
        print(f"{it['unique_id']:<24} valid={int(it['mask'].sum()):>2}  "
              f"mm/canvas-px={float(it['mm_per_canvas_px']):.3f}  round-trip max err={np.nanmax(err):.4f} mm")
    print(f"[ok] pictures in {out} (green = landmark, red glow = heatmap; they must coincide)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="training_data")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    _smoke_test(a.root, a.split)