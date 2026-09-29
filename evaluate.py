"""
evaluate.py
============================================================
Independent evaluation and benchmarking script for the Cephalometric AI model.

Features:
    - Evaluates any saved checkpoint (.pth) on test or valid splits.
    - Computes Mean Radial Error (MRE), Median, Std, and SDR at 2.0, 2.5, 3.0, 4.0 mm.
    - Detailed breakdowns by source dataset (Aariz, ISBI2015, DentalCepha102) and machine.
    - Optional generation of qualitative visual overlays with anatomical traces and error lines.
    - Exports metrics to JSON.

Usage:
    python3 evaluate.py --checkpoint saved_models/best_hrnet_w48.pth --split test
    python3 evaluate.py --split valid --save-overlays 10
============================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, get_config
from data_loader.dataset import CephDataset, canvas_to_original
from models.hrnet import HRNetW48
from utils.dark import dark_decode_heatmaps
from utils.metrics import compute_sdr, radial_error_mm, summarize_metrics
from utils.visualizer import draw_comparison_overlay


def get_eval_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_evaluation(
    checkpoint_path: str | Path,
    data_root: str = "training_data",
    split: str = "test",
    batch_size: int = 4,
    save_overlays: int = 0,
    output_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Run full evaluation on a specified split and compute metrics.
    """
    if device is None:
        device = get_eval_device()

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 72)
    print(f"Evaluating Cephalometric Model on split: [{split.upper()}]")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")
    print(f"Data Root  : {data_root}")
    print("=" * 72)

    # 1. Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load configuration
    cfg = get_config()
    cfg_used_path = ckpt_path.parent / "config_used.json"
    if cfg_used_path.exists():
        try:
            cfg = Config.load(cfg_used_path)
        except Exception:
            pass

    # 2. Build model
    model = HRNetW48(
        num_landmarks=cfg.data.num_landmarks,
        in_channels=cfg.data.channels,
    )

    state_dict = None
    if isinstance(checkpoint, dict):
        if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
            state_dict = checkpoint["ema_state_dict"]
            print("[INFO] Using EMA model weights from checkpoint.")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean prefixes
    clean_state = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.", "")
        clean_state[clean_k] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()

    # 3. Build Dataset & DataLoader
    dataset = CephDataset(
        root=data_root,
        split=split,
        input_size=cfg.data.input_size,
        stride=cfg.data.stride,
        sigma=cfg.data.sigma,
        target_mm_per_px=cfg.data.target_mm_per_px,
        augment=False,
        channels=cfg.data.channels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Total evaluation samples: {len(dataset)}")

    # 4. Evaluation Loop
    all_errors_mm: List[float] = []
    errors_by_source: Dict[str, List[float]] = {}
    errors_by_machine: Dict[str, List[float]] = {}
    saved_overlay_count = 0

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ckpt_path.parent / f"eval_{split}_results"
        out_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir = out_dir / "overlays"
    if save_overlays > 0:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split}]"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].cpu().numpy()
            matrices = batch["M"].cpu().numpy()
            gt_orig = batch["landmarks_orig"].cpu().numpy()
            pixel_sizes = batch["pixel_size"].cpu().numpy()
            sources = batch["source"]
            machines = batch["machine"]
            unique_ids = batch["unique_id"]

            heatmaps = model(images)
            canvas_pred, _ = dark_decode_heatmaps(
                heatmaps,
                stride=cfg.model.dark_stride,
                blur_kernel=cfg.model.dark_blur_kernel,
            )

            b_size = images.shape[0]
            for b in range(b_size):
                pred_orig_b = canvas_to_original(canvas_pred[b], matrices[b])
                gt_orig_b = gt_orig[b]
                mask_b = masks[b]
                px_size = float(pixel_sizes[b])
                src = sources[b]
                mach = machines[b]
                uid = unique_ids[b]

                errs = radial_error_mm(pred_orig_b, gt_orig_b, mask_b, px_size)
                valid_errs = errs[~np.isnan(errs)].tolist()

                if valid_errs:
                    all_errors_mm.extend(valid_errs)
                    errors_by_source.setdefault(src, []).extend(valid_errs)
                    errors_by_machine.setdefault(mach, []).extend(valid_errs)

                # Optional visual overlay
                if save_overlays > 0 and saved_overlay_count < save_overlays:
                    # Load original image for overlay
                    sample_info = next((s for s in dataset.samples if s["unique_id"] == uid), None)
                    if sample_info:
                        orig_img_path = Path(data_root) / sample_info["image_relative_path"]
                        orig_bgr = cv2.imread(str(orig_img_path))
                        if orig_bgr is not None:
                            annotated = draw_comparison_overlay(
                                orig_bgr,
                                pred_orig_b,
                                gt_orig_b,
                                symbols=dataset.symbols,
                                mask=mask_b,
                            )
                            cv2.imwrite(str(overlays_dir / f"{uid}_comparison.jpg"), annotated)
                            saved_overlay_count += 1

    # 5. Summarize metrics
    summary = summarize_metrics(all_errors_mm, cfg.train.sdr_thresholds)

    by_source_summary = {
        src: summarize_metrics(errs, cfg.train.sdr_thresholds)
        for src, errs in sorted(errors_by_source.items())
    }

    by_machine_summary = {
        mach: summarize_metrics(errs, cfg.train.sdr_thresholds)
        for mach, errs in sorted(errors_by_machine.items())
    }

    # 6. Pretty Print Results
    print("\n" + "=" * 72)
    print(f"BENCHMARK RESULTS [{split.upper()}] - {len(dataset)} SAMPLES")
    print("=" * 72)
    print(f"Mean Radial Error (MRE) : {summary['mre']:.3f} mm")
    print(f"Median Radial Error     : {summary['median']:.3f} mm")
    print(f"Standard Deviation      : {summary['std']:.3f} mm")
    print(f"Valid Landmarks Evaluated: {summary['count']}")
    print("-" * 72)
    sdr_line = " | ".join(f"{k}: {summary[k]:.1f}%" for k in sorted(summary) if k.startswith("SDR_"))
    print(f"SDR (Success Detection) : {sdr_line}")
    print("-" * 72)

    print("Breakdown By Dataset:")
    for src, res in by_source_summary.items():
        print(f"  * {src:<18} MRE: {res['mre']:.3f} mm | Median: {res['median']:.3f} mm | SDR 2.0mm: {res.get('SDR_2.0mm', 0):.1f}% (n={res['count']})")

    print("-" * 72)
    print("Breakdown By Imaging Machine:")
    for mach, res in by_machine_summary.items():
        print(f"  * {mach:<22} MRE: {res['mre']:.3f} mm | SDR 2.0mm: {res.get('SDR_2.0mm', 0):.1f}% (n={res['count']})")
    print("=" * 72)

    # 7. Export JSON Report
    report = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "overall": summary,
        "by_dataset": by_source_summary,
        "by_machine": by_machine_summary,
    }

    report_path = out_dir / f"evaluation_{split}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Full report exported to: {report_path}")
    if saved_overlay_count > 0:
        print(f"[OK] {saved_overlay_count} visual overlays saved to: {overlays_dir}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained cephalometric model.")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_hrnet_w48.pth")
    parser.add_argument("--data-root", type=str, default="training_data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-overlays", type=int, default=5, help="Number of qualitative overlays to save.")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        save_overlays=args.save_overlays,
        output_dir=args.output_dir,
    )
