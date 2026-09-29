# Model Checkpoints Guide

This directory contains the production-ready model weights for Cephalometric Landmark Detection.

### Available Checkpoints

| Checkpoint File | Size | Architecture | Description |
| :--- | :---: | :--- | :--- |
| **`best_two_stage_cephnet.pth`** | ~264 MB | Yao et al. Two-Stage Coarse-to-Fine | **Recommended for Production.** Combines global detection + high-res patch refinement. |
| **`best_hrnet_w48_clean.pth`** | ~252 MB | HRNet-W48 Single-Stage | Model state dict only (no optimizer states). Clean export. |
| **`best_hrnet_w48_fp16.pth`** | ~126 MB | HRNet-W48 FP16 | Half-precision lightweight weights for edge devices. |

### How to Use

By default, `predict.py` and `api_server.py` automatically detect and load `best_two_stage_cephnet.pth`.

To override the checkpoint:
```bash
export CHECKPOINT_PATH=saved_models/best_two_stage_cephnet.pth
python3 api_server.py
```
Or via CLI:
```bash
python3 predict.py --image path/to/image.png --checkpoint saved_models/best_two_stage_cephnet.pth --mode two-stage
```
