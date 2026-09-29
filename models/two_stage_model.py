"""
models/two_stage_model.py
============================================================
Two-Stage Landmark Localization Framework (Yao et al. Architecture)
Reference: Jie Yao et al., "Automatic localization of cephalometric landmarks based
on convolutional neural network", Am J Orthod Dentofacial Orthop (AJO-DO), 2022; 161(3): e250-e259.
DOI: 10.1016/j.ajodo.2021.09.012

Key Technical Principles:
-------------------------
1. Architectural Change:
   - Stage 1 (Global Detection Module): Full radiograph input at normalized resolution (800x800).
     HRNet-W48 extracts global anatomical context and predicts coarse landmark locations.
   - Stage 2 (Locally Modified Module): Crops high-resolution ROI patches (>= 512x512) around
     each coarse landmark at full native resolution to capture micro-anatomical structures.

2. Output Representation:
   - High-Precision Gaussian Heatmaps across both stages instead of direct coordinate regression.
   - Preserves 2D spatial topological continuity and anatomical contour boundaries.
   - Sub-pixel coordinate extraction via DARK (Distribution-Aware Coordinate Representation).

3. Image Resolution:
   - Local ROI patches maintain high resolution (>= 512x512 px) from native radiograph.

4. Loss Function Compatibility:
   - Trained with Adaptive Wing (AWing) Loss + Weighted Heatmap MSE.
============================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hrnet import HRNetW48, build_hrnet_w48
from utils.dark import dark_decode_heatmaps


class ConvBlock(nn.Module):
    """Residual Conv-BN-Mish block with optional down/up-sampling shortcut."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Mish(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class LocallyModifiedModule(nn.Module):
    """
    Locally Modified Module (Yao et al. 2022).
    
    Ingests high-resolution ROI patches (e.g. 512x512 px) cropped from native radiographs
    and outputs fine-grained Gaussian heatmaps (e.g. 128x128 px) for each landmark.
    """
    def __init__(
        self,
        num_landmarks: int = 29,
        in_channels: int = 3,
        patch_size: int = 512,
        heatmap_size: int = 128,
        feat_dim: int = 64,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size

        # Landmark category query embedding
        self.landmark_embedding = nn.Embedding(num_landmarks, 32)

        # Multi-scale Encoder (512x512 -> 256x256 -> 128x128 -> 64x64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 256
            nn.BatchNorm2d(32),
            nn.Mish(inplace=True),
        )
        self.enc2 = ConvBlock(32, 48, stride=2)   # 128
        self.enc3 = ConvBlock(48, feat_dim, stride=2)  # 64

        # Fusion with landmark embedding at bottleneck
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim + 32, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Mish(inplace=True),
        )

        # Decoder to generate high-resolution Gaussian heatmap (64x64 -> 128x128)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 48, kernel_size=4, stride=2, padding=1, bias=False), # 128
            nn.BatchNorm2d(48),
            nn.Mish(inplace=True),
        )
        self.dec_conv = ConvBlock(48 + 48, 32, stride=1) # skip-connection from enc2

        # Final local Gaussian heatmap prediction head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Mish(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 2D Gaussian spatial anchor prior centered at patch origin
        y, x = torch.meshgrid(
            torch.arange(heatmap_size, dtype=torch.float32),
            torch.arange(heatmap_size, dtype=torch.float32),
            indexing="ij"
        )
        cx, cy = heatmap_size / 2.0, heatmap_size / 2.0
        sigma = 16.0
        prior = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2)) * 5.0
        self.register_buffer("center_prior", prior.unsqueeze(0).unsqueeze(0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        patches: torch.Tensor,
        landmark_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patches: [N, C, 512, 512] ROI patches (N = B * K)
            landmark_indices: [N] landmark IDs
        Returns:
            local_heatmaps: [N, 1, 128, 128] high-resolution Gaussian response heatmaps
        """
        N, _, H, W = patches.shape
        e1 = self.enc1(patches)      # [N, 32, 256, 256]
        e2 = self.enc2(e1)           # [N, 48, 128, 128]
        e3 = self.enc3(e2)           # [N, feat_dim, 64, 64]

        # Broadcast landmark embedding
        l_emb = self.landmark_embedding(landmark_indices)  # [N, 32]
        l_emb_map = l_emb[:, :, None, None].expand(-1, -1, e3.shape[2], e3.shape[3])

        bottleneck = self.fusion(torch.cat([e3, l_emb_map], dim=1)) # [N, feat_dim, 64, 64]

        d1 = self.dec1(bottleneck)   # [N, 48, 128, 128]
        # Skip connection from encoder stage 2
        fused = self.dec_conv(torch.cat([d1, e2], dim=1)) # [N, 32, 128, 128]

        logits = self.heatmap_head(fused) # [N, 1, 128, 128]
        heatmaps = torch.sigmoid(logits + self.center_prior)
        return heatmaps


class TwoStageCephNet(nn.Module):
    """
    Two-Stage Cephalometric Landmark Localization Framework (Yao et al. 2022).
    
    Stage 1: Global Detection Module (HRNet-W48) -> Coarse Heatmaps & Coarse Coordinates.
    Stage 2: Locally Modified Module (High-Resolution Heatmaps on >= 512x512 ROI Patches).
    """
    def __init__(
        self,
        num_landmarks: int = 29,
        patch_size: int = 512,
        local_heatmap_size: int = 128,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.patch_size = patch_size
        self.local_heatmap_size = local_heatmap_size

        # Stage 1: Global Detection Module (HRNet-W48)
        self.global_module = build_hrnet_w48(
            num_landmarks=num_landmarks,
            in_channels=in_channels,
        )

        # Stage 2: Locally Modified Module (Yao et al.)
        self.local_module = LocallyModifiedModule(
            num_landmarks=num_landmarks,
            in_channels=in_channels,
            patch_size=patch_size,
            heatmap_size=local_heatmap_size,
            feat_dim=64,
        )

    def load_global_weights(self, checkpoint_path: str | Path | dict) -> None:
        """Loads pretrained weights into the Stage 1 Global Detection Module."""
        if isinstance(checkpoint_path, (str, Path)):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            ckpt = checkpoint_path

        state_dict = (
            ckpt.get("ema_state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
        cleaned = {}
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "").replace("global_module.", "")
            if not k_clean.startswith("local_module."):
                cleaned[k_clean] = v

        self.global_module.load_state_dict(cleaned, strict=False)

    def extract_roi_patches(
        self,
        images_tensor: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract square high-resolution (512x512) ROI patches centered at coordinates.
        
        Args:
            images_tensor: [B, C, H, W] full-resolution image tensor
            coords: [B, K, 2] in pixel space (X, Y)
        Returns:
            patches: [B * K, C, patch_size, patch_size]
        """
        B, C, H, W = images_tensor.shape
        K = coords.shape[1]
        P = self.patch_size
        device = images_tensor.device

        half_p = P / 2.0
        base_y, base_x = torch.meshgrid(
            torch.linspace(-half_p, half_p, P, device=device, dtype=torch.float32),
            torch.linspace(-half_p, half_p, P, device=device, dtype=torch.float32),
            indexing="ij"
        )  # [P, P]

        # Centers shape: [B, K, 1, 1, 2]
        centers = coords.unsqueeze(2).unsqueeze(3)
        grid_x = centers[..., 0] + base_x.unsqueeze(0).unsqueeze(0)
        grid_y = centers[..., 1] + base_y.unsqueeze(0).unsqueeze(0)

        # Normalize to [-1, 1] for torch.nn.functional.grid_sample
        norm_x = (grid_x / (W - 1.0)) * 2.0 - 1.0
        norm_y = (grid_y / (H - 1.0)) * 2.0 - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)  # [B, K, P, P, 2]
        grid = grid.view(B * K, P, P, 2)

        # Expand images: [B, 1, C, H, W] -> [B, K, C, H, W] -> [B*K, C, H, W]
        img_exp = images_tensor.unsqueeze(1).expand(B, K, C, H, W).reshape(B * K, C, H, W)
        patches = F.grid_sample(img_exp, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return patches

    def decode_local_heatmaps(
        self,
        local_heatmaps: torch.Tensor,
        coarse_coords: torch.Tensor,
        stride: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes sub-pixel peaks from local high-resolution Gaussian heatmaps
        using DARK Taylor expansion and projects back to global image coordinates.
        
        Args:
            local_heatmaps: [B, K, 1, H_hm, W_hm]
            coarse_coords: [B, K, 2] coarse coordinates
        Returns:
            refined_coords: [B, K, 2]
            offsets: [B, K, 2] (dx, dy)
            local_conf: [B, K]
        """
        B, K, _, H_hm, W_hm = local_heatmaps.shape
        device = coarse_coords.device
        hm_flat = local_heatmaps.view(B * K, 1, H_hm, W_hm).squeeze(1).detach().cpu().numpy()

        # Gaussian blur + DARK sub-pixel log-domain Taylor expansion
        peaks = np.zeros((B * K, 2), dtype=np.float32)
        confidences = np.zeros(B * K, dtype=np.float32)

        for i in range(B * K):
            hm = hm_flat[i]
            hm_blurred = cv2.GaussianBlur(hm, (11, 11), 11.0 / 6.0)
            max_idx = int(np.argmax(hm_blurred))
            vy, ux = np.unravel_index(max_idx, hm_blurred.shape)
            confidences[i] = float(hm[vy, ux])

            # Taylor expansion around (ux, vy)
            px, py = float(ux), float(vy)
            if 1 <= ux < W_hm - 1 and 1 <= vy < H_hm - 1:
                g = np.array([
                    (hm_blurred[vy, ux + 1] - hm_blurred[vy, ux - 1]) / 2.0,
                    (hm_blurred[vy + 1, ux] - hm_blurred[vy - 1, ux]) / 2.0
                ])
                H = np.array([
                    [hm_blurred[vy, ux + 1] - 2 * hm_blurred[vy, ux] + hm_blurred[vy, ux - 1],
                     (hm_blurred[vy + 1, ux + 1] - hm_blurred[vy + 1, ux - 1] - hm_blurred[vy - 1, ux + 1] + hm_blurred[vy - 1, ux - 1]) / 4.0],
                    [(hm_blurred[vy + 1, ux + 1] - hm_blurred[vy + 1, ux - 1] - hm_blurred[vy - 1, ux + 1] + hm_blurred[vy - 1, ux - 1]) / 4.0,
                     hm_blurred[vy + 1, ux] - 2 * hm_blurred[vy, ux] + hm_blurred[vy - 1, ux]]
                ])
                try:
                    delta = -np.linalg.solve(H + 1e-4 * np.eye(2), g)
                    px += np.clip(delta[0], -1.0, 1.0)
                    py += np.clip(delta[1], -1.0, 1.0)
                except np.linalg.LinAlgError:
                    pass

            peaks[i] = [px, py]

        peaks = torch.from_numpy(peaks).to(device).view(B, K, 2)
        confidences = torch.from_numpy(confidences).to(device).view(B, K)

        # Map local heatmap coordinates to residual displacement in patch pixels
        # Center of heatmap (W_hm/2, H_hm/2) corresponds to patch center (coarse coordinate)
        scale_x = float(self.patch_size) / float(W_hm)
        scale_y = float(self.patch_size) / float(H_hm)

        dx = (peaks[..., 0] - (W_hm / 2.0)) * scale_x
        dy = (peaks[..., 1] - (H_hm / 2.0)) * scale_y
        offsets = torch.stack([dx, dy], dim=-1)

        refined_coords = coarse_coords + offsets
        return refined_coords, offsets, confidences

    def forward(
        self,
        global_images: torch.Tensor,
        full_res_images: Optional[torch.Tensor] = None,
        stride: int = 4,
        dark_blur_kernel: int = 11,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass of Yao et al. Two-Stage Network.
        
        Args:
            global_images: [B, C, 800, 800] canvas images for Stage 1.
            full_res_images: [B, C, H_orig, W_orig] images for Stage 2 ROI crops.
        """
        # ====================================================
        # Stage 1: Global Detection Module (Coarse Localization)
        # ====================================================
        coarse_heatmaps = self.global_module(global_images)
        coarse_coords_canvas, aux = dark_decode_heatmaps(
            coarse_heatmaps, stride=stride, blur_kernel=dark_blur_kernel
        )
        coarse_conf = aux["scores"]

        device = global_images.device
        coarse_tensor = torch.from_numpy(coarse_coords_canvas.astype(np.float32)).to(device)
        B, K, _ = coarse_tensor.shape

        # ====================================================
        # Stage 2: Locally Modified Module (High-Res Heatmap Refinement)
        # ====================================================
        ref_source = full_res_images if full_res_images is not None else global_images
        scale_x = ref_source.shape[3] / global_images.shape[3]
        scale_y = ref_source.shape[2] / global_images.shape[2]

        # Project coarse coordinates to ref_source space
        coords_ref = coarse_tensor.clone()
        coords_ref[..., 0] *= scale_x
        coords_ref[..., 1] *= scale_y

        # Extract 512x512 high-resolution patches
        patches = self.extract_roi_patches(ref_source, coords_ref)
        l_ids = torch.arange(K, device=device).repeat(B)

        # Generate fine Gaussian heatmaps for each patch
        local_heatmaps = self.local_module(patches, l_ids)  # [B*K, 1, 128, 128]
        local_heatmaps_reshaped = local_heatmaps.view(B, K, 1, self.local_heatmap_size, self.local_heatmap_size)

        # Decode local heatmaps to find refined sub-pixel coordinates
        refined_coords_ref, offsets, local_conf = self.decode_local_heatmaps(
            local_heatmaps_reshaped, coords_ref
        )

        # Scale refined coordinates back to canvas coordinates
        refined_coords_canvas = refined_coords_ref.clone()
        refined_coords_canvas[..., 0] /= scale_x
        refined_coords_canvas[..., 1] /= scale_y

        coarse_conf_t = torch.from_numpy(coarse_conf.astype(np.float32)).to(device)
        final_conf = coarse_conf_t * (0.8 + 0.2 * local_conf)

        return {
            "coarse_heatmaps": coarse_heatmaps,
            "local_heatmaps": local_heatmaps_reshaped,
            "coarse_coords_canvas": coarse_tensor,
            "refined_coords_canvas": refined_coords_canvas,
            "offsets": offsets,
            "coarse_confidence": coarse_conf_t,
            "refined_confidence": final_conf,
        }


def build_two_stage_cephnet(
    checkpoint_path: Optional[str | Path] = "saved_models/best_hrnet_w48_clean.pth",
    num_landmarks: int = 29,
    patch_size: int = 512,
    local_heatmap_size: int = 128,
    device: Optional[torch.device] = None,
) -> TwoStageCephNet:
    """Helper factory to build the TwoStageCephNet model."""
    model = TwoStageCephNet(
        num_landmarks=num_landmarks,
        patch_size=patch_size,
        local_heatmap_size=local_heatmap_size,
    )
    if checkpoint_path is not None:
        model.load_global_weights(checkpoint_path)
    if device is not None:
        model.to(device)
    return model
