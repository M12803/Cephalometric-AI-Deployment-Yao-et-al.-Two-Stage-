"""
models/hrnet.py
============================================================

High-Resolution Network (HRNet-W48) for Cephalometric Landmark Detection.

Architecture:
-------------
1. Stem:
   - conv1 (stride 2) -> bn1 -> relu -> conv2 (stride 2) -> bn2 -> relu (total 4x downsample)
2. Stage 1:
   - 4 Bottleneck blocks (64 -> 256 channels)
3. Transitions & Multi-branch Stages:
   - Stage 2: 2 parallel branches [48, 96] channels, 1 HighResolutionModule
   - Stage 3: 3 parallel branches [48, 96, 192] channels, 4 HighResolutionModules
   - Stage 4: 4 parallel branches [48, 96, 192, 384] channels, 3 HighResolutionModules
4. Multi-Scale High-Resolution Representation Fusion:
   - Bilinear upsampling of branches 1, 2, 3 to resolution of branch 0 (H/4, W/4)
   - Channel concatenation: 48 + 96 + 192 + 384 = 720 channels
5. Landmark Heatmap Head:
   - Conv2d(720, 720, 1) -> BatchNorm2d(720) -> ReLU
   - Conv2d(720, num_landmarks, 1) -> [B, num_landmarks, H/4, W/4]

References:
    Wang et al., "Deep High-Resolution Representation Learning for Visual Recognition", TPAMI 2020.
    Sun et al., "Deep High-Resolution Representation Learning for Human Pose Estimation", CVPR 2019.
============================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Basic Building Blocks
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


# ============================================================
# High-Resolution Module
# ============================================================

class HighResolutionModule(nn.Module):
    """
    Parallel multi-resolution branch module with cross-resolution fuse layers.
    """

    def __init__(
        self,
        num_branches: int,
        blocks: type,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers(
            num_branches, num_inchannels, num_channels, multi_scale_output
        )
        self.relu = nn.ReLU(inplace=True)

    def _make_branches(
        self,
        num_branches: int,
        block: type,
        num_blocks: List[int],
        num_channels: List[int],
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )
        return nn.ModuleList(branches)

    def _make_one_branch(
        self,
        branch_index: int,
        block: type,
        num_blocks: List[int],
        num_channels: List[int],
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            block(num_channels[branch_index], num_channels[branch_index], stride)
        )
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(num_channels[branch_index], num_channels[branch_index])
            )
        return nn.Sequential(*layers)

    def _make_fuse_layers(
        self,
        num_branches: int,
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output: bool,
    ) -> nn.ModuleList:
        fuse_layers = []
        num_out = num_branches if multi_scale_output else 1
        for i in range(num_out):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(
                                scale_factor=2 ** (j - i), mode="nearest"
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_channels[i]),
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_inchannels[j]),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


# ============================================================
# HRNet-W48 Architecture
# ============================================================

class HRNetW48(nn.Module):
    """
    Standard HRNet-W48 model adapted for Cephalometric Landmark Detection.

    Args:
        num_landmarks: Number of output landmarks (canonical 29).
        in_channels: Input image channels (3 for RGB/replicated grayscale, 1 for single channel).
    """

    def __init__(
        self,
        num_landmarks: int = 29,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.in_channels = in_channels

        # Stem (downsampling 4x)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Stage 2 Channels for W48: [48, 96]
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, 96, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                )
            ),
        ])
        self.stage2 = nn.Sequential(
            HighResolutionModule(2, BasicBlock, [4, 4], [48, 96], [48, 96])
        )

        # Stage 3 Channels for W48: [48, 96, 192]
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                )
            ),
        ])
        self.stage3 = nn.Sequential(
            HighResolutionModule(3, BasicBlock, [4, 4, 4], [48, 96, 192], [48, 96, 192]),
            HighResolutionModule(3, BasicBlock, [4, 4, 4], [48, 96, 192], [48, 96, 192]),
            HighResolutionModule(3, BasicBlock, [4, 4, 4], [48, 96, 192], [48, 96, 192]),
            HighResolutionModule(3, BasicBlock, [4, 4, 4], [48, 96, 192], [48, 96, 192]),
        )

        # Stage 4 Channels for W48: [48, 96, 192, 384]
        self.transition3 = nn.ModuleList([
            None,
            None,
            None,
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=True),
                )
            ),
        ])
        self.stage4 = nn.Sequential(
            HighResolutionModule(4, BasicBlock, [4, 4, 4, 4], [48, 96, 192, 384], [48, 96, 192, 384]),
            HighResolutionModule(4, BasicBlock, [4, 4, 4, 4], [48, 96, 192, 384], [48, 96, 192, 384]),
            HighResolutionModule(4, BasicBlock, [4, 4, 4, 4], [48, 96, 192, 384], [48, 96, 192, 384]),
        )

        # Head: Multi-scale fusion (48 + 96 + 192 + 384 = 720 channels) -> heatmaps
        self.head = nn.Sequential(
            nn.Conv2d(720, 720, kernel_size=1, bias=False),
            nn.BatchNorm2d(720),
            nn.ReLU(inplace=True),
            nn.Conv2d(720, num_landmarks, kernel_size=1, bias=True),
        )

    def _make_layer(
        self,
        block: type,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            heatmaps: Predicted landmark heatmaps [B, num_landmarks, H/4, W/4]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Stage 2
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        y_list = self.stage2[0](x_list)

        # Stage 3
        x_list = [y_list[0], y_list[1], self.transition2[2](y_list[1])]
        for m in self.stage3:
            x_list = m(x_list)
        y_list = x_list

        # Stage 4
        x_list = [y_list[0], y_list[1], y_list[2], self.transition3[3](y_list[2])]
        for m in self.stage4:
            x_list = m(x_list)
        y_list = x_list

        # Multi-scale high-resolution representation aggregation
        h, w = y_list[0].shape[2:]
        y1 = F.interpolate(y_list[1], size=(h, w), mode="bilinear", align_corners=True)
        y2 = F.interpolate(y_list[2], size=(h, w), mode="bilinear", align_corners=True)
        y3 = F.interpolate(y_list[3], size=(h, w), mode="bilinear", align_corners=True)
        feats = torch.cat([y_list[0], y1, y2, y3], dim=1)

        out = self.head(feats)
        return out


def build_hrnet_w48(
    num_landmarks: int = 29,
    in_channels: int = 3,
    pretrained_path: Optional[str | Path] = None,
) -> HRNetW48:
    """Factory helper to build HRNet-W48 with optional pretrained weight loading."""
    model = HRNetW48(num_landmarks=num_landmarks, in_channels=in_channels)
    if pretrained_path:
        ckpt_path = Path(pretrained_path)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            clean_sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
            model.load_state_dict(clean_sd, strict=False)
    return model


if __name__ == "__main__":
    device = torch.device("cpu")
    model = HRNetW48(num_landmarks=29, in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"HRNet-W48 initialized successfully. Total parameters: {total_params:.2f} M")

    dummy_input = torch.randn(1, 3, 800, 800, device=device)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output heatmaps shape: {out.shape}")
    assert out.shape == (1, 29, 200, 200), f"Wrong shape: {out.shape}"
    print("Smoke test: PASS")
