"""
Loss functions for the Two-Stage Landmark Detection Model.

Includes:
- Masked heatmap MSE for global module training
- Coordinate-based losses for local refinement
- Combined loss for end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_heatmap_mse(pred, target, mask):
    """
    Masked MSE loss for heatmap prediction.
    
    Args:
        pred: Predicted heatmaps [B, K, H, W]
        target: Target heatmaps [B, K, H, W]
        mask: Validity mask [B, K]
    
    Returns:
        loss: Scalar loss value
    """
    se = ((pred - target) ** 2) * mask[:, :, None, None]
    denom = mask.sum().clamp(min=1.0) * pred.shape[-1] * pred.shape[-2]
    return se.sum() / denom


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing (AWing) Loss for High-Precision Heatmap Landmark Localization.
    Reference: Wang et al., 'Adaptive Wing Analysis for Robust Facial Landmark Detection in the Wild', ICCV 2019.
    
    Adapted specifically for Cephalometric Gaussian Heatmaps (Yao et al. 2022).
    Penalizes slight errors around the peak sharply while attenuating background gradient noise.
    """
    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0, alpha: float = 2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = (pred - target).abs()
        p = self.alpha - target
        
        # When delta < theta
        term = (delta / self.epsilon) ** p
        loss_small = self.omega * torch.log(1.0 + term)
        
        # Smooth continuous transition for delta >= theta
        A = self.omega * (1.0 / (1.0 + (self.theta / self.epsilon) ** p)) * p * ((self.theta / self.epsilon) ** (p - 1.0)) * (1.0 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1.0 + (self.theta / self.epsilon) ** p)
        loss_large = A * delta - C
        
        loss = torch.where(delta < self.theta, loss_small, loss_large)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, :, None, None]
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0) / (pred.shape[-2] * pred.shape[-1])
        return loss.mean()


class HeatmapAWingMSELoss(nn.Module):
    """
    Combined Adaptive Wing Loss + Weighted Heatmap MSE for Yao et al. architecture.
    """
    def __init__(self, awing_weight: float = 1.0, mse_weight: float = 0.5, peak_weight: float = 50.0):
        super().__init__()
        self.awing = AdaptiveWingLoss()
        self.awing_weight = awing_weight
        self.mse_weight = mse_weight
        self.peak_weight = peak_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss_awing = self.awing(pred, target, mask=mask)
        loss_mse = weighted_masked_heatmap_mse(pred, target, mask=mask, peak_weight=self.peak_weight) if mask is not None else F.mse_loss(pred, target)
        return self.awing_weight * loss_awing + self.mse_weight * loss_mse


def weighted_masked_heatmap_mse(pred, target, mask, peak_weight=50.0):
    """
    Weighted masked MSE loss that emphasizes peak regions.
    
    Args:
        pred: Predicted heatmaps [B, K, H, W]
        target: Target heatmaps [B, K, H, W]
        mask: Validity mask [B, K]
        peak_weight: Weight for peak regions
    
    Returns:
        loss: Scalar loss value
    """
    # Create weight map based on target heatmap
    weight_map = 1.0 + (peak_weight - 1.0) * target
    
    se = ((pred - target) ** 2) * weight_map * mask[:, :, None, None]
    denom = mask.sum().clamp(min=1.0) * pred.shape[-1] * pred.shape[-2]
    return se.sum() / denom


def coordinate_l1_loss(pred_landmarks, target_landmarks, mask):
    """
    L1 loss for landmark coordinates.
    
    Args:
        pred_landmarks: Predicted landmarks [B, K, 2]
        target_landmarks: Target landmarks [B, K, 2]
        mask: Validity mask [B, K]
    
    Returns:
        loss: Scalar loss value
    """
    diff = torch.abs(pred_landmarks - target_landmarks)  # [B, K, 2]
    diff = diff * mask[:, :, None]
    denom = mask.sum().clamp(min=1.0) * 2
    return diff.sum() / denom


def coordinate_l2_loss(pred_landmarks, target_landmarks, mask):
    """
    L2 loss for landmark coordinates.
    
    Args:
        pred_landmarks: Predicted landmarks [B, K, 2]
        target_landmarks: Target landmarks [B, K, 2]
        mask: Validity mask [B, K]
    
    Returns:
        loss: Scalar loss value
    """
    diff = (pred_landmarks - target_landmarks) ** 2  # [B, K, 2]
    diff = diff * mask[:, :, None]
    denom = mask.sum().clamp(min=1.0) * 2
    return diff.sum() / denom


def smooth_l1_loss(pred_landmarks, target_landmarks, mask, beta=1.0):
    """
    Smooth L1 (Huber) loss for landmark coordinates.
    
    Args:
        pred_landmarks: Predicted landmarks [B, K, 2]
        target_landmarks: Target landmarks [B, K, 2]
        mask: Validity mask [B, K]
        beta: Threshold for smooth L1
    
    Returns:
        loss: Scalar loss value
    """
    diff = torch.abs(pred_landmarks - target_landmarks)  # [B, K, 2]
    diff = diff * mask[:, :, None]
    
    # Smooth L1
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
    denom = mask.sum().clamp(min=1.0) * 2
    return loss.sum() / denom


def offset_loss(pred_offsets, target_offsets, mask):
    """
    Loss for predicted offsets in local refinement.
    
    Args:
        pred_offsets: Predicted offsets [B, K, 2]
        target_offsets: Target offsets [B, K, 2]
        mask: Validity mask [B, K]
    
    Returns:
        loss: Scalar loss value
    """
    return smooth_l1_loss(pred_offsets, target_offsets, mask, beta=1.0)


class TwoStageLoss(nn.Module):
    """
    Combined loss for the Two-Stage Landmark Model.
    
    Combines:
    - Global heatmap loss (for coarse localization)
    - Local coordinate loss (for fine refinement)
    - Offset loss (for offset prediction)
    """
    
    def __init__(
        self,
        heatmap_weight=1.0,
        coordinate_weight=1.0,
        offset_weight=0.5,
        peak_weight=50.0,
    ):
        super(TwoStageLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.coordinate_weight = coordinate_weight
        self.offset_weight = offset_weight
        self.peak_weight = peak_weight
    
    def forward(self, predictions, targets, mask):
        """
        Compute combined loss.
        
        Args:
            predictions: Dict containing model outputs
                - coarse_heatmaps: [B, K, H, W]
                - refined_landmarks: [B, K, 2]
                - coarse_landmarks: [B, K, 2]
                - offsets: [B, K, 2]
            targets: Dict containing target values
                - heatmaps: [B, K, H, W]
                - landmarks: [B, K, 2]
            mask: Validity mask [B, K]
        
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dict containing individual loss components
        """
        # Global heatmap loss
        # Resize target heatmaps if needed to match prediction size
        pred_heatmaps = predictions['coarse_heatmaps']
        target_heatmaps = targets['heatmaps']
        if pred_heatmaps.shape[-2:] != target_heatmaps.shape[-2:]:
            target_heatmaps = torch.nn.functional.interpolate(
                target_heatmaps,
                size=pred_heatmaps.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        heatmap_loss = weighted_masked_heatmap_mse(
            pred_heatmaps,
            target_heatmaps,
            mask,
            peak_weight=self.peak_weight,
        )
        
        # Local coordinate loss (refined landmarks)
        coord_loss = smooth_l1_loss(
            predictions['refined_landmarks'],
            targets['landmarks'],
            mask,
        )
        
        # Coarse coordinate loss (auxiliary)
        coarse_coord_loss = smooth_l1_loss(
            predictions['coarse_landmarks'],
            targets['landmarks'],
            mask,
        )
        
        # Offset loss
        target_offsets = targets['landmarks'] - predictions['coarse_landmarks']
        off_loss = offset_loss(
            predictions['offsets'],
            target_offsets,
            mask,
        )
        
        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.coordinate_weight * coord_loss +
            0.1 * coarse_coord_loss +  # Auxiliary loss
            self.offset_weight * off_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'heatmap': heatmap_loss.item(),
            'coordinate': coord_loss.item(),
            'coarse_coordinate': coarse_coord_loss.item(),
            'offset': off_loss.item(),
        }
        
        return total_loss, loss_dict


class RefinementOnlyLoss(nn.Module):
    """
    Loss for training only the local refinement module.
    
    Used when the global module is pre-trained and frozen.
    """
    
    def __init__(self, coordinate_weight=1.0, offset_weight=0.5):
        super(RefinementOnlyLoss, self).__init__()
        self.coordinate_weight = coordinate_weight
        self.offset_weight = offset_weight
    
    def forward(self, predictions, targets, mask):
        """
        Compute refinement loss.
        
        Args:
            predictions: Dict containing model outputs
            targets: Dict containing target values
            mask: Validity mask [B, K]
        
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dict containing individual loss components
        """
        # Coordinate loss
        coord_loss = smooth_l1_loss(
            predictions['refined_landmarks'],
            targets['landmarks'],
            mask,
        )
        
        # Offset loss
        target_offsets = targets['landmarks'] - predictions['coarse_landmarks']
        off_loss = offset_loss(
            predictions['offsets'],
            target_offsets,
            mask,
        )
        
        total_loss = (
            self.coordinate_weight * coord_loss +
            self.offset_weight * off_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'coordinate': coord_loss.item(),
            'offset': off_loss.item(),
        }
        
        return total_loss, loss_dict
