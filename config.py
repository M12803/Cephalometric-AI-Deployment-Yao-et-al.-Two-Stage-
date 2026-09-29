"""
Configuration for the Two-Stage Landmark Detection Model.

This module defines all hyperparameters and settings for training
and evaluation.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Data-related configuration."""
    # Paths
    data_root: str = "training_data"
    log_dir: str = "logs"
    save_dir: str = "saved_models"
    pretrained_backbone: Optional[str] = None
    
    # Data processing
    input_size: int = 1024  # Original image size
    global_input_size: int = 512  # Size for global module
    local_patch_size: int = 64  # Patch size for local refinement
    stride: int = 4  # Heatmap stride
    sigma: float = 2.5  # Gaussian sigma for heatmaps
    target_mm_per_px: float = 0.3  # Target physical scale
    channels: int = 3  # Number of input channels
    num_landmarks: int = 29  # Number of landmarks
    
    # Augmentation
    rot_deg: float = 10.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    trans_frac: float = 0.05
    use_clahe: bool = True
    
    # DataLoader
    batch_size: int = 8
    val_batch_size: int = 16
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model-related configuration."""
    # Architecture
    num_landmarks: int = 29
    in_channels: int = 3
    backbone_channels: int = 48
    global_input_size: int = 512
    local_patch_size: int = 64
    local_feature_channels: int = 32
    
    # Pretraining
    pretrained: bool = False
    freeze_backbone_epochs: int = 0
    
    # DARK decoding
    dark_stride: int = 4
    dark_blur_kernel: int = 11


@dataclass
class OptimConfig:
    """Optimization configuration."""
    # Training
    epochs: int = 100
    lr_backbone: float = 1e-4
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9998
    
    # AMP
    use_amp: bool = True
    
    # Gradient accumulation
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0


@dataclass
class TrainConfig:
    """Training configuration."""
    # Seed
    seed: int = 42
    
    # Evaluation
    eval_every: int = 1
    save_every: int = 5
    
    # Early stopping
    early_stop_patience: int = 15
    
    # SDR thresholds
    sdr_thresholds: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 4.0])
    
    # TTA
    tta_flip: bool = False
    flip_pairs: List[Tuple[int, int]] = field(default_factory=list)
    
    # Loss weights
    heatmap_weight: float = 1.0
    coordinate_weight: float = 1.0
    offset_weight: float = 0.5
    peak_weight: float = 50.0


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # Paths (for compatibility)
    paths: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        """Initialize paths reference."""
        self.paths = self.data
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        if 'data' in data:
            config.data = DataConfig(**data['data'])
            config.paths = config.data
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'optim' in data:
            config.optim = OptimConfig(**data['optim'])
        if 'train' in data:
            config.train = TrainConfig(**data['train'])
        
        return config


def get_config() -> Config:
    """Get default configuration."""
    return Config()
