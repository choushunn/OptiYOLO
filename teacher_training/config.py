from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class TrainConfig:
    # Core training setup
    img_size: int = 640
    batch_size: int = 4
    teacher_epochs: int = 400
    detector_epochs: int = 400
    seed: int = 42
    num_workers: int = 0
    deterministic: bool = True
    cudnn_benchmark: bool = False

    # Detector losses
    box_weight: float = 5.0
    obj_weight: float = 1.2
    cls_weight: float = 1.8
    cls_label_smoothing: float = 0.02
    heat_loss_weight: float = 0.7
    feat_loss_weight: float = 0.4
    bce_loss_weight: float = 0.05

    # Train controls
    teacher_patience: int = 15
    detector_patience: int = 20
    detector_min_delta_map5095: float = 1e-4
    detector_min_delta_f1: float = 1e-4
    eval_interval: int = 5
    vis_interval: int = 5
    grad_clip_norm: float = 1.0
    anchor_match_ratio_thresh: float = 4.0
    assign_neighbor_cells: bool = True
    enable_detector_rollback: bool = False
    allow_metric_fallback: bool = False
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.1
    accumulate: int = 1
    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_tau: float = 2000.0
    eval_conf_thre: float = 0.001
    vis_conf_thre: float = 0.25
    mixup_prob: float = 0.15
    mixup_alpha: float = 8.0

    # Optimizer setup
    teacher_lr: float = 8e-5
    teacher_weight_decay: float = 1e-5
    detector_lr: float = 2e-4
    detector_weight_decay: float = 5e-5

    # YOLO head setup
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    anchors: List[List[List[int]]] = field(
        default_factory=lambda: [
            [[38, 25], [97, 53], [117, 145]],
            [[226, 78], [311, 150], [515, 172]],
            [[252, 430], [540, 297], [539, 521]],
        ]
    )

    # Paths and logging
    yaml_path: str = "yolov5/data/coco128.yaml"
    root_path: str = ""
    teacher_weight_path: str = "teacher_best.pth"
    base_save: str = "outputs"
    experiment_name: str = "teacher_detector"
    enable_tensorboard: bool = True
    device_override: Optional[str] = None

    @property
    def device(self) -> str:
        """
        获取训练设备
        
        Returns:
            str: 设备名称 (cuda 或 cpu)
        """
        if self.device_override:
            return self.device_override
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def pin_memory(self) -> bool:
        """
        是否使用 pin_memory
        
        Returns:
            bool: 如果设备是 cuda 则返回 True，否则返回 False
        """
        return self.device.startswith("cuda")

    def validate(self) -> None:
        """
        验证配置参数的有效性
        
        Raises:
            ValueError: 如果配置参数无效
        """
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.teacher_epochs <= 0 or self.detector_epochs <= 0:
            raise ValueError("teacher_epochs and detector_epochs must be positive")
        if self.eval_interval <= 0 or self.vis_interval <= 0:
            raise ValueError("eval_interval and vis_interval must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if not (0.0 <= self.warmup_start_factor <= 1.0):
            raise ValueError("warmup_start_factor must be in [0, 1]")
        if self.accumulate <= 0:
            raise ValueError("accumulate must be positive")
        if not (0.0 <= self.cls_label_smoothing < 1.0):
            raise ValueError("cls_label_smoothing must be in [0, 1)")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        if self.ema_tau <= 0:
            raise ValueError("ema_tau must be positive")
        if not (0.0 <= self.eval_conf_thre <= 1.0):
            raise ValueError("eval_conf_thre must be in [0, 1]")
        if not (0.0 <= self.vis_conf_thre <= 1.0):
            raise ValueError("vis_conf_thre must be in [0, 1]")
        if not (0.0 <= self.mixup_prob <= 1.0):
            raise ValueError("mixup_prob must be in [0, 1]")
        if self.mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be positive")
        if self.anchor_match_ratio_thresh <= 1.0:
            raise ValueError("anchor_match_ratio_thresh must be > 1.0")
        if self.cudnn_benchmark and self.deterministic:
            raise ValueError("cudnn_benchmark must be False when deterministic is True")
        if len(self.strides) != 3 or len(self.anchors) != 3:
            raise ValueError("strides and anchors must have 3 detection scales")


def enable_cudnn_optimizations(deterministic: bool, cudnn_benchmark: bool) -> None:
    """
    启用 CUDA 相关的优化
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = (not deterministic) and cudnn_benchmark
