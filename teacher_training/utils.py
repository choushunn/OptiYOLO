import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class OutputDirs:
    """输出目录结构"""
    save_dir: Path  # 保存目录
    weights_dir: Path  # 权重目录
    logs_dir: Path  # 日志目录
    plots_dir: Path  # 图表目录
    visuals_dir: Path  # 可视化目录
    shared_dir: Path  # 共享目录

    @property
    def tensorboard_dir(self) -> Path:
        """TensorBoard 日志目录"""
        return self.logs_dir / "tensorboard"


def prepare_output_dirs(base_save: str, experiment_name: str) -> OutputDirs:
    """
    准备输出目录结构
    
    Args:
        base_save: 基础保存目录
        experiment_name: 实验名称
    
    Returns:
        OutputDirs: 输出目录结构
    """
    exp_root = Path(base_save) / experiment_name
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = exp_root / run_name
    weights_dir = save_dir / "weights"
    logs_dir = save_dir / "logs"
    plots_dir = save_dir / "plots"
    visuals_dir = save_dir / "visuals"
    shared_dir = exp_root / "shared"
    for path in [save_dir, weights_dir, logs_dir, plots_dir, visuals_dir, shared_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return OutputDirs(
        save_dir=save_dir,
        weights_dir=weights_dir,
        logs_dir=logs_dir,
        plots_dir=plots_dir,
        visuals_dir=visuals_dir,
        shared_dir=shared_dir,
    )


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置全局随机种子
    
    Args:
        seed: 种子值
        deterministic: 是否使用确定性算法
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """
    为数据加载器的工作进程设置种子
    
    Args:
        worker_id: 工作进程 ID
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
