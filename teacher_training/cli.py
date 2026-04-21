import argparse

from .config import TrainConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的命令行参数解析器
    """
    parser = argparse.ArgumentParser(description="Train OptiYOLO teacher and detector")
    parser.add_argument("--yaml", dest="yaml_path", default=None, help="Path to dataset yaml file")
    parser.add_argument("--img-size", type=int, default=None, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--teacher-epochs", type=int, default=None, help="Teacher training epochs")
    parser.add_argument("--detector-epochs", type=int, default=None, help="Detector training epochs")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluate every N epochs")
    parser.add_argument("--vis-interval", type=int, default=None, help="Visualize every N epochs")
    parser.add_argument("--experiment-name", default=None, help="Experiment directory name")
    parser.add_argument("--base-save", default=None, help="Output root directory")
    parser.add_argument("--teacher-weight-path", default=None, help="Teacher checkpoint path")
    parser.add_argument("--num-workers", type=int, default=None, help="PyTorch dataloader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", dest="device_override", default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--non-deterministic", action="store_true", help="Disable deterministic mode")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn benchmark when non-deterministic")
    parser.add_argument("--anchor-match-ratio-thresh", type=float, default=None, help="Anchor ratio matching threshold")
    parser.add_argument("--disable-neighbor-cells", action="store_true", help="Disable neighbor cell assignment")
    parser.add_argument("--enable-detector-rollback", action="store_true", help="Enable detector rollback on plateau")
    parser.add_argument("--allow-metric-fallback", action="store_true", help="Allow metric fallback when backend fails")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Warmup epochs for detector training")
    parser.add_argument("--warmup-start-factor", type=float, default=None, help="Initial lr ratio for warmup")
    parser.add_argument("--accumulate", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--ema-decay", type=float, default=None, help="EMA decay")
    parser.add_argument("--ema-tau", type=float, default=None, help="EMA warmup tau")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--disable-ema", action="store_true", help="Disable EMA for detector")
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard writer")
    return parser


def config_from_args() -> TrainConfig:
    """
    从命令行参数创建训练配置对象
    
    Returns:
        TrainConfig: 配置好的训练配置对象
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = TrainConfig()
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    if args.disable_tensorboard:
        cfg.enable_tensorboard = False
    if args.non_deterministic:
        cfg.deterministic = False
    if args.cudnn_benchmark:
        cfg.cudnn_benchmark = True
    if args.disable_neighbor_cells:
        cfg.assign_neighbor_cells = False
    if args.enable_detector_rollback:
        cfg.enable_detector_rollback = True
    if args.allow_metric_fallback:
        cfg.allow_metric_fallback = True
    if args.disable_amp:
        cfg.amp = False
    if args.disable_ema:
        cfg.use_ema = False
    cfg.validate()
    return cfg
