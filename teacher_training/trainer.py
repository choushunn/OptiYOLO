import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig, enable_cudnn_optimizations
from .data import (
    MilitaryDataset,
    TeacherDataset,
    collate_fn,
    load_class_names,
    resolve_dataset_dirs,
    teacher_collate,
)
from .losses import YOLOLoss, build_targets_multi_scale
from .metrics import evaluate_model
from .models import ConvTeacher, YOLOLightHead
from .utils import OutputDirs, prepare_output_dirs, seed_worker, set_global_seed
from .visualization import visualize_with_opencv

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


@dataclass
class DetectorTrainState:
    """检测器训练状态"""
    best_map: float = 0.0  # 最佳 mAP@0.5:0.95
    best_f1: float = 0.0  # 最佳 F1 分数
    det_stop_cnt: int = 0  # 检测器早停计数
    best_det_state: Optional[dict] = None  # 最佳检测器状态
    best_train_state: Optional[dict] = None  # 最佳训练态检测器状态
    best_opt_state: Optional[dict] = None  # 最佳优化器状态
    best_sch_state: Optional[dict] = None  # 最佳调度器状态
    last_map: float = 0.0  # 上次 mAP@0.5
    last_map_5095: float = 0.0  # 上次 mAP@0.5:0.95
    last_precision: float = 0.0  # 上次精确率
    last_recall: float = 0.0  # 上次召回率
    last_f1: float = 0.0  # 上次 F1 分数


class ModelEMA:
    """模型 EMA (Exponential Moving Average)"""
    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: float = 2000.0):
        """
        初始化模型 EMA
        
        Args:
            model: 原始模型
            decay: EMA 衰减系数
            tau: 控制衰减速度的参数
        """
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _decay(self) -> float:
        """
        计算动态衰减系数
        
        Returns:
            float: 衰减系数
        """
        # 早期更快跟随，后期逐步接近稳定衰减
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        更新 EMA 模型
        
        Args:
            model: 原始模型
        """
        self.updates += 1
        d = self._decay()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
                continue
            v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)


def _apply_warmup_lr(
    optimizer,
    step_idx: int,
    warmup_steps: int,
    target_lr: float,
    start_factor: float,
) -> float:
    """
    应用学习率预热
    
    Args:
        optimizer: 优化器
        step_idx: 当前步骤索引
        warmup_steps: 预热步数
        target_lr: 目标学习率
        start_factor: 起始因子
    
    Returns:
        float: 当前学习率
    """
    if warmup_steps <= 0:
        return optimizer.param_groups[0]["lr"]
    ratio = min((step_idx + 1) / warmup_steps, 1.0)
    lr = target_lr * (start_factor + (1.0 - start_factor) * ratio)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _create_loader(dataset, cfg: TrainConfig, shuffle: bool, collate, seed_offset: int):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        cfg: 训练配置
        shuffle: 是否打乱数据
        collate: 数据整理函数
        seed_offset: 种子偏移量
    
    Returns:
        DataLoader: 数据加载器
    """
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=loader_generator,
    )


def _resolve_teacher_weight_path(cfg: TrainConfig, dirs: OutputDirs) -> Path:
    """
    解析教师权重路径
    
    Args:
        cfg: 训练配置
        dirs: 输出目录
    
    Returns:
        Path: 教师权重路径
    """
    teacher_weight = Path(cfg.teacher_weight_path)
    if not teacher_weight.is_absolute():
        teacher_weight = dirs.shared_dir / teacher_weight
    cfg.teacher_weight_path = str(teacher_weight)
    return teacher_weight


def _create_writer(cfg: TrainConfig, dirs: OutputDirs):
    """
    创建 TensorBoard 写入器
    
    Args:
        cfg: 训练配置
        dirs: 输出目录
    
    Returns:
        Optional[SummaryWriter]: TensorBoard 写入器
    """
    if not cfg.enable_tensorboard:
        return None
    if SummaryWriter is None:
        print("[Warn] 未安装 tensorboardX，跳过 TensorBoard 记录。可执行: pip install tensorboardX")
        return None
    return SummaryWriter(log_dir=str(dirs.tensorboard_dir))


def _train_teacher_if_needed(
    cfg: TrainConfig,
    teacher: ConvTeacher,
    train_img_dir: str,
    train_label_dir: str,
    val_img_dir: str,
    val_label_dir: str,
    writer: Optional["SummaryWriter"] = None,
) -> None:
    """
    训练教师网络（如果需要）
    
    Args:
        cfg: 训练配置
        teacher: 教师网络
        train_img_dir: 训练图像目录
        train_label_dir: 训练标签目录
        val_img_dir: 验证图像目录
        val_label_dir: 验证标签目录
        writer: TensorBoard 写入器
    """
    teacher_weight = Path(cfg.teacher_weight_path)
    if teacher_weight.exists():
        print(f"检测到教师权重，跳过教师训练: {teacher_weight}")
        return

    best_loss = float("inf")
    stop_cnt = 0
    optimizer = optim.Adam(teacher.parameters(), lr=cfg.teacher_lr, weight_decay=cfg.teacher_weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, min_lr=1e-6)

    t_train = TeacherDataset(cfg.root_path, "train", cfg.img_size, img_dir=train_img_dir, label_dir=train_label_dir)
    t_val = TeacherDataset(cfg.root_path, "val", cfg.img_size, img_dir=val_img_dir, label_dir=val_label_dir)
    t_loader_train = _create_loader(t_train, cfg, shuffle=True, collate=teacher_collate, seed_offset=11)
    t_loader_val = _create_loader(t_val, cfg, shuffle=False, collate=teacher_collate, seed_offset=12)

    for ep in range(cfg.teacher_epochs):
        teacher.train()
        train_total = 0.0
        for img, mask in tqdm(t_loader_train, desc=f"Teacher Epoch {ep + 1}/{cfg.teacher_epochs}"):
            img, mask = img.to(cfg.device), mask.to(cfg.device)
            out = teacher(img)
            loss = criterion(out, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        val_loss = 0.0
        teacher.eval()
        with torch.no_grad():
            for img, mask in t_loader_val:
                img, mask = img.to(cfg.device), mask.to(cfg.device)
                val_loss += criterion(teacher(img), mask).item()

        train_loss = train_total / max(len(t_loader_train), 1)
        val_loss = val_loss / max(len(t_loader_val), 1)
        scheduler.step(val_loss)
        print(f"Teacher | Epoch {ep + 1} train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        if writer is not None:
            writer.add_scalar("teacher/train_loss", train_loss, ep + 1)
            writer.add_scalar("teacher/val_loss", val_loss, ep + 1)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(teacher.state_dict(), teacher_weight)
            stop_cnt = 0
        else:
            stop_cnt += 1
            if stop_cnt >= cfg.teacher_patience:
                print(f"教师网络早停，最优loss={best_loss:.4f}")
                break
    print("教师网络训练完成")


def _plot_metrics(csv_path: Path, plots_dir: Path) -> None:
    """
    绘制训练指标曲线
    
    Args:
        csv_path: CSV 文件路径
        plots_dir: 图表保存目录
    """
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]
    loss = df["train_loss"]
    mAP = df["mAP_50"]
    mAP_5095 = df["mAP_50_95"]
    precision = df["precision"]
    recall = df["recall"]
    f1 = df["f1"]

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, color="red", label="训练损失 Loss")
    plt.plot(epochs, mAP, color="blue", label="mAP@0.5")
    plt.plot(epochs, mAP_5095, color="purple", label="mAP@0.5:0.95")
    plt.plot(epochs, precision, color="green", label="精确率 Precision")
    plt.plot(epochs, recall, color="orange", label="召回率 Recall")
    plt.plot(epochs, f1, color="brown", label="F1")
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("数值")
    plt.title("训练全指标曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(str(plots_dir / "all_metrics.png"), dpi=200)
    plt.close("all")


def _append_epoch_log(csv_path: Path, epoch: int, avg_loss: float, lr: float, state: DetectorTrainState) -> None:
    row = {
        "epoch": epoch,
        "train_loss": round(avg_loss, 4),
        "mAP_50": round(state.last_map, 4),
        "mAP_50_95": round(state.last_map_5095, 4),
        "precision": round(state.last_precision, 4),
        "recall": round(state.last_recall, 4),
        "f1": round(state.last_f1, 4),
        "lr": round(lr, 8),
    }
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)


def _compute_aux_losses(
    p3: torch.Tensor,
    p4: torch.Tensor,
    p5: torch.Tensor,
    heatmap: torch.Tensor,
    targets,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算辅助损失
    
    Args:
        p3: P3 特征图
        p4: P4 特征图
        p5: P5 特征图
        heatmap: 热力图
        targets: 目标标签
        device: 设备
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 热力图损失、特征损失、BCE损失
    """
    heat_loss = torch.tensor(0.0, device=device)
    count = 0
    _, _, h_f, w_f = heatmap.shape
    for i, target in enumerate(targets):
        if len(target) == 0:
            continue
        cx = (target[:, 1] * w_f).long().clamp(0, w_f - 1)
        cy = (target[:, 2] * h_f).long().clamp(0, h_f - 1)
        heat_loss += (1.0 - heatmap[i, 0, cy, cx]).mean()
        count += 1
    if count > 0:
        heat_loss /= count

    feat_loss = nn.MSELoss()(p3.mean(1, keepdim=True), F.interpolate(heatmap, size=p3.shape[2:], mode="bilinear", align_corners=False))

    bce_loss = torch.tensor(0.0, device=device)
    bce_with_logits = nn.BCEWithLogitsLoss()
    for pred in [p3, p4, p5]:
        pred_obj = pred[:, 4:5, :, :]
        target_prob = F.interpolate(heatmap, size=pred.shape[2:], mode="bilinear", align_corners=False).clamp(1e-7, 1 - 1e-7)
        bce_loss = bce_loss + bce_with_logits(pred_obj, target_prob)
    return heat_loss, feat_loss, bce_loss


def _evaluate_and_update_state(
    cfg: TrainConfig,
    state: DetectorTrainState,
    epoch: int,
    avg_loss: float,
    detector,
    eval_detector,
    teacher,
    val_loader,
    num_classes: int,
    scheduler,
    optimizer,
    writer: Optional["SummaryWriter"],
    weights_dir: Path,
) -> None:
    """
    评估并更新训练状态
    
    Args:
        cfg: 训练配置
        state: 检测器训练状态
        epoch: 轮次
        avg_loss: 平均损失
        detector: 检测器
        eval_detector: 用于评估的检测器
        teacher: 教师网络
        val_loader: 验证数据加载器
        num_classes: 类别数量
        scheduler: 学习率调度器
        optimizer: 优化器
        writer: TensorBoard 写入器
        weights_dir: 权重保存目录
    """
    metrics, precision, recall = evaluate_model(
        eval_detector,
        teacher,
        val_loader,
        num_classes,
        cfg.device,
        cfg.img_size,
        cfg.strides,
        cfg.anchors,
        iou_threshold=0.5,
        conf_thre=0.3,
        allow_metric_fallback=cfg.allow_metric_fallback,
    )
    state.last_map = metrics["map_50"].item()
    state.last_map_5095 = metrics["map_50_95"].item()
    state.last_precision = precision
    state.last_recall = recall
    state.last_f1 = metrics["f1"].item()

    print(
        f"Epoch {epoch} | loss: {avg_loss:.3f} | "
        f"mAP@0.5:0.95: {state.last_map_5095:.4f} | mAP@0.5: {state.last_map:.4f} | "
        f"P: {state.last_precision:.4f} | R: {state.last_recall:.4f} | F1: {state.last_f1:.4f}"
    )
    if writer is not None:
        writer.add_scalar("eval/mAP_50", state.last_map, epoch)
        writer.add_scalar("eval/mAP_50_95", state.last_map_5095, epoch)
        writer.add_scalar("eval/precision", state.last_precision, epoch)
        writer.add_scalar("eval/recall", state.last_recall, epoch)
        writer.add_scalar("eval/f1", state.last_f1, epoch)

    scheduler.step(state.last_map_5095)
    improved_map = state.last_map_5095 > (state.best_map + cfg.detector_min_delta_map5095)
    improved_f1 = state.last_f1 > (state.best_f1 + cfg.detector_min_delta_f1)

    if improved_map:
        state.best_map = state.last_map_5095
    if improved_f1:
        state.best_f1 = state.last_f1

    if improved_map or improved_f1:
        if improved_map:
            state.best_det_state = copy.deepcopy(eval_detector.state_dict())
            state.best_train_state = copy.deepcopy(detector.state_dict())
            state.best_opt_state = copy.deepcopy(optimizer.state_dict())
            state.best_sch_state = copy.deepcopy(scheduler.state_dict())
            torch.save(state.best_det_state, weights_dir / "best_det.pth")
        state.det_stop_cnt = 0
        print(
            f"指标改进 | best mAP@0.5:0.95={state.best_map:.4f} | "
            f"best F1={state.best_f1:.4f} | save_ckpt={'yes' if improved_map else 'no'}"
        )
    else:
        state.det_stop_cnt += 1
        if cfg.enable_detector_rollback and state.det_stop_cnt >= cfg.detector_patience and state.best_train_state is not None:
            print("回滚至最优权重")
            detector.load_state_dict(state.best_train_state)
            optimizer.load_state_dict(state.best_opt_state)
            scheduler.load_state_dict(state.best_sch_state)
            state.det_stop_cnt = 0


def run_training(cfg: TrainConfig = None) -> None:
    """
    运行训练主函数
    
    Args:
        cfg: 训练配置
    """
    cfg = cfg or TrainConfig()
    cfg.validate()
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    enable_cudnn_optimizations(cfg.deterministic, cfg.cudnn_benchmark)

    class_names, num_classes = load_class_names(cfg.yaml_path)
    train_img_dir, train_label_dir, val_img_dir, val_label_dir = resolve_dataset_dirs(cfg.yaml_path)
    dirs = prepare_output_dirs(cfg.base_save, cfg.experiment_name)
    _resolve_teacher_weight_path(cfg, dirs)
    writer = _create_writer(cfg, dirs)

    print(f"运行目录: {dirs.save_dir}")
    print(f"设备: {cfg.device} | 随机种子: {cfg.seed}")

    print(f"\n[1/2] 开始训练教师网络 最大{cfg.teacher_epochs}轮（自动早停）...")
    teacher = ConvTeacher().to(cfg.device)
    _train_teacher_if_needed(cfg, teacher, train_img_dir, train_label_dir, val_img_dir, val_label_dir, writer=writer)

    print(f"\n[2/2] 加载教师权重，训练检测头{cfg.detector_epochs}轮...")
    teacher.load_state_dict(torch.load(cfg.teacher_weight_path, map_location=cfg.device, weights_only=True))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    detector = YOLOLightHead(in_channels=1, out_channels=3 * (5 + num_classes)).to(cfg.device)
    optimizer = optim.Adam(detector.parameters(), lr=cfg.detector_lr, weight_decay=cfg.detector_weight_decay)
    criterion = YOLOLoss(cfg.strides, cfg.anchors, num_classes, cfg.box_weight, cfg.obj_weight, cfg.cls_weight)
    scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=5, min_lr=1e-6)
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(detector, decay=cfg.ema_decay, tau=cfg.ema_tau) if cfg.use_ema else None

    csv_path = dirs.logs_dir / "results.csv"
    pd.DataFrame(columns=["epoch", "train_loss", "mAP_50", "mAP_50_95", "precision", "recall", "f1", "lr"]).to_csv(csv_path, index=False)

    train_dataset = MilitaryDataset(cfg.root_path, "train", cfg.img_size, img_dir=train_img_dir, label_dir=train_label_dir)
    val_dataset = MilitaryDataset(cfg.root_path, "val", cfg.img_size, img_dir=val_img_dir, label_dir=val_label_dir)
    train_loader = _create_loader(train_dataset, cfg, shuffle=True, collate=collate_fn, seed_offset=21)
    val_loader = _create_loader(val_dataset, cfg, shuffle=False, collate=collate_fn, seed_offset=22)

    state = DetectorTrainState()
    warmup_steps = cfg.warmup_epochs * max(len(train_loader), 1)
    global_step = 0
    for epoch in range(cfg.detector_epochs):
        detector.train()
        total_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"Det Epoch {epoch + 1}/{cfg.detector_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
            imgs, targets = zip(*batch)
            imgs = torch.stack(imgs).to(cfg.device)
            targets = [t.to(cfg.device) for t in targets]
            with torch.no_grad():
                heatmap = teacher(imgs)

            if global_step < warmup_steps:
                _apply_warmup_lr(
                    optimizer=optimizer,
                    step_idx=global_step,
                    warmup_steps=warmup_steps,
                    target_lr=cfg.detector_lr,
                    start_factor=cfg.warmup_start_factor,
                )

            if use_amp:
                autocast_ctx = torch.autocast(device_type="cuda", enabled=True)
            else:
                autocast_ctx = torch.autocast(device_type="cpu", enabled=False)
            with autocast_ctx:
                p3, p4, p5 = detector(heatmap)
                gt3, gt4, gt5 = build_targets_multi_scale(
                    targets_list=targets,
                    anchors_per_scale=cfg.anchors,
                    strides=cfg.strides,
                    num_classes=num_classes,
                    img_size=cfg.img_size,
                    device=cfg.device,
                    anchor_match_ratio_thresh=cfg.anchor_match_ratio_thresh,
                    assign_neighbor_cells=cfg.assign_neighbor_cells,
                )
                l3, _, _, _ = criterion(p3, gt3, 0)
                l4, _, _, _ = criterion(p4, gt4, 1)
                l5, _, _, _ = criterion(p5, gt5, 2)
                base_loss = l3 + l4 + l5
                heat_loss, feat_loss, bce_loss = _compute_aux_losses(p3, p4, p5, heatmap, targets, cfg.device)
                total_loss = (
                    base_loss
                    + cfg.heat_loss_weight * heat_loss
                    + cfg.feat_loss_weight * feat_loss
                    + cfg.bce_loss_weight * bce_loss
                )

            loss_backward = total_loss / max(cfg.accumulate, 1)
            scaler.scale(loss_backward).backward()

            should_step = ((batch_idx + 1) % cfg.accumulate == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(detector.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(detector)

            total_loss_epoch += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())
            global_step += 1

        avg_loss = total_loss_epoch / max(len(train_loader), 1)
        current_lr = optimizer.param_groups[0]["lr"]
        should_eval = (epoch + 1) % cfg.eval_interval == 0 or epoch == 0
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch + 1)
            writer.add_scalar("train/lr", current_lr, epoch + 1)

        if should_eval:
            _evaluate_and_update_state(
                cfg=cfg,
                state=state,
                epoch=epoch + 1,
                avg_loss=avg_loss,
                detector=detector,
                eval_detector=ema.ema if ema is not None else detector,
                teacher=teacher,
                val_loader=val_loader,
                num_classes=num_classes,
                scheduler=scheduler,
                optimizer=optimizer,
                writer=writer,
                weights_dir=dirs.weights_dir,
            )
        else:
            print(f"Epoch {epoch + 1} | loss: {avg_loss:.3f} | (skip eval)")
        _append_epoch_log(csv_path, epoch + 1, avg_loss, current_lr, state)

        if (epoch + 1) % cfg.vis_interval == 0:
            visualize_with_opencv(
                teacher,
                detector,
                val_loader,
                cfg.device,
                class_names,
                cfg.strides,
                cfg.anchors,
                num_classes,
                save_path=str(dirs.visuals_dir / f"vis_ep{epoch + 1}.png"),
            )

    torch.save(detector.state_dict(), dirs.weights_dir / "final_det.pth")
    if ema is not None:
        torch.save(ema.ema.state_dict(), dirs.weights_dir / "ema_det.pth")
    visualize_with_opencv(
        teacher,
        detector,
        val_loader,
        cfg.device,
        class_names,
        cfg.strides,
        cfg.anchors,
        num_classes,
        save_path=str(dirs.visuals_dir / "final_visual.png"),
    )
    _plot_metrics(csv_path, dirs.plots_dir)

    if writer is not None:
        writer.close()
    print(f"\n训练完成！最优mAP@0.5:0.95={state.best_map:.4f} | 最优F1={state.best_f1:.4f}")
    print(f"CSV: {csv_path}")
    print(f"曲线: {dirs.plots_dir / 'all_metrics.png'}")
    print(f"权重目录: {dirs.weights_dir}")
    print(f"可视化目录: {dirs.visuals_dir}")
    if cfg.enable_tensorboard:
        print(f"TensorBoard目录: {dirs.tensorboard_dir}")
