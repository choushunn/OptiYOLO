# OptiYOLO

基于教师热力图引导的轻量化 YOLO 检测训练工程。  
当前代码已按模块拆分，入口脚本为 `train_teacher.py`。

## 目录说明

- `train_teacher.py`: 训练入口
- `teacher_training/config.py`: 训练配置
- `teacher_training/models.py`: 教师网络与检测头
- `teacher_training/data.py`: 数据集与数据增强
- `teacher_training/losses.py`: 检测损失与目标构建
- `teacher_training/metrics.py`: 评估指标（支持无 COCO 依赖回退）
- `teacher_training/visualization.py`: OpenCV 可视化
- `teacher_training/trainer.py`: 主训练流程

## 环境要求

- Python 3.10+（已在 3.12 环境使用）
- PyTorch / TorchVision
- 其他依赖：
  - `numpy`
  - `opencv-python`
  - `pandas`
  - `matplotlib`
  - `tqdm`
  - `pyyaml`
  - `torchmetrics`
  - `tensorboardX`（推荐，用于 TensorBoard）

可选（用于 `torchmetrics` 官方 COCO mAP 实现）：
- `pycocotools` 或 `faster-coco-eval`

## 安装示例

```bash
pip install torch torchvision
pip install numpy opencv-python pandas matplotlib tqdm pyyaml torchmetrics tensorboardX
```

或使用依赖文件：

```bash
pip install -r requirements.txt
```

可选：

```bash
pip install pycocotools
```

## 数据配置

默认从 `yolov5/data/coco128.yaml` 读取类别与数据路径。  
代码会自动解析 `train/val images` 目录并推断对应 `labels` 目录。

请确认你的标注为 YOLO txt 格式：  
`class cx cy w h`（归一化坐标）。

## 快速开始

```bash
# 1) 安装依赖（先安装匹配你环境的 torch/torchvision）
pip install -r requirements.txt

# 2) 直接训练（使用默认配置）
python train_teacher.py
```

## 训练

```bash
python train_teacher.py

# 可选参数示例
python train_teacher.py --yaml yolov5/data/coco128.yaml --batch-size 8 --teacher-epochs 200 --detector-epochs 300
```

## CLI 参数

可通过命令行覆盖 `TrainConfig` 的关键字段：

- `--yaml`: 数据集 yaml 路径
- `--img-size`: 输入尺寸（默认 640）
- `--batch-size`: 批大小
- `--teacher-epochs`: 教师网络训练轮数
- `--detector-epochs`: 检测头训练轮数
- `--eval-interval`: 每 N 轮评估一次
- `--vis-interval`: 每 N 轮导出可视化
- `--experiment-name`: 实验名（影响输出目录）
- `--base-save`: 输出根目录
- `--teacher-weight-path`: 教师权重路径（相对路径会放入 `shared/`）
- `--num-workers`: DataLoader worker 数
- `--seed`: 随机种子
- `--device`: 强制设备（如 `cuda` / `cpu`）
- `--disable-tensorboard`: 关闭 TensorBoard 日志

查看全部参数：

```bash
python train_teacher.py --help
```

## 输出结构

默认输出到 `outputs/teacher_detector/`，每次运行创建独立目录：

```text
outputs/teacher_detector/
  shared/
    teacher_best.pth
  run_YYYYMMDD_HHMMSS/
    weights/
      best_det.pth
      final_det.pth
    logs/
      results.csv
      tensorboard/
    plots/
      all_metrics.png
    visuals/
      vis_ep*.png
      final_visual.png
```

## 指标说明

- 主优化指标：`mAP@0.5:0.95`
- 同时记录：`mAP@0.5`、`Precision`、`Recall`、`F1`
- 早停策略：`mAP@0.5:0.95` 与 `F1` 双指标联合判断，降低单指标抖动影响
- 验证集可视化会在 GT 与预测框之外，额外叠加各尺度最佳匹配锚框（`A@8 / A@16 / A@32`）。

## TensorBoard

安装 `tensorboardX` 后自动记录训练与评估标量。

```bash
tensorboard --logdir outputs/teacher_detector
```

## 常用配置项

可在 `teacher_training/config.py` 中修改：

- `img_size`
- `batch_size`
- `teacher_epochs`
- `detector_epochs`
- `base_save`
- `experiment_name`
- `enable_tensorboard`
- `detector_patience`
- `detector_min_delta_map5095`
- `detector_min_delta_f1`
- `seed`
- `num_workers`
- `teacher_lr` / `detector_lr`
- `heat_loss_weight` / `feat_loss_weight` / `bce_loss_weight`
- `device_override`（可通过 `--device` 覆盖）
- `grad_clip_norm`
- `teacher_weight_decay` / `detector_weight_decay`

## 复现实验建议

- 固定 `seed`，并保持数据集与 yaml 不变。
- 每次实验使用不同 `experiment_name`，避免结果相互覆盖。
- 关注 `logs/results.csv` 与 `plots/all_metrics.png` 的一致性。
- 通过 `shared/teacher_best.pth` 复用教师权重，减少重复训练开销。

## 常见问题

- `ModuleNotFoundError: No module named 'torch'`
  - 先按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 安装 `torch` 和 `torchvision`。
- `torchmetrics` mAP 依赖缺失
  - 可安装 `pycocotools` 或 `faster-coco-eval`，否则会自动回退到内置 mAP 计算。

## 工程化改进（本版本）

- 训练入口支持 CLI 参数，便于脚本化和实验管理。
- 训练配置增加校验逻辑，避免无效参数在训练中途才报错。
- 输出目录、随机种子、训练状态管理已解耦，`trainer.py` 可维护性更高。
- DataLoader 参数（`num_workers`、`pin_memory`）统一由配置控制。

