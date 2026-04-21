# OptiYOLO

基于教师热力图引导的轻量化 YOLO 检测训练工程。
当前训练入口为 `train_teacher.py`。

## 1. 项目特点

- 双阶段训练流程：先训练教师网络，再训练检测头
- 检测头采用 anchor-based 多尺度输出结构
- 支持 AMP、EMA、Warmup、梯度累计与 Mixup
- 自动输出权重、日志、指标曲线和可视化结果


## 2. 目录结构

- `train_teacher.py`：训练入口
- `teacher_training/config.py`：训练参数定义与校验
- `teacher_training/cli.py`：命令行参数解析
- `teacher_training/models.py`：`ConvTeacher` 与 `YOLOLightHead`
- `teacher_training/data.py`：数据集、增强、letterbox、collate
- `teacher_training/losses.py`：YOLO 损失与目标分配
- `teacher_training/metrics.py`：预测解码与评估指标
- `teacher_training/trainer.py`：完整训练主流程
- `teacher_training/visualization.py`：验证阶段可视化

---

## 3. 环境安装

### 3.1 基础依赖

- Python 3.10+，已在 3.12 环境验证
- PyTorch / TorchVision，请按 CUDA 版本安装
- `tensorboardX`，用于训练日志可视化

### 3.2 安装方式

```bash
pip install -r requirements.txt
```

---

## 4. 数据集格式

项目默认读取 `--yaml` 指定的 YOLO 数据集配置，默认路径为 `yolov5/data/coco128.yaml`。

- 标注格式：`class cx cy w h`
- 坐标要求：归一化到 `[0, 1]`
- 图片与标签目录会根据 yaml 自动解析

常见目录形态：

```text
dataset_root/
  train/
    images/*.jpg
    labels/*.txt
  val/
    images/*.jpg
    labels/*.txt
```

---

## 5. 5 分钟快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 开始训练，使用默认配置
python train_teacher.py
```

查看所有参数：

```bash
python train_teacher.py --help
```

---

## 6. 训练流程

1. 解析配置并创建输出目录  
2. 训练教师网络。如果 `teacher_weight_path` 已存在，则跳过此阶段。  
3. 冻结教师网络，训练检测头  
4. 按 `eval_interval` 评估并保存最佳权重  
5. 按 `vis_interval` 导出可视化  
6. 训练结束保存 `final_det.pth`、曲线和日志

---

## 7. 常用训练命令

### 7.1 默认训练

```bash
python train_teacher.py
```

### 7.2 提升稳定性

```bash
python train_teacher.py \
  --yaml yolov5/data/coco128.yaml \
  --batch-size 8 \
  --detector-epochs 300 \
  --warmup-epochs 3 \
  --accumulate 2
```

### 7.3 快速调参

```bash
python train_teacher.py \
  --mixup-prob 0.2 \
  --cls-label-smoothing 0.02 \
  --eval-conf-thre 0.001 \
  --vis-conf-thre 0.25
```

---

## 8. 关键参数说明

- `--detector-lr`：检测头学习率，过小易欠拟合，过大易震荡
- `--warmup-epochs` / `--warmup-start-factor`：前期学习率预热
- `--accumulate`：梯度累计，等效增大 batch
- `--disable-amp`：关闭混合精度，默认开启
- `--disable-ema` / `--ema-decay` / `--ema-tau`：EMA 相关
- `--mixup-prob` / `--mixup-alpha`：Mixup 强度
- `--anchor-match-ratio-thresh`：目标与锚框匹配松紧度
- `--disable-neighbor-cells`：关闭邻格分配，匹配规则更严格
- `--eval-conf-thre`：验证解码阈值，建议使用较低取值以减少漏检
- `--vis-conf-thre`：可视化解码阈值，建议使用中等取值

---

## 9. 输出文件说明

默认输出目录：`outputs/teacher_detector/`

```text
outputs/teacher_detector/
  shared/
    teacher_best.pth
  run_YYYYMMDD_HHMMSS/
    weights/
      best_det.pth
      final_det.pth
      ema_det.pth
    logs/
      results.csv
      tensorboard/
    plots/
      all_metrics.png
    visuals/
      vis_ep*.png
      final_visual.png
```

重点查看：

- `logs/results.csv`：每轮损失与指标
- `plots/all_metrics.png`：趋势是否收敛
- `weights/best_det.pth`：主评估指标最优权重

---

## 10. 指标解读建议

- 主指标：`mAP@0.5:0.95`
- 辅助指标：`mAP@0.5`、`Precision`、`Recall`、`F1`
- 若 `mAP@0.5` 高而 `mAP@0.5:0.95` 低：定位精度不足
- 若 `Precision` 高而 `Recall` 低：漏检偏多。可尝试降低阈值并增强数据。

---
