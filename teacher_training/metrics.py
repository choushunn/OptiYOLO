from typing import List, Optional, Tuple

import torch
import torchvision
from torchmetrics.detection import MeanAveragePrecision


def _safe_create_map_metric(
    device: str,
    allow_fallback: bool,
) -> Optional[MeanAveragePrecision]:
    """
    安全创建 mAP 指标
    
    Args:
        device: 设备
    
    Returns:
        Optional[MeanAveragePrecision]: mAP 指标实例
    """
    try:
        # 与 YOLOv5 常见评估口径对齐：IoU 0.5:0.95（step=0.05）
        coco_iou_thresholds = [round(x, 2) for x in torch.arange(0.5, 0.96, 0.05).tolist()]
        metric = MeanAveragePrecision(
            class_metrics=True,
            iou_thresholds=coco_iou_thresholds,
            max_detection_thresholds=[100, 200, 300],
            backend="faster_coco_eval",
        ).to(device)
        metric.warn_on_many_detections = False
        return metric
    except Exception as exc:
        if allow_fallback:
            print(f"[Warn] torchmetrics mAP backend 不可用，自动回退到内置 mAP@0.5 计算。详情: {exc}")
            return None
        raise RuntimeError(f"torchmetrics mAP backend 初始化失败: {exc}") from exc


def _compute_ap(scores: List[float], tp_flags: List[int], gt_count: int) -> float:
    """
    计算 AP (Average Precision)
    
    Args:
        scores: 预测分数列表
        tp_flags: 真阳性标记列表
        gt_count: 真实目标数量
    
    Returns:
        float: AP 值
    """
    if gt_count <= 0:
        return 0.0
    if len(scores) == 0:
        return 0.0

    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    tp_tensor = torch.tensor(tp_flags, dtype=torch.float32)
    sort_idx = torch.argsort(scores_tensor, descending=True)
    tp_sorted = tp_tensor[sort_idx]
    fp_sorted = 1.0 - tp_sorted

    tp_cum = torch.cumsum(tp_sorted, dim=0)
    fp_cum = torch.cumsum(fp_sorted, dim=0)
    recalls = tp_cum / (gt_count + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([1.0]), precisions, torch.tensor([0.0])])
    precisions = torch.flip(torch.cummax(torch.flip(precisions, dims=[0]), dim=0).values, dims=[0])
    ap = torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:]).item()
    return float(ap)


def _fallback_map(
    preds_per_image: List[torch.Tensor],
    targets_per_image: List[dict],
    num_classes: int,
    device: str,
):
    """
    当 torchmetrics 不可用时的 mAP 计算回退方案
    
    Args:
        preds_per_image: 每个图像的预测结果
        targets_per_image: 每个图像的目标标签
        num_classes: 类别数量
        device: 设备
    
    Returns:
        dict: mAP 指标
    """
    iou_thresholds = [round(x, 2) for x in torch.arange(0.5, 0.96, 0.05).tolist()]
    ap_scores = [[[] for _ in range(num_classes)] for _ in iou_thresholds]
    ap_tp_flags = [[[] for _ in range(num_classes)] for _ in iou_thresholds]
    gt_count_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)

    for img_idx in range(len(preds_per_image)):
        pred = preds_per_image[img_idx]
        gt_boxes = targets_per_image[img_idx]["boxes"]
        gt_labels = targets_per_image[img_idx]["labels"]

        for cls_id in range(num_classes):
            pred_mask = pred[:, 5].long() == cls_id if len(pred) > 0 else torch.zeros(0, dtype=torch.bool, device=device)
            gt_mask = gt_labels == cls_id if len(gt_labels) > 0 else torch.zeros(0, dtype=torch.bool, device=device)
            pred_cls = pred[pred_mask] if len(pred) > 0 else torch.empty((0, 6), device=device)
            gt_cls = gt_boxes[gt_mask] if len(gt_boxes) > 0 else torch.empty((0, 4), device=device)

            gt_count_per_class[cls_id] += len(gt_cls)
            if len(pred_cls) == 0:
                continue

            scores = pred_cls[:, 4]
            boxes = pred_cls[:, :4]
            sort_idx = torch.argsort(scores, descending=True)
            scores = scores[sort_idx]
            boxes = boxes[sort_idx]

            if len(gt_cls) == 0:
                for thr_idx in range(len(iou_thresholds)):
                    ap_scores[thr_idx][cls_id].extend([float(s.item()) for s in scores])
                    ap_tp_flags[thr_idx][cls_id].extend([0] * len(scores))
                continue

            ious = torchvision.ops.box_iou(boxes, gt_cls)
            for thr_idx, thr in enumerate(iou_thresholds):
                matched_gt = torch.zeros(len(gt_cls), dtype=torch.bool, device=device)
                for pred_idx in range(len(boxes)):
                    pred_iou = ious[pred_idx]
                    max_iou, max_gt_idx = pred_iou.max(dim=0)
                    is_tp = 0
                    if max_iou >= thr and not matched_gt[max_gt_idx]:
                        matched_gt[max_gt_idx] = True
                        is_tp = 1
                    ap_scores[thr_idx][cls_id].append(float(scores[pred_idx].item()))
                    ap_tp_flags[thr_idx][cls_id].append(is_tp)

    per_thr_maps = []
    for thr_idx in range(len(iou_thresholds)):
        ap_list = []
        for cls_id in range(num_classes):
            cls_gt = int(gt_count_per_class[cls_id].item())
            if cls_gt <= 0:
                continue
            ap_list.append(_compute_ap(ap_scores[thr_idx][cls_id], ap_tp_flags[thr_idx][cls_id], cls_gt))
        per_thr_maps.append(float(sum(ap_list) / len(ap_list)) if ap_list else 0.0)

    map50 = per_thr_maps[0] if per_thr_maps else 0.0
    map50_95 = float(sum(per_thr_maps) / len(per_thr_maps)) if per_thr_maps else 0.0
    return {
        "map_50": torch.tensor(map50, dtype=torch.float32, device=device),
        "map_50_95": torch.tensor(map50_95, dtype=torch.float32, device=device),
    }


def decode_predictions(
    preds: List[torch.Tensor],
    strides: List[int],
    anchors: List[List[List[int]]],
    num_classes: int,
    device: str,
    conf_thre: float = 0.1,
    nms_thre: float = 0.45,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """
    解码模型预测结果
    
    Args:
        preds: 模型预测结果列表
        strides: 不同尺度的步长
        anchors: 不同尺度的锚框
        num_classes: 类别数量
        device: 设备
        conf_thre: 置信度阈值
        nms_thre: NMS 阈值
        max_det: 最大检测数量
    
    Returns:
        List[torch.Tensor]: 解码后的预测结果
    """
    batch = preds[0].shape[0]
    per_image_boxes = [[] for _ in range(batch)]
    per_image_scores = [[] for _ in range(batch)]
    per_image_labels = [[] for _ in range(batch)]

    for i, pred in enumerate(preds):
        stride = strides[i]
        anchor = torch.tensor(anchors[i], device=device)
        batch, _, height, width = pred.shape
        pred = pred.view(batch, 3, 5 + num_classes, height, width).permute(0, 1, 3, 4, 2).contiguous()

        grid_x = torch.arange(width, device=device).repeat(height, 1).view(1, 1, height, width)
        grid_y = torch.arange(height, device=device).repeat(width, 1).T.view(1, 1, height, width)

        cx = (torch.sigmoid(pred[..., 0]) * 2 - 0.5 + grid_x) * stride
        cy = (torch.sigmoid(pred[..., 1]) * 2 - 0.5 + grid_y) * stride
        width_pred = torch.exp(torch.clamp(pred[..., 2], max=4)) * anchor[:, 0].view(1, 3, 1, 1)
        height_pred = torch.exp(torch.clamp(pred[..., 3], max=4)) * anchor[:, 1].view(1, 3, 1, 1)

        x1 = cx - width_pred / 2
        y1 = cy - height_pred / 2
        x2 = cx + width_pred / 2
        y2 = cy + height_pred / 2

        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf, cls_idx = torch.max(torch.sigmoid(pred[..., 5:]), dim=-1)
        scores = obj_conf * cls_conf

        x1 = x1.view(batch, -1)
        y1 = y1.view(batch, -1)
        x2 = x2.view(batch, -1)
        y2 = y2.view(batch, -1)
        scores = scores.view(batch, -1)
        cls_idx = cls_idx.view(batch, -1)

        for b in range(batch):
            boxes = torch.stack([x1[b], y1[b], x2[b], y2[b]], dim=1)
            score = scores[b]
            label = cls_idx[b]
            mask = score > conf_thre
            if not mask.any():
                continue
            boxes = boxes[mask]
            score = score[mask]
            label = label[mask]
            per_image_boxes[b].append(boxes)
            per_image_scores[b].append(score)
            per_image_labels[b].append(label)

    output = []
    for b in range(batch):
        if per_image_boxes[b]:
            boxes = torch.cat(per_image_boxes[b], dim=0)
            scores = torch.cat(per_image_scores[b], dim=0)
            labels = torch.cat(per_image_labels[b], dim=0)
            keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_thre)
            if len(keep) > max_det:
                keep = keep[:max_det]
            detections = torch.cat(
                [boxes[keep], scores[keep].unsqueeze(1), labels[keep].float().unsqueeze(1)],
                dim=1,
            )
            output.append(detections)
        else:
            output.append(torch.empty((0, 6), device=device))
    return output


def _update_pr_counts_for_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
    class_tp: torch.Tensor,
    class_fp: torch.Tensor,
    class_fn: torch.Tensor,
) -> None:
    """
    更新单个图像的 PR 计数
    
    Args:
        pred_boxes: 预测边界框
        pred_scores: 预测分数
        pred_labels: 预测标签
        gt_boxes: 真实边界框
        gt_labels: 真实标签
        iou_threshold: IOU 阈值
        class_tp: 类别真阳性计数
        class_fp: 类别假阳性计数
        class_fn: 类别假阴性计数
    """
    if len(pred_boxes) == 0:
        for label in gt_labels:
            class_fn[label] += 1
        return
    if len(gt_boxes) == 0:
        for label in pred_labels:
            class_fp[label] += 1
        return

    classes = torch.unique(torch.cat([pred_labels, gt_labels], dim=0))
    for cls in classes:
        cls = int(cls.item())
        pred_mask = pred_labels == cls
        gt_mask = gt_labels == cls
        pred_boxes_c = pred_boxes[pred_mask]
        pred_scores_c = pred_scores[pred_mask]
        gt_boxes_c = gt_boxes[gt_mask]

        if len(pred_boxes_c) == 0 and len(gt_boxes_c) > 0:
            class_fn[cls] += len(gt_boxes_c)
            continue
        if len(gt_boxes_c) == 0 and len(pred_boxes_c) > 0:
            class_fp[cls] += len(pred_boxes_c)
            continue
        if len(pred_boxes_c) == 0 and len(gt_boxes_c) == 0:
            continue

        ious = torchvision.ops.box_iou(pred_boxes_c, gt_boxes_c)
        order = torch.argsort(pred_scores_c, descending=True)
        matched_gt = torch.zeros(len(gt_boxes_c), dtype=torch.bool, device=pred_boxes.device)

        tp = 0
        fp = 0
        for pred_idx in order:
            iou_row = ious[pred_idx]
            best_iou, best_gt_idx = iou_row.max(dim=0)
            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                matched_gt[best_gt_idx] = True
                tp += 1
            else:
                fp += 1

        fn = int((~matched_gt).sum().item())
        class_tp[cls] += tp
        class_fp[cls] += fp
        class_fn[cls] += fn


def evaluate_model(
    detector,
    teacher,
    val_loader,
    num_classes: int,
    device: str,
    img_size: int,
    strides: List[int],
    anchors: List[List[List[int]]],
    iou_threshold: float = 0.5,
    conf_thre: float = 0.3,
    allow_metric_fallback: bool = False,
) -> Tuple[dict, float, float]:
    """
    评估模型性能
    
    Args:
        detector: 检测器模型
        teacher: 教师模型
        val_loader: 验证数据加载器
        num_classes: 类别数量
        device: 设备
        img_size: 图像尺寸
        strides: 不同尺度的步长
        anchors: 不同尺度的锚框
        iou_threshold: IOU 阈值
        conf_thre: 置信度阈值
    
    Returns:
        Tuple[dict, float, float]: 评估指标、宏精确率、宏召回率
    """
    detector.eval()
    teacher.eval()

    map_metric = _safe_create_map_metric(device, allow_fallback=allow_metric_fallback)

    class_tp = torch.zeros(num_classes, device=device)
    class_fp = torch.zeros(num_classes, device=device)
    class_fn = torch.zeros(num_classes, device=device)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            imgs, targets = zip(*batch)
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]

            feat = teacher(imgs)
            p3, p4, p5 = detector(feat)
            preds = decode_predictions(
                [p3, p4, p5],
                strides,
                anchors,
                num_classes,
                device,
                conf_thre=conf_thre,
                nms_thre=0.45,
                max_det=300,
            )

            target_format = []
            for target in targets:
                if len(target) == 0:
                    target_format.append(
                        {
                            "boxes": torch.empty(0, 4).to(device),
                            "labels": torch.empty(0, dtype=torch.int64).to(device),
                        }
                    )
                    continue
                cx = target[:, 1] * img_size
                cy = target[:, 2] * img_size
                width = target[:, 3] * img_size
                height = target[:, 4] * img_size
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2
                target_format.append({"boxes": torch.stack([x1, y1, x2, y2], dim=1), "labels": target[:, 0].long()})

            pred_format = []
            for pred in preds:
                if len(pred) == 0:
                    pred_format.append(
                        {
                            "boxes": torch.empty(0, 4).to(device),
                            "scores": torch.empty(0).to(device),
                            "labels": torch.empty(0, dtype=torch.int64).to(device),
                        }
                    )
                else:
                    pred_format.append({"boxes": pred[:, :4], "scores": pred[:, 4], "labels": pred[:, 5].long()})
            if map_metric is not None:
                map_metric.update(pred_format, target_format)

            for idx in range(len(preds)):
                pred_boxes = preds[idx][:, :4]
                pred_scores = preds[idx][:, 4]
                pred_labels = preds[idx][:, 5].long()
                gt_boxes = target_format[idx]["boxes"]
                gt_labels = target_format[idx]["labels"]
                _update_pr_counts_for_image(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=iou_threshold,
                    class_tp=class_tp,
                    class_fp=class_fp,
                    class_fn=class_fn,
                )
                all_preds.append(preds[idx])
                all_targets.append(target_format[idx])

    if map_metric is not None:
        try:
            raw_metrics = map_metric.compute()
            map_metric.reset()
            map50 = raw_metrics.get("map_50", raw_metrics.get("map", torch.tensor(0.0, device=device))).item()
            map50_95 = raw_metrics.get("map", raw_metrics.get("map_50", torch.tensor(0.0, device=device))).item()
            metrics = {
                "map_50": torch.tensor(float(map50), dtype=torch.float32, device=device),
                "map_50_95": torch.tensor(float(map50_95), dtype=torch.float32, device=device),
            }
        except Exception as exc:
            if allow_metric_fallback:
                print(f"[Warn] torchmetrics mAP 计算失败，回退到内置 mAP。详情: {exc}")
                metrics = _fallback_map(all_preds, all_targets, num_classes, device)
            else:
                raise RuntimeError(f"torchmetrics mAP 计算失败: {exc}") from exc
    else:
        metrics = _fallback_map(all_preds, all_targets, num_classes, device)

    eps = 1e-6
    macro_precision = (class_tp / (class_tp + class_fp + eps)).mean().item()
    macro_recall = (class_tp / (class_tp + class_fn + eps)).mean().item()
    f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + eps)
    metrics["f1"] = torch.tensor(float(f1), dtype=torch.float32, device=device)
    return metrics, macro_precision, macro_recall
