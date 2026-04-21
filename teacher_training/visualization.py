import gc
import traceback
from typing import Dict, List

import cv2
import numpy as np
import torch

from .metrics import decode_predictions


def _wh_iou(w1: float, h1: float, w2: float, h2: float) -> float:
    """
    计算宽高的 IoU
    
    Args:
        w1: 第一个框的宽度
        h1: 第一个框的高度
        w2: 第二个框的宽度
        h2: 第二个框的高度
    
    Returns:
        float: IoU 值
    """
    inter = min(w1, w2) * min(h1, h2)
    union = (w1 * h1) + (w2 * h2) - inter + 1e-6
    return float(inter / union)


def _draw_anchor_boxes_for_target(
    canvas_img: np.ndarray,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    scale_x: float,
    scale_y: float,
    anchors: List[List[List[int]]],
    strides: List[int],
) -> None:
    """
    为目标绘制锚框
    
    Args:
        canvas_img: 画布图像
        cx: 目标中心 x 坐标
        cy: 目标中心 y 坐标
        bw: 目标宽度
        bh: 目标高度
        scale_x: x 方向缩放因子
        scale_y: y 方向缩放因子
        anchors: 锚框配置
        strides: 步长配置
    """
    # 每个尺度选一个与当前 GT 形状最匹配的锚框，并在 GT 中心位置绘制。
    anchor_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]  # BGR: yellow/cyan/magenta
    for scale_idx, scale_anchors in enumerate(anchors):
        best_iou = -1.0
        best_anchor = scale_anchors[0]
        for aw, ah in scale_anchors:
            iou = _wh_iou(bw, bh, float(aw), float(ah))
            if iou > best_iou:
                best_iou = iou
                best_anchor = (aw, ah)

        aw, ah = float(best_anchor[0]), float(best_anchor[1])
        x1 = (cx - aw / 2.0) * scale_x
        y1 = (cy - ah / 2.0) * scale_y
        x2 = (cx + aw / 2.0) * scale_x
        y2 = (cy + ah / 2.0) * scale_y
        pt1 = (int(max(0, x1)), int(max(0, y1)))
        pt2 = (int(min(canvas_img.shape[1] - 1, x2)), int(min(canvas_img.shape[0] - 1, y2)))
        cv2.rectangle(canvas_img, pt1, pt2, anchor_colors[scale_idx % len(anchor_colors)], 1)

        label = f"A@{strides[scale_idx]}"
        cv2.putText(
            canvas_img,
            label,
            (pt1[0], max(12, pt1[1] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            anchor_colors[scale_idx % len(anchor_colors)],
            1,
        )


def visualize_with_opencv(
    teacher_model,
    detector_model,
    dataloader,
    device: str,
    class_names: Dict[int, str],
    strides: List[int],
    anchors: List[List[List[int]]],
    num_classes: int,
    save_path: str = "teacher_feature_visualization.png",
    num_samples: int = 2,
    iou_threshold: float = 0.5,
) -> None:
    """
    使用 OpenCV 可视化结果
    
    Args:
        teacher_model: 教师模型
        detector_model: 检测器模型
        dataloader: 数据加载器
        device: 设备
        class_names: 类别名称
        strides: 步长配置
        anchors: 锚框配置
        num_classes: 类别数量
        save_path: 保存路径
        num_samples: 样本数量
        iou_threshold: IoU 阈值
    """
    teacher_model.eval()
    detector_model.eval()
    try:
        with torch.no_grad():
            batch = next(iter(dataloader))
            imgs, targets = zip(*batch)
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]
            samples = list(zip(imgs, targets))[:num_samples]
            feats = teacher_model(imgs)
            preds = detector_model(feats)
            det_predictions = decode_predictions(
                preds,
                strides,
                anchors,
                num_classes,
                device,
                conf_thre=0.3,
                nms_thre=0.45,
            )

        rows = num_samples
        cols = 5
        cell_w, cell_h = 400, 400
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, (img, target) in enumerate(samples):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]
            img_resized = cv2.resize(img_bgr, (cell_w, cell_h))
            scale_x = cell_w / w
            scale_y = cell_h / h

            gt_boxes = []
            for target_item in target:
                cls, cx, cy, bw, bh = target_item.cpu().numpy()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                gt_boxes.append((x1, y1, x2, y2, int(cls)))
                pt1 = (int(x1 * scale_x), int(y1 * scale_y))
                pt2 = (int(x2 * scale_x), int(y2 * scale_y))
                cv2.rectangle(img_resized, pt1, pt2, (0, 255, 0), 2)
                label = f"GT {class_names[int(cls)]}"
                cv2.putText(img_resized, label, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 在验证可视化中叠加锚框（每尺度绘制一个最佳匹配锚框）。
                cx_px = float(cx * w)
                cy_px = float(cy * h)
                bw_px = float(bw * w)
                bh_px = float(bh * h)
                _draw_anchor_boxes_for_target(
                    canvas_img=img_resized,
                    cx=cx_px,
                    cy=cy_px,
                    bw=bw_px,
                    bh=bh_px,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    anchors=anchors,
                    strides=strides,
                )

            pred_boxes = det_predictions[idx]
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                pred_xyxy = pred_boxes[:, :4].cpu().numpy()
                pred_scores = pred_boxes[:, 4].cpu().numpy()
                pred_labels = pred_boxes[:, 5].cpu().numpy().astype(int)
                gt_xyxy = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in gt_boxes], dtype=np.float32)
                ious = []
                for gt in gt_xyxy:
                    iou_row = []
                    for pred in pred_xyxy:
                        inter_x1 = max(gt[0], pred[0])
                        inter_y1 = max(gt[1], pred[1])
                        inter_x2 = min(gt[2], pred[2])
                        inter_y2 = min(gt[3], pred[3])
                        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
                        area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
                        union = area_gt + area_pred - inter
                        iou = inter / union if union > 0 else 0
                        iou_row.append(iou)
                    ious.append(iou_row)
                ious = np.array(ious)
                matched_pred = set()
                for gt_idx in range(len(gt_boxes)):
                    best_iou = iou_threshold
                    best_pred_idx = -1
                    for pred_idx in range(len(pred_xyxy)):
                        if pred_idx in matched_pred:
                            continue
                        if ious[gt_idx, pred_idx] > best_iou:
                            best_iou = ious[gt_idx, pred_idx]
                            best_pred_idx = pred_idx
                    if best_pred_idx != -1:
                        matched_pred.add(best_pred_idx)
                        box = pred_xyxy[best_pred_idx].astype(int)
                        pt1 = (int(box[0] * scale_x), int(box[1] * scale_y))
                        pt2 = (int(box[2] * scale_x), int(box[3] * scale_y))
                        cv2.rectangle(img_resized, pt1, pt2, (255, 0, 0), 2)
                        text = f"{class_names[pred_labels[best_pred_idx]]} {pred_scores[best_pred_idx]:.2f}"
                        cv2.putText(img_resized, text, (pt1[0], pt2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            canvas[idx * cell_h : (idx + 1) * cell_h, 0:cell_w] = img_resized

            feat = teacher_model(img.unsqueeze(0).to(device))
            heatmap_np = feat.squeeze().detach().cpu().numpy()
            heatmap_np = (heatmap_np * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap_color, (cell_w, cell_h))
            cv2.putText(heatmap_resized, "Teacher Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            canvas[idx * cell_h : (idx + 1) * cell_h, cell_w : 2 * cell_w] = heatmap_resized

            p3, p4, p5 = detector_model(feat)
            for col_idx, (feat_map, name) in enumerate([(p3, "P3"), (p4, "P4"), (p5, "P5")], start=2):
                fmap = feat_map.squeeze().abs().mean(dim=0).detach().cpu().numpy()
                fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                fmap_uint8 = (fmap * 255).astype(np.uint8)
                fmap_color = cv2.applyColorMap(fmap_uint8, cv2.COLORMAP_JET)
                fmap_resized = cv2.resize(fmap_color, (cell_w, cell_h))
                cv2.putText(fmap_resized, f"{name} Feature", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                canvas[idx * cell_h : (idx + 1) * cell_h, col_idx * cell_w : (col_idx + 1) * cell_w] = fmap_resized

        cv2.imwrite(save_path, canvas)
        print(f"可视化已保存: {save_path}")
    except Exception as exc:
        print(f"可视化失败: {exc}")
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
