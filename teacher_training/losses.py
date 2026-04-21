import math
from typing import List

import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """YOLO 损失函数"""
    def __init__(
        self,
        strides: List[int],
        anchors: List[List[List[int]]],
        num_classes: int,
        box_weight: float,
        obj_weight: float,
        cls_weight: float,
        label_smoothing: float = 0.0,
    ):
        """
        初始化 YOLO 损失函数
        
        Args:
            strides: 不同尺度的步长
            anchors: 不同尺度的锚框
            num_classes: 类别数量
            box_weight: 边界框损失权重
            obj_weight: 目标置信度损失权重
            cls_weight: 类别损失权重
        """
        super().__init__()
        self.strides = strides
        self.anchors = anchors
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def _bbox_iou_xywh(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_cx, target_cy, target_w, target_h = (
            target_boxes[:, 0],
            target_boxes[:, 1],
            target_boxes[:, 2],
            target_boxes[:, 3],
        )
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2
        target_x1 = target_cx - target_w / 2
        target_y1 = target_cy - target_h / 2
        target_x2 = target_cx + target_w / 2
        target_y2 = target_cy + target_h / 2
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area + 1e-7
        return inter_area / union_area

    def ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算 CIOU 损失
        
        Args:
            pred_boxes: 预测边界框
            target_boxes: 目标边界框
        
        Returns:
            torch.Tensor: CIOU 损失值
        """
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_cx, target_cy, target_w, target_h = (
            target_boxes[:, 0],
            target_boxes[:, 1],
            target_boxes[:, 2],
            target_boxes[:, 3],
        )
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2
        target_x1 = target_cx - target_w / 2
        target_y1 = target_cy - target_h / 2
        target_x2 = target_cx + target_w / 2
        target_y2 = target_cy + target_h / 2
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area + 1e-7
        iou = inter_area / union_area
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        center_distance = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        enclose_diagonal = enclose_w ** 2 + enclose_h ** 2 + 1e-7
        v = (4 / (torch.pi**2)) * torch.pow(
            torch.atan2(target_w, target_h + 1e-7) - torch.atan(pred_w / (pred_h + 1e-7)), 2
        )
        alpha = v / (1 - iou + v + 1e-7)
        ciou = iou - (center_distance / enclose_diagonal + alpha * v)
        return (1 - ciou).mean()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, stride_idx: int):
        """
        前向传播计算损失
        
        Args:
            preds: 模型预测结果
            targets: 目标标签
            stride_idx: 尺度索引
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 总损失、边界框损失、置信度损失、类别损失
        """
        stride = self.strides[stride_idx]
        anchors = self.anchors[stride_idx]
        batch_size, _, height, width = preds.shape
        num_anchors = len(anchors)
        preds = preds.permute(0, 2, 3, 1).reshape(batch_size, height, width, num_anchors, -1).contiguous()
        grid_x = torch.arange(width, device=preds.device).view(1, 1, width, 1).repeat(batch_size, 1, 1, num_anchors)
        grid_y = torch.arange(height, device=preds.device).view(1, height, 1, 1).repeat(batch_size, 1, 1, num_anchors)
        scaled_anchors = torch.tensor(anchors, device=preds.device) / stride
        anchor_w = scaled_anchors[:, 0].view(1, 1, 1, num_anchors)
        anchor_h = scaled_anchors[:, 1].view(1, 1, 1, num_anchors)
        pred_cx = (torch.sigmoid(preds[..., 0]) * 2 - 0.5) + grid_x
        pred_cy = (torch.sigmoid(preds[..., 1]) * 2 - 0.5) + grid_y
        pred_w = torch.exp(torch.clamp(preds[..., 2], max=4)) * anchor_w
        pred_h = torch.exp(torch.clamp(preds[..., 3], max=4)) * anchor_h
        pred_conf = preds[..., 4]
        pred_cls = preds[..., 5:]
        target_cx = targets[..., 0]
        target_cy = targets[..., 1]
        target_w = targets[..., 2]
        target_h = targets[..., 3]
        target_conf = targets[..., 4]
        target_cls = targets[..., 5:]
        obj_mask = target_conf == 1.0
        noobj_mask = target_conf == 0.0
        tgt_cx = grid_x + target_cx
        tgt_cy = grid_y + target_cy
        tgt_w = torch.exp(target_w) * anchor_w
        tgt_h = torch.exp(target_h) * anchor_h
        box_loss = torch.tensor(0.0, device=preds.device)
        iou_target = torch.zeros_like(target_conf)
        if obj_mask.sum() > 0:
            pred_box = torch.stack([pred_cx[obj_mask], pred_cy[obj_mask], pred_w[obj_mask], pred_h[obj_mask]], dim=1)
            tgt_box = torch.stack([tgt_cx[obj_mask], tgt_cy[obj_mask], tgt_w[obj_mask], tgt_h[obj_mask]], dim=1)
            box_loss = self.ciou_loss(pred_box, tgt_box)
            iou_target[obj_mask] = self._bbox_iou_xywh(pred_box, tgt_box).detach().clamp(0.0, 1.0)
        obj_loss = self.bce(pred_conf[obj_mask], iou_target[obj_mask]) if obj_mask.sum() > 0 else torch.tensor(0.0, device=preds.device)
        noobj_loss = (
            self.bce(pred_conf[noobj_mask], iou_target[noobj_mask])
            if noobj_mask.sum() > 0
            else torch.tensor(0.0, device=preds.device)
        )
        conf_loss = obj_loss + 0.2 * noobj_loss
        cls_loss = torch.tensor(0.0, device=preds.device)
        if obj_mask.sum() > 0:
            if self.label_smoothing > 0:
                cls_t = target_cls[obj_mask] * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                cls_t = target_cls[obj_mask]
            cls_loss = self.bce(pred_cls[obj_mask], cls_t)
        total_loss = self.box_weight * box_loss + self.obj_weight * conf_loss + self.cls_weight * cls_loss
        return total_loss, box_loss, conf_loss, cls_loss


def build_target(
    targets_list: List[torch.Tensor],
    anchors: List[List[int]],
    stride: int,
    num_classes: int,
    img_size: int,
    device: str,
) -> torch.Tensor:
    """
    构建单尺度目标张量
    
    Args:
        targets_list: 目标列表
        anchors: 锚框列表
        stride: 步长
        num_classes: 类别数量
        img_size: 图像尺寸
        device: 设备
    
    Returns:
        torch.Tensor: 目标张量
    """
    batch_size = len(targets_list)
    grid_h = img_size // stride
    grid_w = img_size // stride
    num_anchors = len(anchors)
    target_tensor = torch.zeros((batch_size, grid_h, grid_w, num_anchors, 5 + num_classes), device=device)
    scaled_anchors = torch.tensor(anchors, device=device) / stride

    for batch_idx, targets in enumerate(targets_list):
        if len(targets) == 0:
            continue
        for target in targets:
            cls, cx_norm, cy_norm, bw_norm, bh_norm = target
            if bw_norm <= 0 or bh_norm <= 0:
                continue
            cx = cx_norm * grid_w
            cy = cy_norm * grid_h
            bw = bw_norm * grid_w
            bh = bh_norm * grid_h
            gi = int(cx)
            gj = int(cy)
            if gi < 0 or gi >= grid_w or gj < 0 or gj >= grid_h:
                continue
            box_wh = torch.tensor([bw, bh], device=device)
            inter = torch.min(box_wh[0], scaled_anchors[:, 0]) * torch.min(box_wh[1], scaled_anchors[:, 1])
            union = box_wh[0] * box_wh[1] + scaled_anchors[:, 0] * scaled_anchors[:, 1] - inter
            iou = inter / (union + 1e-6)
            best_idx = torch.argmax(iou).item()
            target_tensor[batch_idx, gj, gi, best_idx, 0] = cx - gi
            target_tensor[batch_idx, gj, gi, best_idx, 1] = cy - gj
            target_tensor[batch_idx, gj, gi, best_idx, 2] = torch.log(bw / scaled_anchors[best_idx, 0] + 1e-6)
            target_tensor[batch_idx, gj, gi, best_idx, 3] = torch.log(bh / scaled_anchors[best_idx, 1] + 1e-6)
            target_tensor[batch_idx, gj, gi, best_idx, 4] = 1.0
            target_tensor[batch_idx, gj, gi, best_idx, 5 + int(cls)] = 1.0
    return target_tensor


def build_targets_multi_scale(
    targets_list: List[torch.Tensor],
    anchors_per_scale: List[List[List[int]]],
    strides: List[int],
    num_classes: int,
    img_size: int,
    device: str,
    anchor_match_ratio_thresh: float = 4.0,
    assign_neighbor_cells: bool = True,
) -> List[torch.Tensor]:
    """
    构建多尺度目标张量
    
    Args:
        targets_list: 目标列表
        anchors_per_scale: 每个尺度的锚框
        strides: 不同尺度的步长
        num_classes: 类别数量
        img_size: 图像尺寸
        device: 设备
    
    Returns:
        List[torch.Tensor]: 每个尺度的目标张量列表
    """
    batch_size = len(targets_list)
    targets_per_scale = []
    flat_anchors = []

    for scale_idx, anchors in enumerate(anchors_per_scale):
        grid_h = img_size // strides[scale_idx]
        grid_w = img_size // strides[scale_idx]
        targets_per_scale.append(torch.zeros((batch_size, grid_h, grid_w, len(anchors), 5 + num_classes), device=device))
        for anchor_idx, (aw, ah) in enumerate(anchors):
            flat_anchors.append((scale_idx, anchor_idx, float(aw), float(ah)))

    for batch_idx, targets in enumerate(targets_list):
        if len(targets) == 0:
            continue
        for target in targets:
            cls, cx_norm, cy_norm, bw_norm, bh_norm = target
            if bw_norm <= 0 or bh_norm <= 0:
                continue

            bw_px = float(bw_norm * img_size)
            bh_px = float(bh_norm * img_size)
            ratios = []
            for _, _, aw, ah in flat_anchors:
                rw = max(bw_px / (aw + 1e-6), aw / (bw_px + 1e-6))
                rh = max(bh_px / (ah + 1e-6), ah / (bh_px + 1e-6))
                ratios.append(max(rw, rh))
            ratio_tensor = torch.tensor(ratios, device=device)
            matched = (ratio_tensor < anchor_match_ratio_thresh).nonzero(as_tuple=False).view(-1)
            if len(matched) == 0:
                matched = torch.tensor([int(torch.argmin(ratio_tensor).item())], device=device)

            for flat_idx_tensor in matched:
                flat_idx = int(flat_idx_tensor.item())
                scale_idx, anchor_idx, aw_px, ah_px = flat_anchors[flat_idx]
                stride = strides[scale_idx]
                grid_h = img_size // stride
                grid_w = img_size // stride
                cx = float(cx_norm * grid_w)
                cy = float(cy_norm * grid_h)
                bw = float(bw_norm * grid_w)
                bh = float(bh_norm * grid_h)
                gi = int(cx)
                gj = int(cy)
                if gi < 0 or gi >= grid_w or gj < 0 or gj >= grid_h:
                    continue

                offsets = [(0, 0)]
                if assign_neighbor_cells:
                    dx_l = cx - gi
                    dx_r = (gi + 1) - cx
                    dy_t = cy - gj
                    dy_b = (gj + 1) - cy
                    if dx_l < 0.5:
                        offsets.append((-1, 0))
                    if dx_r < 0.5:
                        offsets.append((1, 0))
                    if dy_t < 0.5:
                        offsets.append((0, -1))
                    if dy_b < 0.5:
                        offsets.append((0, 1))

                aw = aw_px / stride
                ah = ah_px / stride
                target_tensor = targets_per_scale[scale_idx]
                for ox, oy in offsets:
                    gx = gi + ox
                    gy = gj + oy
                    if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
                        continue
                    target_tensor[batch_idx, gy, gx, anchor_idx, 0] = cx - gx
                    target_tensor[batch_idx, gy, gx, anchor_idx, 1] = cy - gy
                    target_tensor[batch_idx, gy, gx, anchor_idx, 2] = math.log(bw / (aw + 1e-6) + 1e-6)
                    target_tensor[batch_idx, gy, gx, anchor_idx, 3] = math.log(bh / (ah + 1e-6) + 1e-6)
                    target_tensor[batch_idx, gy, gx, anchor_idx, 4] = 1.0
                    target_tensor[batch_idx, gy, gx, anchor_idx, 5 + int(cls)] = 1.0

    return targets_per_scale
