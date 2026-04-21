import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


def load_class_names(yaml_path: str) -> Tuple[Dict[int, str], int]:
    """
    从 YAML 文件加载类别名称
    
    Args:
        yaml_path: YAML 文件路径
    
    Returns:
        Tuple[Dict[int, str], int]: 类别映射字典和类别数量
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    class_names = config.get("names", [])
    if isinstance(class_names, list):
        class_map = {i: name for i, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_map = {int(i): name for i, name in class_names.items()}
    else:
        raise ValueError(f"Unsupported 'names' format in {yaml_path}")
    return class_map, len(class_map)


def resolve_dataset_dirs(yaml_path: str) -> Tuple[str, str, str, str]:
    """
    解析数据集目录
    
    Args:
        yaml_path: YAML 文件路径
    
    Returns:
        Tuple[str, str, str, str]: 训练图像目录、训练标签目录、验证图像目录、验证标签目录
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    yaml_abs = os.path.abspath(yaml_path)
    yaml_dir = os.path.dirname(yaml_abs)
    project_dir = os.path.dirname(os.path.dirname(yaml_abs))
    path_cfg = str(config.get("path", "")).strip()

    if path_cfg:
        if os.path.isabs(path_cfg):
            root_candidates = [os.path.normpath(path_cfg)]
        else:
            root_candidates = [
                os.path.normpath(os.path.join(yaml_dir, path_cfg)),
                os.path.normpath(os.path.join(project_dir, path_cfg)),
                os.path.normpath(os.path.join(os.getcwd(), path_cfg)),
            ]
    else:
        root_candidates = [os.path.dirname(yaml_abs), os.getcwd()]

    uniq_candidates = []
    seen = set()
    for path in root_candidates:
        if path not in seen:
            uniq_candidates.append(path)
            seen.add(path)
    root_candidates = uniq_candidates

    def _resolve_split_dir(split_value: str, default_mode: str) -> str:
        if not split_value:
            return os.path.join(root_candidates[0], default_mode, "images")
        if os.path.isabs(split_value):
            return os.path.normpath(split_value)

        candidates = [os.path.normpath(os.path.join(root, split_value)) for root in root_candidates]
        for cand in candidates:
            if os.path.isdir(cand):
                return cand
        return candidates[0]

    def _infer_label_dir(img_dir: str) -> str:
        marker = f"{os.sep}images{os.sep}"
        if marker in img_dir:
            return img_dir.replace(marker, f"{os.sep}labels{os.sep}", 1)
        if img_dir.endswith(f"{os.sep}images"):
            return img_dir[: -len("images")] + "labels"
        return os.path.join(os.path.dirname(img_dir), "labels")

    train_img_dir = _resolve_split_dir(config.get("train"), "train")
    val_img_dir = _resolve_split_dir(config.get("val"), "val")
    train_label_dir = _infer_label_dir(train_img_dir)
    val_label_dir = _infer_label_dir(val_img_dir)
    return train_img_dir, train_label_dir, val_img_dir, val_label_dir


def letterbox(img: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)):
    """
    对图像进行 letterbox 处理
    
    Args:
        img: 输入图像
        new_shape: 目标形状
        color: 填充颜色
    
    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]: 处理后的图像、缩放比例、填充偏移
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (left, top)


def color_jitter(img: np.ndarray) -> np.ndarray:
    """
    对图像进行颜色抖动增强
    
    Args:
        img: 输入图像
    
    Returns:
        np.ndarray: 增强后的图像
    """
    if random.random() < 0.4:
        delta = random.uniform(-0.15, 0.15) * 255
        img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        alpha = random.uniform(0.85, 1.15)
        img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        alpha = random.uniform(0.85, 1.15)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img


class MilitaryDataset(Dataset):
    """军事目标检测数据集"""
    def __init__(self, root_dir: str, mode: str = "train", img_size: int = 640, img_dir: str = None, label_dir: str = None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            mode: 数据集模式 (train/val)
            img_size: 图像尺寸
            img_dir: 图像目录
            label_dir: 标签目录
        """
        self.img_size = img_size
        self.mode = mode
        self.img_dir = img_dir if img_dir is not None else os.path.join(root_dir, mode, "images")
        self.label_dir = label_dir if label_dir is not None else os.path.join(root_dir, mode, "labels")
        self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    def __len__(self) -> int:
        """
        获取数据集长度
        
        Returns:
            int: 数据集长度
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 图像和目标框
        """
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        targets = torch.zeros((0, 5), dtype=torch.float32)
        if os.path.exists(label_path):
            try:
                labels = np.loadtxt(label_path).reshape(-1, 5)
                targets = torch.tensor(labels, dtype=torch.float32)
            except Exception:
                pass

        if self.mode == "train":
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                if len(targets) > 0:
                    targets[:, 1] = 1.0 - targets[:, 1]
            img = color_jitter(img)

        img, ratio, (pad_left, pad_top) = letterbox(img, self.img_size)

        if len(targets) > 0:
            cx_ori = targets[:, 1] * w0
            cy_ori = targets[:, 2] * h0
            w_ori = targets[:, 3] * w0
            h_ori = targets[:, 4] * h0
            cx_new = cx_ori * ratio + pad_left
            cy_new = cy_ori * ratio + pad_top
            w_new = w_ori * ratio
            h_new = h_ori * ratio
            targets[:, 1] = cx_new / self.img_size
            targets[:, 2] = cy_new / self.img_size
            targets[:, 3] = w_new / self.img_size
            targets[:, 4] = h_new / self.img_size

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, targets


class TeacherDataset(Dataset):
    """教师模型数据集"""
    def __init__(self, root_dir: str, mode: str = "train", img_size: int = 640, img_dir: str = None, label_dir: str = None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            mode: 数据集模式 (train/val)
            img_size: 图像尺寸
            img_dir: 图像目录
            label_dir: 标签目录
        """
        self.img_size = img_size
        self.mode = mode
        self.img_dir = img_dir if img_dir is not None else os.path.join(root_dir, mode, "images")
        self.label_dir = label_dir if label_dir is not None else os.path.join(root_dir, mode, "labels")
        self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    def __len__(self) -> int:
        """
        获取数据集长度
        
        Returns:
            int: 数据集长度
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 图像和掩码
        """
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        targets = np.zeros((0, 5), dtype=np.float32)
        if os.path.exists(label_path):
            try:
                targets = np.loadtxt(label_path).reshape(-1, 5)
            except Exception:
                pass

        img, ratio, (pad_left, pad_top) = letterbox(img, self.img_size)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        for target in targets:
            _, cx, cy, bw, bh = target
            x1 = int((cx - bw / 2) * w0 * ratio + pad_left)
            y1 = int((cy - bh / 2) * h0 * ratio + pad_top)
            x2 = int((cx + bw / 2) * w0 * ratio + pad_left)
            y2 = int((cy + bh / 2) * h0 * ratio + pad_top)
            x1 = np.clip(x1, 0, self.img_size - 1)
            y1 = np.clip(y1, 0, self.img_size - 1)
            x2 = np.clip(x2, 0, self.img_size - 1)
            y2 = np.clip(y2, 0, self.img_size - 1)
            mask[y1:y2, x1:x2] = 1.0

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    数据加载的 collate 函数
    
    Args:
        batch: 批量数据
    
    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: 处理后的批量数据
    """
    return batch


def teacher_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    教师模型数据加载的 collate 函数
    
    Args:
        batch: 批量数据
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 处理后的批量图像和掩码
    """
    imgs = []
    masks = []
    for img, mask in batch:
        imgs.append(img)
        masks.append(mask)
    return torch.stack(imgs), torch.stack(masks)
