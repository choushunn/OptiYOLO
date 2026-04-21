import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime


# =========================================================
# 轻量化卷积块：深度可分离卷积 + BN + SiLU（轻量化激活函数）
# =========================================================
class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LightConvBlock, self).__init__()
        # 深度卷积（逐通道卷积，轻量化核心）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        # 点卷积（1×1卷积，调整通道数）
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # 比ReLU更高效的轻量化激活

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# =========================================================
# 轻量化检测头（基于FPN的多尺度特征生成）- 保持原样不变
# =========================================================
class YOLOLightHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=27):
        super(YOLOLightHead, self).__init__()
        # 基础通道数（轻量化设计，避免过多通道）
        base_ch = 32

        # 1. 初始特征提取（将1通道特征图扩展为基础通道）
        self.init_conv = LightConvBlock(in_channels, base_ch, kernel_size=3, stride=1)

        # 2. 下采样生成P5（20×20）：640→320→160→80→40→20（5次下采样，stride=2）
        self.down_to_p5 = nn.Sequential(
            LightConvBlock(base_ch, base_ch * 2, stride=2),  # 640→320, 32→64
            LightConvBlock(base_ch * 2, base_ch * 4, stride=2),  # 320→160, 64→128
            LightConvBlock(base_ch * 4, base_ch * 8, stride=2),  # 160→80, 128→256
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),  # 80→40, 256→256
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2)  # 40→20, 256→256
        )

        # 3. P5上采样 + 融合生成P4（40×40）
        self.up_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')  # 轻量化上采样
        self.fuse_p4 = LightConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4)  # 融合P5上采样特征和40×40中间特征

        # 4. P4上采样 + 融合生成P3（80×80）
        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p3 = LightConvBlock(base_ch * 4 + base_ch * 8, base_ch * 2)  # 融合P4上采样特征和80×80中间特征

        # 5. 各尺度特征调整为27通道输出
        self.head_p5 = nn.Conv2d(base_ch * 8, out_channels, 1)  # P5:20×20×27
        self.head_p4 = nn.Conv2d(base_ch * 4, out_channels, 1)  # P4:40×40×27
        self.head_p3 = nn.Conv2d(base_ch * 2, out_channels, 1)  # P3:80×80×27

    def forward(self, x):
        # x: [B, 1, 640, 640]（Backbone输出）
        B = x.shape[0]

        # 初始特征提取
        x_init = self.init_conv(x)  # [B, 32, 640, 640]

        # 分步下采样，保存中间特征（用于融合）
        x320 = self.down_to_p5[0](x_init)  # [B, 64, 320, 320]
        x160 = self.down_to_p5[1](x320)  # [B, 128, 160, 160]
        x80 = self.down_to_p5[2](x160)  # [B, 256, 80, 80]
        x40 = self.down_to_p5[3](x80)  # [B, 256, 40, 40]
        p5 = self.down_to_p5[4](x40)  # [B, 256, 20, 20] → P5基础特征

        # 生成P4
        p5_up = self.up_p5_to_p4(p5)  # [B, 256, 40, 40]
        p4_fuse = torch.cat([p5_up, x40], dim=1)  # 特征融合：[B, 512, 40, 40]
        p4 = self.fuse_p4(p4_fuse)  # [B, 256, 40, 40] → P4基础特征

        # 生成P3
        p4_up = self.up_p4_to_p3(p4)  # [B, 256, 80, 80]
        p3_fuse = torch.cat([p4_up, x80], dim=1)  # 特征融合：[B, 512, 80, 80]
        p3 = self.fuse_p3(p3_fuse)  # [B, 128, 80, 80] → P3基础特征

        # 调整通道数为27
        p5_out = self.head_p5(p5)  # [B, 27, 20, 20]
        p4_out = self.head_p4(p4)  # [B, 27, 40, 40]
        p3_out = self.head_p3(p3)  # [B, 27, 80, 80]

        return p3_out, p4_out, p5_out


# =========================================================
# 教师网络（修改为输出1通道640×640特征图，匹配检测头输入）
# =========================================================
class ConvTeacher(nn.Module):
    """
    Conv-based teacher that outputs 1-channel feature map at full resolution
    Output:
        - single-channel [B, 1, 640, 640] feature map
    """

    def __init__(self):
        super().__init__()

        # Stem: downsample x2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        # Downsample x4
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Downsample x8  → P3 scale
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        # P3 refinement (YOLOv5 C3-like, but lightweight)
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # 新增：上采样回原尺寸
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.Conv2d(8, 1, kernel_size=1, bias=False)
        )

        # 新增：跳跃连接，保留更多细节
        self.skip_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: RGB or Gray image tensor [B, C, H, W]
        Returns:
            f: [B, 1, H, W] feature map at full resolution
        """
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)  # to gray
        else:
            x_gray = x

        # 保存原始输入用于跳跃连接
        skip = self.skip_conv(x_gray)

        # 下采样路径
        f = self.conv1(x_gray)
        f = self.conv2(f)
        f = self.conv3(f)  # [B, 64, H/8, W/8]
        f = self.refine(f)  # [B, 32, H/8, W/8]

        # 上采样回原尺寸
        f = self.upsample(f)  # [B, 1, H, W]

        # 添加跳跃连接，保留原始信息
        f = f + skip

        # 去除负值并归一化到[0,1]
        f = torch.abs(f)
        f = torch.sigmoid(f)

        return f


# =========================================================
# 自定义数据集 (使用military数据集)
# =========================================================
class MilitaryDataset(Dataset):
    """
    读取military数据集中的训练图像
    路径: E:/military/train/images/
    """

    def __init__(self, data_root, teacher, detection_head, device, img_size=(640, 640), split='train'):
        """
        Args:
            data_root (str): 数据集根目录，例如: E:/military
            teacher (nn.Module): 教师网络（输出1通道640×640特征图）
            detection_head (nn.Module): 检测头（输入1通道640×640）
            device: 设备
            img_size (tuple): 输入图像大小
            split (str): 'train', 'val', 或 'test'
        """
        # 构建图像路径
        self.img_dir = os.path.join(data_root, split, 'images')

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Images directory not found: {self.img_dir}")

        # 获取所有图像文件
        self.files = sorted([
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        print(f"Found {len(self.files)} images in {self.img_dir}")

        self.teacher = teacher.to(device).eval()
        self.detection_head = detection_head.to(device).eval()
        self.device = device
        self.img_size = img_size

        # 转换：灰度图用于输入，RGB用于教师（教师内部会转灰度）
        self.gray_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        # 输入灰度图 (用于可视化输入)
        x_gray = self.gray_transform(img)  # [1, H, W]

        # 使用教师网络和检测头生成多尺度特征图
        with torch.no_grad():
            rgb_tensor = self.rgb_transform(img).unsqueeze(0).to(self.device)  # [1,3,H,W]

            # 教师网络输出1通道640×640特征图
            teacher_feat = self.teacher(rgb_tensor)  # [1,1,640,640]

            # 检测头直接使用教师特征图作为输入
            p3, p4, p5 = self.detection_head(teacher_feat)  # 各尺度输出

            # 移动到CPU
            teacher_feat = teacher_feat.squeeze(0).cpu()  # [1,640,640]
            p3 = p3.squeeze(0).cpu()  # [27,80,80]
            p4 = p4.squeeze(0).cpu()  # [27,40,40]
            p5 = p5.squeeze(0).cpu()  # [27,20,20]

        return x_gray, teacher_feat, (p3, p4, p5)


# =========================================================
# 可视化函数 (显示教师特征和多尺度特征)
# =========================================================
def save_all_features(epoch, teacher_feat, multiscale_features, save_dir, prefix="train", input_images=None):
    """
    创建完整的特征对比图：输入图像 | 教师特征 | P3 | P4 | P5
    """
    os.makedirs(save_dir, exist_ok=True)

    p3, p4, p5 = multiscale_features
    batch_size = min(4, teacher_feat.shape[0])  # 最多显示4个样本
    if batch_size == 0:
        return

    # 创建5列的子图
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))

    # 如果只有一行，确保axes是2D数组
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    # 归一化辅助函数
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    for i in range(batch_size):
        # 输入图像（第一列）
        if input_images is not None:
            inp = input_images[i, 0].numpy()
            axes[i, 0].imshow(norm(inp), cmap="gray")
            axes[i, 0].set_title(f"Input {i + 1}")
        else:
            axes[i, 0].set_title(f"Sample {i + 1} (No input)")
        axes[i, 0].axis("off")

        # 教师特征（第二列）
        t_feat = teacher_feat[i, 0].numpy()
        im_t = axes[i, 1].imshow(norm(t_feat), cmap="hot")
        axes[i, 1].set_title(f"Teacher Feature {i + 1}")
        axes[i, 1].axis("off")
        plt.colorbar(im_t, ax=axes[i, 1], fraction=0.046)

        # P3特征 (80×80) - 第三列
        p3_feat = p3[i, 0].numpy()
        im3 = axes[i, 2].imshow(norm(p3_feat), cmap="hot")
        axes[i, 2].set_title(f"P3 (80×80) - Sample {i + 1}")
        axes[i, 2].axis("off")
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)

        # P4特征 (40×40) - 第四列
        p4_feat = p4[i, 0].numpy()
        p4_up = F.interpolate(torch.from_numpy(p4_feat).unsqueeze(0).unsqueeze(0),
                              size=(80, 80), mode='bilinear')[0, 0].numpy()
        im4 = axes[i, 3].imshow(norm(p4_up), cmap="hot")
        axes[i, 3].set_title(f"P4 (40×40) - Sample {i + 1}")
        axes[i, 3].axis("off")
        plt.colorbar(im4, ax=axes[i, 3], fraction=0.046)

        # P5特征 (20×20) - 第五列
        p5_feat = p5[i, 0].numpy()
        p5_up = F.interpolate(torch.from_numpy(p5_feat).unsqueeze(0).unsqueeze(0),
                              size=(80, 80), mode='bilinear')[0, 0].numpy()
        im5 = axes[i, 4].imshow(norm(p5_up), cmap="hot")
        axes[i, 4].set_title(f"P5 (20×20) - Sample {i + 1}")
        axes[i, 4].axis("off")
        plt.colorbar(im5, ax=axes[i, 4], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Vis] saved: {save_path}")


# =========================================================
# 训练函数 (教师网络 + 检测头，保持检测头不变)
# =========================================================
def train_teacher_with_head():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化教师网络（修改后：输出1通道640×640）
    teacher = ConvTeacher().to(device)
    teacher.eval()  # 教师始终在eval模式，不更新参数

    # 初始化检测头（保持原样，输入1通道640×640）
    detection_head = YOLOLightHead(in_channels=1, out_channels=27).to(device)
    detection_head.eval()  # 检测头也设置为eval模式

    print("=" * 60)
    print("Teacher network (Modified) initialized.")
    print(f"  Output shape: [B, 1, 640, 640] (full resolution)")
    print("Detection head (unchanged) initialized.")
    print(f"  Input shape: [B, 1, 640, 640]")
    print(f"  Output shapes: P3 [B,27,80,80], P4 [B,27,40,40], P5 [B,27,20,20]")
    print("=" * 60)

    # ========== 设置数据集路径 ==========
    data_root = r"E:/military"  # 数据集根目录

    # 使用训练集
    dataset = MilitaryDataset(data_root, teacher, detection_head, device,
                              img_size=(640, 640), split='train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = f"output/teacher_head_vis_{timestamp}"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Results will be saved to: {vis_dir}")

    # 记录各尺度特征的平均值作为监控
    loss_curve_teacher = []
    loss_curve_p3 = []
    loss_curve_p4 = []
    loss_curve_p5 = []

    for epoch in range(400):  # 训练轮次
        epoch_mean_teacher = 0.0
        epoch_mean_p3 = 0.0
        epoch_mean_p4 = 0.0
        epoch_mean_p5 = 0.0
        batch_count = 0

        for x, teacher_feat, (p3, p4, p5) in tqdm(loader, desc=f"Epoch {epoch}"):
            # 计算各特征图的均值
            mean_teacher = teacher_feat.mean().item()
            mean_p3 = p3.mean().item()
            mean_p4 = p4.mean().item()
            mean_p5 = p5.mean().item()

            epoch_mean_teacher += mean_teacher
            epoch_mean_p3 += mean_p3
            epoch_mean_p4 += mean_p4
            epoch_mean_p5 += mean_p5
            batch_count += 1

        avg_teacher = epoch_mean_teacher / batch_count
        avg_p3 = epoch_mean_p3 / batch_count
        avg_p4 = epoch_mean_p4 / batch_count
        avg_p5 = epoch_mean_p5 / batch_count

        loss_curve_teacher.append(avg_teacher)
        loss_curve_p3.append(avg_p3)
        loss_curve_p4.append(avg_p4)
        loss_curve_p5.append(avg_p5)

        # 每2个epoch可视化一次
        if epoch % 2 == 0:
            # 获取一批数据用于可视化
            vis_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            vis_batch = next(iter(vis_loader))
            x_vis, teacher_vis, (p3_vis, p4_vis, p5_vis) = vis_batch

            save_all_features(epoch, teacher_vis, (p3_vis, p4_vis, p5_vis), vis_dir,
                              input_images=x_vis)

        print(f"Epoch {epoch:3d} | Teacher: {avg_teacher:.6f} | P3: {avg_p3:.6f} | P4: {avg_p4:.6f} | P5: {avg_p5:.6f}")

    # 保存监控曲线
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(loss_curve_teacher, label="Teacher", linewidth=2, color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")
    plt.title("Teacher Feature (640×640) Mean")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(loss_curve_p3, label="P3", linewidth=2, color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")
    plt.title("P3 (80×80) Feature Mean")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(loss_curve_p4, label="P4", linewidth=2, color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")
    plt.title("P4 (40×40) Feature Mean")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(loss_curve_p5, label="P5", linewidth=2, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")
    plt.title("P5 (20×20) Feature Mean")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "monitor_curves.png"), dpi=120)
    plt.close()

    print(f"\nTeacher + Detection Head processing completed!")
    print(f"All results saved to: {vis_dir}")
    print(
        f"Final feature means - Teacher: {loss_curve_teacher[-1]:.6f}, P3: {loss_curve_p3[-1]:.6f}, P4: {loss_curve_p4[-1]:.6f}, P5: {loss_curve_p5[-1]:.6f}")


if __name__ == "__main__":
    train_teacher_with_head()