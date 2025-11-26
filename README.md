# 图像分割基线与衍生模型

本仓库整理了用于视网膜血管分割（DRIVE 数据集）的多种 U-Net 系列模型实现，覆盖经典 U-Net、UNet++、支持 Task-Adaptive Mixture of Skip Connections (TA-MoSC) 的 UTANet，以及进一步结合残差编码器与 MoE 跳连的 ReMoE-UNet。项目提供从数据准备、训练到推理评估的完整脚本，方便横向对比不同架构的效果和代价。

## 目录结构

```

image-segmentation/

├── dataset.py                # DRIVE 数据增广脚本

├── unet/                     # 经典 U-Net

├── unet_pp/                  # UNet++

├── utanet/                   # UTANet（TA-MoSC 版本）

├── Re_unet/                  # ReMoE-UNet（残差+MoE）

└── data/                     # 运行后生成的 train/test 图像与掩码

```

每个模型子目录都包含 `train.py`、`test.py`、`loss.py`、`preprocess.py`、`utils.py` 以及可视化结果 (`files/`、`results/`)。

## 模型概览

-**U-Net (`unet/`)**

  经典编码器-解码器结构，4 层下采样+上采样，对 512×512 彩色图像输出单通道概率图。

-**UNet++ (`unet_pp/`)**

  在解码侧构建稠密跳连网格，使用多级节点逐步聚合不同尺度特征，提升细粒度复原。

-**UTANet (`utanet/`)**

  在 U-Net 跳连处插入 TA-MoSC 模块：共享 Mixture-of-Experts 对 skip feature 做动态自适应重标定，训练时需将辅助 MoE loss 累加到主损失。

-**ReMoE-UNet (`Re_unet/`)**

  以残差块替换基本卷积模块，在编码器/解码器中保持恒定深度；skip 侧同样使用 TA-MoSC，并额外引入残差瓶颈，适合更深或更难的场景。

各模型训练脚本均支持 `DiceBCELoss`、自适应学习率调度以及训练/验证曲线绘制。

## 环境准备

1. 安装 Python 3.9+（建议 3.10）。
2. 创建并激活虚拟环境（conda 或 venv）。
3. 安装依赖：

   ```bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 需按显卡环境调整

   pip install albumentations==1.3.1opencv-python==4.10.0.84imageio==2.35.1\

               matplotlib==3.9.0 numpy==1.26.4 scikit-learn==1.5.1 tqdm==4.66.4

   ```

> 如果只在 CPU 上运行，将 `pip install torch` 部分替换为官网提供的 CPU 版本即可。

## 数据准备

1. 从 [DRIVE 官方站点](https://drive.grand-challenge.org/) 下载数据，解压到 `DRIVE/`。
2. 运行 `dataset.py` 以生成网络输入尺寸一致的 PNG，并执行基础增广：

   ```bash

   python dataset.py

   ```

   脚本会在 `data/train/` 与 `data/test/` 下生成 `image/` 与 `mask/` 两个子文件夹。训练阶段默认使用翻转与旋转扩增，测试集仅做尺度对齐。

## 训练流程

所有模型的训练脚本结构一致，可按以下步骤使用（以 U-Net 为例）：

```bash

cdunet

pythontrain.py

```

通用超参数（可在各自 `train.py` 中修改）：

- 输入尺寸 `512×512`
- 批大小 2（受限于 24GB 显存以内的卡）
- 训练轮数 50
- 优化器 Adam，初始学习率 1e-4，`ReduceLROnPlateau` 自动调节
- 评估指标：DiceBCELoss、像素级准确率

训练完成后会在当前子目录生成：

-`files/checkpoint.pth`：验证集最佳权重

-`files/<model>_train_curves.png`：Loss/Accuracy 曲线

### 不同架构的启动命令

| 模型           | 训练命令              | 说明 |

| -------------- | --------------------- | ---- |

| U-Net          | `cd unet && python train.py` | 基线 |

| UNet++         | `cd unet_pp && python train.py` | 稠密跳连版本 |

| UTANet         | `cd utanet && python train.py` | 训练时自动叠加 TA-MoSC 辅助损失 |

| ReMoE-UNet     | `cd Re_unet && python train.py` | 残差+MoE 结构 |

## 推理与评估

1. 确保 `files/checkpoint.pth` 已生成。
2. 运行对应目录下的 `test.py`：

   ```bash

   cd unet

   python test.py

   ```

脚本将：

- 在 `results/` 中保存拼接图（原图 / GT / 预测）。
- 输出 Jaccard、F1、Recall、Precision、Accuracy、FPS 等指标。
- 生成 `results/<model>_test_curves.png` 记录逐样本 loss/acc。

UTANet、ReMoE-UNet 在推理阶段会自动丢弃辅助损失，仅使用主输出。

## 常见自定义点

-**分辨率**：如需处理其他尺寸，可同时修改 `dataset.py` 中的 `size` 以及 `train.py/test.py` 的 `H/W`。

-**多类别分割**：将网络输出通道改为类别数、把 `DiceBCELoss` 替换为交叉熵或 Focal Loss，并调整 `binary_accuracy`。

-**数据集替换**：更改 `data/*` 中的图像与掩码路径即可，保持文件名一一对应。

-**MoE 配置**：UTANet、ReMoE-UNet 的 `moe_embed_dim`、`num_experts`、`top_k` 在网络构造函数中暴露，可依据显存与任务复杂度调整。

## 结果展示

-`*/files/*_train_curves.png`：训练/验证损失与准确率。

-`*/results/*.png`：示例预测（输入、真值、预测并排）。

实际指标依赖于数据划分与随机种子，推荐在同等训练轮数下对比不同模型的 `curves` 与 `results` 文件夹。
