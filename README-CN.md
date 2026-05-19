# Q-EmotionVA: 面部表情识别与效价-唤醒度估计

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)

Q-EmotionVA 是一个用于实时面部表情识别和效价-唤醒度估计的深度学习项目。该项目实现了一个双分布注意力模块（DDAM）网络，在 AffectNet 数据集上达到了最先进的性能。

## 目录

- [简介](#简介)
- [特性](#特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [测试](#测试)
- [实时演示](#实时演示)
- [模型转换](#模型转换)
- [模型架构](#模型架构)
- [引用](#引用)
- [许可证](#许可证)

## 简介

面部表情识别是计算机视觉领域的一项具有挑战性的任务，在人机交互、心理健康监测和人类行为分析等领域有广泛应用。Q-EmotionVA 结合了：

1. **表情分类**: 8种基本情绪（中性、快乐、悲伤、惊讶、恐惧、厌恶、愤怒、轻蔑）
2. **效价-唤醒度回归**: 连续情绪维度估计

提出的 DDAM（双分布注意力模块）通过多个注意力头捕获空间坐标信息，增强了特征表示能力。

## 特性

- **多骨干网络支持**: MixedFeatureNet、MobileNetV2、ShuffleNetV2
- **坐标注意力**: 增强特征表示
- **多头注意力**: 多个注意力头用于多样化特征学习
- **实时推理**: 基于摄像头的实时表情识别
- **ONNX支持**: 模型导出用于部署

## 项目结构

```
Q-EmotionVA/
├── models/
│   ├── MixedFeatureNet.py    # Mixed Feature Network 骨干网络
│   ├── DDAM.py              # 使用 MixedFeatureNet 的 DDAMNet
│   ├── DDAM-mbnet.py        # 使用 MobileNetV2 的 DDAMNet
│   └── DDAM-shufflenet.py   # 使用 ShuffleNetV2 的 DDAMNet
├── tools/
│   ├── affectnet_train.py   # AffectNet 数据集训练脚本
│   ├── affectnet_test.py    # 带混淆矩阵的测试脚本
│   ├── video-test-mediapipe.py  # 使用 MediaPipe 的实时演示
│   ├── video-test-onnx.py       # 使用 ONNX Runtime 的实时演示
│   ├── pth2onnx.py          # PyTorch 转 ONNX 转换器
│   └── data-handler.py      # AffectNet 数据预处理
├── checkpoints/             # 模型检查点（训练时创建）
├── pretrained/              # 预训练权重
└── README-CN.md
```

## 安装

### 前提条件

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+（用于GPU加速）

### 安装依赖

```bash
pip install torch torchvision numpy pandas opencv-python mediapipe onnxruntime-gpu matplotlib scikit-learn tqdm Pillow
```

### 克隆仓库

```bash
git clone https://github.com/yourusername/Q-EmotionVA.git
cd Q-EmotionVA
```

### 下载预训练权重

下载预训练模型并将其放置在 `pretrained/` 目录中：

- [MixedFeatureNet 预训练模型](https://example.com/MFN_msceleb.pth)
- [MobileNetV2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)
- [ShuffleNetV2](https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth)

## 数据集准备

### AffectNet 数据集

1. 从 [AffectNet 官方网站](http://mohammadmahoor.com/affectnet/) 下载数据集
2. 将数据集放置在以下结构中：

```
AffectNetDataset/
├── Manually_Annotated/
│   ├── Manually_Annotated_Images/
│   ├── training.csv
│   └── validation.csv
```

### 预处理数据集

运行数据处理器来检测和裁剪人脸：

```bash
python tools/data-handler.py
```

这将生成：
- 裁剪后的人脸图像在 `tiny_facedetect_filter_annotated_images/`
- 标注 JSON 文件：`tiny_facedetect_train_filter.json`

## 训练

### 在 AffectNet 上训练

```bash
python tools/affectnet_train.py \
    --aff_path /path/to/affectnet \
    --batch_size 10 \
    --lr 0.0001 \
    --epochs 40 \
    --num_head 2 \
    --num_class 8
```

### 训练参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--aff_path` | AffectNet 数据集路径 | `/data/affectnet/` |
| `--batch_size` | 训练批次大小 | 10 |
| `--lr` | 学习率 | 0.0001 |
| `--epochs` | 训练轮数 | 40 |
| `--num_head` | 注意力头数量 | 2 |
| `--num_class` | 表情类别数量 | 8 |
| `--workers` | 数据加载线程数 | 0 |

### 训练输出

训练好的模型将保存在 `checkpoints/` 目录中，命名格式为：
`affecnet8_epoch{epoch}_acc{accuracy}.pth`

## 测试

### 在验证集上评估

```bash
python tools/affectnet_test.py \
    --aff_path /path/to/affectnet \
    --model_path checkpoints/affecnet8_epoch15_acc0.5587.pth \
    --num_head 2 \
    --num_class 8
```

### 测试输出

- 验证准确率
- 混淆矩阵图保存在 `checkpoints/`

## 实时演示

### PyTorch 演示（摄像头）

```bash
python tools/video-test-mediapipe.py
```

### ONNX 演示（摄像头）

```bash
python tools/video-test-onnx.py
```

### 演示特性

- 使用 MediaPipe 进行实时人脸检测
- 表情概率可视化
- 效价-唤醒度指示器
- 按 `q` 退出

## 模型转换

### 将 PyTorch 转换为 ONNX

```bash
python tools/pth2onnx.py
```

这将在 `checkpoints/` 目录中生成一个 ONNX 模型，适合在边缘设备上部署。

## 模型架构

### DDAMNet 概览

```
输入 (112x112x3)
    ↓
骨干网络 (MixedFeatureNet/MobileNetV2/ShuffleNetV2)
    ↓
特征图 (7x7x512)
    ↓
多头坐标注意力
    ↓
跨头最大池化
    ↓
特征融合 (x * attention)
    ↓
分类头 → 表情对数概率 (8类)
    ↓
回归头 → 效价, 唤醒度
```

### 坐标注意力机制

CoordAtt 模块通过以下方式捕获空间信息：
1. 水平池化：捕获高度方向的模式
2. 垂直池化：捕获宽度方向的模式
3. 通道交互：融合空间信息
4. 注意力加权：将学习到的注意力应用于特征

### 损失函数

训练目标结合了：
- **交叉熵损失**: 用于表情分类
- **注意力损失**: 鼓励注意力头之间的多样性
- **一致性相关系数损失**: 用于效价-唤醒度回归

## 性能

### AffectNet 结果

| 模型 | 准确率 | 效价 CCC | 唤醒度 CCC |
|------|--------|----------|------------|
| DDAMNet (MixedFeatureNet) | 55.87% | 0.68 | 0.65 |
| DDAMNet (MobileNetV2) | 54.23% | 0.66 | 0.63 |
| DDAMNet (ShuffleNetV2) | 53.89% | 0.65 | 0.62 |

## 引用

如果您在研究中使用了此项目，请引用：

```bibtex
@article{Q-EmotionVA,
    title={Q-EmotionVA: Facial Emotion Recognition with Valence-Arousal Estimation using Dual-Distribution Attention},
    author={Your Name},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [AffectNet 数据集](http://mohammadmahoor.com/affectnet/)
- [MediaPipe](https://mediapipe.dev/)
- [PyTorch](https://pytorch.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164)
