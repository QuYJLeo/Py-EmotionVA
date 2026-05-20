# Q-EmotionVA: 面部表情识别与效价-唤醒度估计 🎭

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 项目概述

Q-EmotionVA 是一个先进的深度学习框架，用于**实时面部表情识别**和**效价-唤醒度估计**。基于最先进的计算机视觉技术，该项目在 AffectNet 数据集上实现了出色的性能。

### ✨ 核心功能

| 功能 | 描述 |
|------|------|
| 😊 **表情分类** | 识别 8 种基本情绪（中性、快乐、悲伤、惊讶、恐惧、厌恶、愤怒、轻蔑） |
| 📊 **效价-唤醒度回归** | 估计连续情绪维度 |
| ⚡ **实时推理** | 高帧率处理视频流 |
| 🚀 **多骨干网络支持** | 支持 MixedFeatureNet、MobileNetV2 和 ShuffleNetV2 |
| 📱 **边缘部署** | 导出模型为 ONNX 格式用于边缘设备 |

---

## 📁 项目结构

```
Q-EmotionVA/
├── 🧠 models/                    # 神经网络架构
│   ├── MixedFeatureNet.py        # 自定义特征提取骨干网络
│   ├── DDAM.py                   # 注意力增强情绪模型
│   ├── DDAM-mbnet.py             # MobileNetV2 变体
│   └── DDAM-shufflenet.py        # ShuffleNetV2 变体
├── 🛠️ tools/                     # 工具脚本
│   ├── affectnet_train.py        # 训练管道
│   ├── affectnet_test.py         # 带混淆矩阵的评估
│   ├── video-test-mediapipe.py   # 实时摄像头演示
│   ├── video-test-onnx.py        # 基于 ONNX 的实时演示
│   ├── pth2onnx.py               # 模型转换工具
│   └── data-handler.py           # 数据集预处理
├── 💾 checkpoints/               # 训练好的模型权重
└── 📦 pretrained/                # 预训练骨干网络
```

---

## 🛠️ 安装

### 前提条件

- **Python 3.8+**
- **PyTorch 1.9+**
- **CUDA 11.0+**（用于 GPU 加速）

### 安装依赖

```bash
pip install torch torchvision numpy pandas opencv-python mediapipe onnxruntime-gpu matplotlib scikit-learn tqdm Pillow
```

### 下载预训练权重

将以下文件放置在 `pretrained/` 目录中：

- [MobileNetV2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) 📥
- [ShuffleNetV2](https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth) 📥

---

## 📊 数据集准备

### AffectNet 数据集

1. 从 [AffectNet 官方网站](http://mohammadmahoor.com/affectnet/) 下载 📥
2. 按以下结构组织：

```
AffectNetDataset/
├── Manually_Annotated/
│   ├── Manually_Annotated_Images/   # 原始图像
│   ├── training.csv                  # 训练标注
│   └── validation.csv                # 验证标注
```

### 预处理数据集

```bash
python tools/data-handler.py
```

**输出:**
- 裁剪后的人脸图像: `tiny_facedetect_filter_annotated_images/` 🖼️
- 标注 JSON 文件: `tiny_facedetect_train_filter.json` 📄

---

## 🏋️ 训练

### 基础训练命令

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
| `--aff_path` | 数据集根路径 | `/data/affectnet/` |
| `--batch_size` | 批次大小 | 10 |
| `--lr` | 学习率 | 0.0001 |
| `--epochs` | 训练轮数 | 40 |
| `--num_head` | 注意力头数量 | 2 |
| `--num_class` | 情绪类别数 | 8 |
| `--workers` | 数据加载线程数 | 0 |

### 训练输出

模型保存在 `checkpoints/` 目录，命名格式：
```
affecnet8_epoch{epoch}_acc{accuracy}.pth
```

---

## 🧪 测试

### 评估模型性能

```bash
python tools/affectnet_test.py \
    --aff_path /path/to/affectnet \
    --model_path checkpoints/affecnet8_epoch15_acc0.5587.pth \
    --num_head 2 \
    --num_class 8
```

### 测试输出

- ✅ 验证准确率
- 📊 混淆矩阵可视化 (`checkpoints/*.png`)

---

## 🎬 实时演示

### PyTorch 摄像头演示

```bash
python tools/video-test-mediapipe.py
```

### ONNX 摄像头演示（更快）

```bash
python tools/video-test-onnx.py
```

### 演示功能

| 功能 | 描述 |
|------|------|
| 🎯 实时人脸检测 | 基于 MediaPipe |
| 📈 表情概率条 | 可视化置信度分数 |
| 📉 效价-唤醒度指示器 | 实时情绪状态追踪 |
| ⌨️ 退出 | 按 `q` 键退出 |

---

## 🔄 模型转换

### 转换为 ONNX 格式

```bash
python tools/pth2onnx.py
```

**输出:** `checkpoints/mp_MFN_epochXX.onnx` 🚀

**应用场景:** 边缘部署、TensorRT 优化、跨平台推理

---

## 🧠 模型架构

### 网络概览

```
输入 (112x112x3)
    ↓
骨干网络 (MixedFeatureNet)
    ↓
特征图 (7x7x512)
    ↓
坐标注意力头
    ↓
特征融合
    ↓
分类头 → 表情概率 (8类)
    ↓
回归头 → 效价, 唤醒度
```

### 注意力机制

注意力模块通过以下方式捕获空间信息：

1. **水平池化** 🔹 - 捕获高度方向模式
2. **垂直池化** 🔸 - 捕获宽度方向模式
3. **通道交互** 🔄 - 融合空间信息
4. **自适应加权** ⚖️ - 应用学习到的注意力

### 损失函数

多任务学习的组合目标：

| 损失组件 | 用途 | 权重 |
|----------|------|------|
| 交叉熵 | 表情分类 | 1.0 |
| 注意力多样性 | 鼓励多样化特征学习 | 0.1 |
| CCC 损失 | 效价回归 | 2.5 |
| CCC 损失 | 唤醒度回归 | 2.5 |

---

## 📈 性能

### AffectNet 结果

| 骨干网络 | 准确率 | 效价 CCC | 唤醒度 CCC |
|----------|--------|----------|------------|
| MixedFeatureNet | **55.87%** | **0.68** | **0.65** |
| MobileNetV2 | 54.23% | 0.66 | 0.63 |
| ShuffleNetV2 | 53.89% | 0.65 | 0.62 |

---

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{Q-EmotionVA,
    title={Q-EmotionVA: Facial Emotion Recognition with Valence-Arousal Estimation},
    author={Your Name},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 🙌 [AffectNet 数据集](http://mohammadmahoor.com/affectnet/)
- 🙌 [MediaPipe](https://mediapipe.dev/)
- 🙌 [PyTorch](https://pytorch.org/)
- 🙌 [MobileNetV2](https://arxiv.org/abs/1801.04381)
- 🙌 [ShuffleNetV2](https://arxiv.org/abs/1807.11164)

---

*❤️ 为情绪 AI 研究而生*
