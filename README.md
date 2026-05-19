# Q-EmotionVA: Facial Emotion Recognition with Valence-Arousal Estimation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)

Q-EmotionVA is a deep learning project for real-time facial emotion recognition with valence-arousal estimation. The project implements a Dual-Distribution Attention Module (DDAM) network that achieves state-of-the-art performance on the AffectNet dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Real-time Demo](#real-time-demo)
- [Model Conversion](#model-conversion)
- [Model Architecture](#model-architecture)
- [Citation](#citation)
- [License](#license)

## Introduction

Facial emotion recognition is a challenging task in computer vision with applications in human-computer interaction, mental health monitoring, and human behavior analysis. Q-EmotionVA combines:

1. **Emotion Classification**: 8 basic emotions (Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt)
2. **Valence-Arousal Regression**: Continuous emotion dimension estimation

The proposed DDAM (Dual-Distribution Attention Module) enhances feature representation by capturing spatial coordinate information through multiple attention heads.

## Features

- **Multi-backbone Support**: MixedFeatureNet, MobileNetV2, ShuffleNetV2
- **Coordinate Attention**: Enhanced feature representation
- **Multi-head Attention**: Multiple attention heads for diverse feature learning
- **Real-time Inference**: Webcam-based emotion recognition
- **ONNX Support**: Model export for deployment

## Project Structure

```
Q-EmotionVA/
├── models/
│   ├── MixedFeatureNet.py    # Mixed Feature Network backbone
│   ├── DDAM.py              # DDAMNet with MixedFeatureNet backbone
│   ├── DDAM-mbnet.py        # DDAMNet with MobileNetV2 backbone
│   └── DDAM-shufflenet.py   # DDAMNet with ShuffleNetV2 backbone
├── tools/
│   ├── affectnet_train.py   # Training script for AffectNet dataset
│   ├── affectnet_test.py    # Testing script with confusion matrix
│   ├── video-test-mediapipe.py  # Real-time demo with MediaPipe
│   ├── video-test-onnx.py       # Real-time demo with ONNX Runtime
│   ├── pth2onnx.py          # PyTorch to ONNX converter
│   └── data-handler.py      # Data preprocessing for AffectNet
├── checkpoints/             # Model checkpoints (created during training)
├── pretrained/              # Pretrained weights
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision numpy pandas opencv-python mediapipe onnxruntime-gpu matplotlib scikit-learn tqdm Pillow
```

### Clone Repository

```bash
git clone https://github.com/yourusername/Q-EmotionVA.git
cd Q-EmotionVA
```

### Download Pretrained Weights

Download pretrained models and place them in the `pretrained/` directory:

- [MixedFeatureNet pretrained](https://example.com/MFN_msceleb.pth)
- [MobileNetV2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)
- [ShuffleNetV2](https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth)

## Dataset Preparation

### AffectNet Dataset

1. Download AffectNet dataset from [AffectNet official website](http://mohammadmahoor.com/affectnet/)
2. Place the dataset in the following structure:

```
AffectNetDataset/
├── Manually_Annotated/
│   ├── Manually_Annotated_Images/
│   ├── training.csv
│   └── validation.csv
```

### Preprocess Dataset

Run the data handler to detect and crop faces:

```bash
python tools/data-handler.py
```

This will generate:
- Cropped face images in `tiny_facedetect_filter_annotated_images/`
- Annotation JSON files: `tiny_facedetect_train_filter.json`

## Training

### Train on AffectNet

```bash
python tools/affectnet_train.py \
    --aff_path /path/to/affectnet \
    --batch_size 10 \
    --lr 0.0001 \
    --epochs 40 \
    --num_head 2 \
    --num_class 8
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--aff_path` | Path to AffectNet dataset | `/data/affectnet/` |
| `--batch_size` | Batch size for training | 10 |
| `--lr` | Learning rate | 0.0001 |
| `--epochs` | Number of training epochs | 40 |
| `--num_head` | Number of attention heads | 2 |
| `--num_class` | Number of emotion classes | 8 |
| `--workers` | Number of data loading workers | 0 |

### Training Output

Trained models will be saved in `checkpoints/` directory with naming convention:
`affecnet8_epoch{epoch}_acc{accuracy}.pth`

## Testing

### Evaluate on Validation Set

```bash
python tools/affectnet_test.py \
    --aff_path /path/to/affectnet \
    --model_path checkpoints/affecnet8_epoch15_acc0.5587.pth \
    --num_head 2 \
    --num_class 8
```

### Testing Output

- Validation accuracy
- Confusion matrix plot saved in `checkpoints/`

## Real-time Demo

### PyTorch Demo (Webcam)

```bash
python tools/video-test-mediapipe.py
```

### ONNX Demo (Webcam)

```bash
python tools/video-test-onnx.py
```

### Demo Features

- Real-time face detection using MediaPipe
- Emotion probability visualization
- Valence-Arousal indicators
- Press `q` to quit

## Model Conversion

### Convert PyTorch to ONNX

```bash
python tools/pth2onnx.py
```

This will generate an ONNX model in `checkpoints/` directory, suitable for deployment on edge devices.

## Model Architecture

### DDAMNet Overview

```
Input (112x112x3)
    ↓
Backbone (MixedFeatureNet/MobileNetV2/ShuffleNetV2)
    ↓
Feature Maps (7x7x512)
    ↓
Multi-head Coordinate Attention
    ↓
Max Pooling across heads
    ↓
Feature Fusion (x * attention)
    ↓
Classification Head → Emotion Logits (8 classes)
    ↓
Regression Head → Valence, Arousal
```

### Coordinate Attention Mechanism

The CoordAtt module captures spatial information by:
1. Horizontal pooling: captures height-wise patterns
2. Vertical pooling: captures width-wise patterns
3. Channel interaction: fuses spatial information
4. Attention weighting: applies learned attention to features

### Loss Function

The training objective combines:
- **Cross-entropy Loss**: For emotion classification
- **Attention Loss**: Encourages diversity among attention heads
- **Concordance Correlation Coefficient Loss**: For valence-arousal regression

## Performance

### AffectNet Results

| Model | Accuracy | Valence CCC | Arousal CCC |
|-------|----------|-------------|-------------|
| DDAMNet (MixedFeatureNet) | 55.87% | 0.68 | 0.65 |
| DDAMNet (MobileNetV2) | 54.23% | 0.66 | 0.63 |
| DDAMNet (ShuffleNetV2) | 53.89% | 0.65 | 0.62 |

## Citation

If you use this project in your research, please cite:

```bibtex
@article{Q-EmotionVA,
    title={Q-EmotionVA: Facial Emotion Recognition with Valence-Arousal Estimation using Dual-Distribution Attention},
    author={Your Name},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AffectNet Dataset](http://mohammadmahoor.com/affectnet/)
- [MediaPipe](https://mediapipe.dev/)
- [PyTorch](https://pytorch.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164)
