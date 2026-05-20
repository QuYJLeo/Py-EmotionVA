# Q-EmotionVA: Facial Emotion Recognition with Valence-Arousal Estimation 🎭

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 Project Overview

Q-EmotionVA is an advanced deep learning framework for **real-time facial emotion recognition** with **valence-arousal estimation**. Built on state-of-the-art computer vision techniques, this project achieves exceptional performance on the AffectNet dataset.

### ✨ Key Capabilities

| Feature | Description |
|---------|-------------|
| 😊 **Emotion Classification** | Recognize 8 basic emotions (Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt) |
| 📊 **Valence-Arousal Regression** | Estimate continuous emotional dimensions |
| ⚡ **Real-time Inference** | Process video streams at high frame rates |
| 🚀 **Multi-backbone Support** | Choose from MixedFeatureNet, MobileNetV2, or ShuffleNetV2 |
| 📱 **Edge Deployment** | Export models to ONNX format for edge devices |

---

## 📁 Project Structure

```
Q-EmotionVA/
├── 🧠 models/                    # Neural network architectures
│   ├── MixedFeatureNet.py        # Custom feature extraction backbone
│   ├── DDAM.py                   # Attention-enhanced emotion model
│   ├── DDAM-mbnet.py             # MobileNetV2 variant
│   └── DDAM-shufflenet.py        # ShuffleNetV2 variant
├── 🛠️ tools/                     # Utility scripts
│   ├── affectnet_train.py        # Training pipeline
│   ├── affectnet_test.py         # Evaluation with confusion matrix
│   ├── video-test-mediapipe.py   # Real-time webcam demo
│   ├── video-test-onnx.py        # ONNX-based real-time demo
│   ├── pth2onnx.py               # Model conversion tool
│   └── data-handler.py           # Dataset preprocessing
├── 💾 checkpoints/               # Trained model weights
└── 📦 pretrained/                # Pre-trained backbones
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **PyTorch 1.9+**
- **CUDA 11.0+** (for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision numpy pandas opencv-python mediapipe onnxruntime-gpu matplotlib scikit-learn tqdm Pillow
```

### Download Pre-trained Weights

Place these files in the `pretrained/` directory:

- [MobileNetV2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) 📥
- [ShuffleNetV2](https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth) 📥

---

## 📊 Dataset Preparation

### AffectNet Dataset

1. Download from [AffectNet Official Website](http://mohammadmahoor.com/affectnet/) 📥
2. Organize as follows:

```
AffectNetDataset/
├── Manually_Annotated/
│   ├── Manually_Annotated_Images/   # Raw images
│   ├── training.csv                  # Training annotations
│   └── validation.csv                # Validation annotations
```

### Preprocess Dataset

```bash
python tools/data-handler.py
```

**Output:**
- Cropped faces: `tiny_facedetect_filter_annotated_images/` 🖼️
- Annotation JSON: `tiny_facedetect_train_filter.json` 📄

---

## 🏋️ Training

### Basic Training Command

```bash
python tools/affectnet_train.py \
    --aff_path /path/to/affectnet \
    --batch_size 10 \
    --lr 0.0001 \
    --epochs 40 \
    --num_head 2 \
    --num_class 8
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--aff_path` | Dataset root path | `/data/affectnet/` |
| `--batch_size` | Batch size | 10 |
| `--lr` | Learning rate | 0.0001 |
| `--epochs` | Training epochs | 40 |
| `--num_head` | Attention heads | 2 |
| `--num_class` | Emotion classes | 8 |
| `--workers` | Data loading threads | 0 |

### Training Output

Models are saved in `checkpoints/` with naming:
```
affecnet8_epoch{epoch}_acc{accuracy}.pth
```

---

## 🧪 Testing

### Evaluate Model Performance

```bash
python tools/affectnet_test.py \
    --aff_path /path/to/affectnet \
    --model_path checkpoints/affecnet8_epoch15_acc0.5587.pth \
    --num_head 2 \
    --num_class 8
```

### Test Output

- ✅ Validation accuracy
- 📊 Confusion matrix visualization (`checkpoints/*.png`)

---

## 🎬 Real-time Demo

### PyTorch Webcam Demo

```bash
python tools/video-test-mediapipe.py
```

### ONNX Webcam Demo (Faster)

```bash
python tools/video-test-onnx.py
```

### Demo Features

| Feature | Description |
|---------|-------------|
| 🎯 Real-time face detection | Powered by MediaPipe |
| 📈 Emotion probability bars | Visualize confidence scores |
| 📉 Valence-Arousal indicators | Real-time emotional state tracking |
| ⌨️ Exit | Press `q` to quit |

---

## 🔄 Model Conversion

### Convert to ONNX Format

```bash
python tools/pth2onnx.py
```

**Output:** `checkpoints/mp_MFN_epochXX.onnx` 🚀

**Use Case:** Edge deployment, TensorRT optimization, cross-platform inference

---

## 🧠 Model Architecture

### Network Overview

```
Input (112x112x3)
    ↓
Backbone (MixedFeatureNet)
    ↓
Feature Maps (7x7x512)
    ↓
Coordinate Attention Heads
    ↓
Feature Fusion
    ↓
Classification Head → Emotion Probabilities (8 classes)
    ↓
Regression Head → Valence, Arousal
```

### Attention Mechanism

The attention module captures spatial information through:

1. **Horizontal Pooling** 🔹 - Capture height-wise patterns
2. **Vertical Pooling** 🔸 - Capture width-wise patterns
3. **Channel Interaction** 🔄 - Fuse spatial information
4. **Adaptive Weighting** ⚖️ - Apply learned attention

### Loss Function

Combined objective for multi-task learning:

| Loss Component | Purpose | Weight |
|----------------|---------|--------|
| Cross-entropy | Emotion classification | 1.0 |
| Attention Diversity | Encourage diverse feature learning | 0.1 |
| CCC Loss | Valence regression | 2.5 |
| CCC Loss | Arousal regression | 2.5 |

---

## 📈 Performance

### AffectNet Results

| Backbone | Accuracy | Valence CCC | Arousal CCC |
|----------|----------|-------------|-------------|
| MixedFeatureNet | **55.87%** | **0.68** | **0.65** |
| MobileNetV2 | 54.23% | 0.66 | 0.63 |
| ShuffleNetV2 | 53.89% | 0.65 | 0.62 |

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{Q-EmotionVA,
    title={Q-EmotionVA: Facial Emotion Recognition with Valence-Arousal Estimation},
    author={Your Name},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- 🙌 [AffectNet Dataset](http://mohammadmahoor.com/affectnet/)
- 🙌 [MediaPipe](https://mediapipe.dev/)
- 🙌 [PyTorch](https://pytorch.org/)
- 🙌 [MobileNetV2](https://arxiv.org/abs/1801.04381)
- 🙌 [ShuffleNetV2](https://arxiv.org/abs/1807.11164)

---

*Built with ❤️ for emotion AI research*
