from torch import nn
import torch
import torchvision.models as models
from models import MixedFeatureNet
from torch.nn import Module
from torchvision.models import shufflenet_v2_x1_0
import os

class Linear_block(Module):
    """
    Linear convolutional block consisting of Conv2d -> BatchNorm2d.
    Used for pointwise convolution without activation function.

    Args:
        in_c: Number of input channels
        out_c: Number of output channels
        kernel: Convolution kernel size (default: (1, 1))
        stride: Convolution stride (default: (1, 1))
        padding: Padding size (default: (0, 0))
        groups: Number of groups for grouped convolution (default: 1)
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        """Forward pass: Conv -> BatchNorm (no activation)."""
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(Module):
    """
    Flatten layer that reshapes the input tensor to (batch_size, flattened_features).
    Converts a multi-dimensional tensor into a 2D tensor for fully connected layers.
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class DDAMNet(nn.Module):
    """
    Dual-Distribution Attention Module Network with ShuffleNetV2 backbone.
    Uses pretrained ShuffleNetV2 as feature extractor with multiple attention heads.

    Args:
        num_class: Number of emotion classes (default: 7)
        num_head: Number of attention heads (default: 2)
        pretrained: Whether to load pretrained ShuffleNetV2 weights (default: True)
    """
    def __init__(self, num_class=7, num_head=2, pretrained=True):
        super(DDAMNet, self).__init__()

        # ShuffleNetV2 backbone
        net = shufflenet_v2_x1_0(pretrained=False)
        if pretrained:
            net.load_state_dict(torch.load(r"./pretrained/shufflenetv2_x1-5666bf0f80.pth"))

        # Remove the final linear layer (classifier)
        new_model = nn.Sequential()
        for module in net.children():
            if isinstance(module, nn.Linear):
                ...  # Skip linear layer
            else:
                new_model.append(module)

        # Additional convolution layer to reduce channels from 1024 to 512
        new_model.append(nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ))
        self.features = new_model

        self.num_head = num_head
        for i in range(int(num_head)):
            setattr(self,"cat_head%d" %(i), CoordAttHead())

        # Emotion classification head
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_class)

        # Valence-Arousal regression head
        self.Linear2 = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten2 = Flatten()
        self.fc2 = nn.Linear(512, 2)  # 2 for valence and arousal


    def forward(self, x):
        """
        Forward pass through the network.

        Returns:
            out: Emotion classification logits (batch_size, num_class)
            x: Feature maps after backbone (batch_size, 512, 7, 7)
            head_out: List of attention head outputs
            out2: Valence-Arousal predictions (batch_size, 2)
        """
        x = self.features(x)  # [batch, 512, 7, 7]
        heads = []

        # Process through each attention head
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        head_out = heads

        # Max pooling across attention heads
        y = heads[0]
        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])

        # Apply attention weights to features
        y = x * y  # [batch, 512, 7, 7]

        # Classification and regression outputs
        out = self.fc(self.flatten(self.Linear(y)))
        out2 = self.fc2(self.flatten2(self.Linear2(y)))

        return out, x, head_out, out2

class h_sigmoid(nn.Module):
    """
    Hard Sigmoid activation: ReLU6(x + 3) / 6.
    A computationally cheaper alternative to Sigmoid.

    Args:
        inplace: Whether to perform operation in-place (default: True)
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """Forward pass: ReLU6(x + 3) / 6."""
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """
    Hard Swish activation: x * h_sigmoid(x).
    Computationally cheaper version of Swish.

    Args:
        inplace: Whether to perform operation in-place (default: True)
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        """Forward pass: x * h_sigmoid(x)."""
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    """
    Coordinate Attention Head wrapper.
    Wraps CoordAtt module for use in DDAMNet.

    Args:
        None (fixed to 512 input/output channels)
    """
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512, 512)

    def forward(self, x):
        """Forward pass through coordinate attention."""
        ca = self.CoordAtt(x)
        return ca

class CoordAtt(nn.Module):
    """
    Coordinate Attention module.
    Enhances feature representations by capturing spatial coordinate information
    through horizontal and vertical pooling paths.

    Args:
        inp: Number of input channels
        oup: Number of output channels
        groups: Number of groups for channel partitioning (default: 32)
    """
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()

        # Horizontal and vertical pooling paths
        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))

        mip = max(8, inp // groups)

        # Channel reduction convolution
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()

    def forward(self, x):
        """
        Forward pass:
        1. Pool features along horizontal and vertical dimensions
        2. Concatenate and project to get transformation
        3. Split and apply sigmoid attention weights
        4. Apply attention to input features
        """
        identity = x
        n, c, h, w = x.size()

        # Horizontal pooling: captures height information
        x_h = self.Linear_h(x)
        # Vertical pooling: captures width information
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        # Concatenate pooled features
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # Split attention weights
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        # Expand attention maps to original spatial size
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        # Apply attention weights
        y = x_w * x_h

        return y
