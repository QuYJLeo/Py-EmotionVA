from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch
import torch.nn as nn

class Flatten(Module):
    """
    Flatten layer that reshapes the input tensor to (batch_size, flattened_features).
    Converts a multi-dimensional tensor into a 2D tensor for fully connected layers.
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    """
    L2 normalization along specified axis.
    Normalizes input tensor to unit norm (L2 norm = 1).

    Args:
        input: Input tensor
        axis: Axis along which to compute norm (default: 1)

    Returns:
        L2-normalized tensor
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Conv_block(Module):
    """
    Convolutional block consisting of Conv2d -> BatchNorm2d -> PReLU.
    A standard building block for MobileNet-style architectures.

    Args:
        in_c: Number of input channels
        out_c: Number of output channels
        kernel: Convolution kernel size (default: (1, 1))
        stride: Convolution stride (default: (1, 1))
        padding: Padding size (default: (0, 0))
        groups: Number of groups for grouped convolution (default: 1)
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        """Forward pass: Conv -> BatchNorm -> PReLU activation."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

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
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        """Forward pass: Conv -> BatchNorm (no activation)."""
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
    """
    Depth-wise separable convolution block.
    Performs pointwise convolution, depthwise convolution, and projection.

    Args:
        in_c: Number of input channels
        out_c: Number of output channels
        residual: Whether to use residual connection (default: False)
        kernel: Depthwise convolution kernel size (default: (3, 3))
        stride: Depthwise convolution stride (default: (2, 2))
        padding: Padding size (default: (1, 1))
        groups: Number of groups for grouping (default: 1)
    """
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        """Forward pass with optional residual connection."""
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x).
    Proposed as a better alternative to ReLU in some contexts.
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass: x * sigmoid(x)."""
        return x * self.sigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}


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

class swish(nn.Module):
    """
    Simplified Swish activation using torch.sigmoid directly.
    Equivalent to x * torch.sigmoid(x).
    """
    def forward(self, x):
        """Forward pass: x * sigmoid(x)."""
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention module.
    Enhances feature representations by capturing spatial coordinate information.
    Uses mean pooling instead of AdaptiveAvgPool2d for ONNX export compatibility.

    Args:
        inp: Number of input channels
        oup: Number of output channels
        groups: Number of groups for channel partitioning (default: 32)
    """
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        """
        Forward pass:
        1. Pool spatial dimensions to get height and width attention vectors
        2. Concatenate and project to get transformation
        3. Split and apply sigmoid attention weights
        4. Apply attention to input features
        """
        identity = x
        n, c, h, w = x.size()

        # Equivalent to AdaptiveAvgPool2d((None, 1)) - pool along width
        x_h = torch.mean(x, dim=3, keepdim=True)   # (N, C, H, 1)

        # Equivalent to AdaptiveAvgPool2d((1, None)) - pool along height
        x_w = torch.mean(x, dim=2, keepdim=True)   # (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)             # (N, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class MDConv(Module):
    """
    Mixed Depthwise Convolution.
    Splits channels into groups and applies different kernel sizes to each group.

    Args:
        channels: Total number of channels
        kernel_size: List of kernel sizes for each group
        split_out_channels: Number of channels for each group
        stride: Convolution stride
    """
    def __init__(self, channels, kernel_size, split_out_channels, stride):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = BatchNorm2d(channels)
        self.prelu = PReLU(channels)

    def forward(self, x):
        """Forward pass: split channels, apply different convolutions, concatenate."""
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class Mix_Depth_Wise(Module):
    """
    Mixed Depth-wise block with Coordinate Attention.
    Combines pointwise conv, mixed depthwise conv, coordinate attention, and projection.

    Args:
        in_c: Number of input channels
        out_c: Number of output channels
        residual: Whether to use residual connection (default: False)
        kernel: Convolution kernel size (default: (3, 3))
        stride: Convolution stride (default: (2, 2))
        padding: Padding size (default: (1, 1))
        groups: Number of groups (default: 1)
        kernel_size: List of kernel sizes for MDConv (default: [3,5,7])
        split_out_channels: Channel split for MDConv (default: [64,32,32])
    """
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, kernel_size=[3,5,7], split_out_channels=[64,32,32]):
        super(Mix_Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = MDConv(channels=groups, kernel_size=kernel_size, split_out_channels=split_out_channels, stride=stride)
        self.CA = CoordAtt(groups, groups)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        """Forward pass with optional residual connection."""
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.CA(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    """
    Residual block consisting of multiple Depth_Wise blocks.
    Stacks Depth_Wise blocks with residual connections.

    Args:
        c: Number of channels
        num_block: Number of Depth_Wise blocks to stack
        groups: Number of groups for grouped convolution
        kernel: Convolution kernel size (default: (3, 3))
        stride: Convolution stride (default: (1, 1))
        padding: Padding size (default: (1, 1))
    """
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        """Forward pass through all residual blocks."""
        return self.model(x)


class Mix_Residual(Module):
    """
    Mixed Residual block consisting of multiple Mix_Depth_Wise blocks.
    Stacks Mix_Depth_Wise blocks with residual connections.

    Args:
        c: Number of channels
        num_block: Number of Mix_Depth_Wise blocks to stack
        groups: Number of groups for grouped convolution
        kernel: Convolution kernel size (default: (3, 3))
        stride: Convolution stride (default: (1, 1))
        padding: Padding size (default: (1, 1))
        kernel_size: List of kernel sizes for MDConv (default: [3,5])
        split_out_channels: Channel split for MDConv (default: [64,64])
    """
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5], split_out_channels=[64,64]):
        super(Mix_Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Mix_Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups, kernel_size=kernel_size, split_out_channels=split_out_channels))
        self.model = Sequential(*modules)

    def forward(self, x):
        """Forward pass through all mixed residual blocks."""
        return self.model(x)


class MixedFeatureNet(Module):
    """
    Mixed Feature Network for facial emotion recognition.
    A MobileNet-style CNN with mixed depthwise convolutions and coordinate attention.
    Produces an L2-normalized embedding vector.

    Args:
        embedding_size: Size of the output embedding vector (default: 256)
        out_h: Height for final spatial pooling (default: 7)
        out_w: Width for final spatial pooling (default: 7)
    """
    def __init__(self, embedding_size=256, out_h=7, out_w=7):
        super(MixedFeatureNet, self).__init__()
        # 112x112 input
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        # 56x56
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Mix_Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, kernel_size=[3,5,7], split_out_channels=[64,32,32])

        # 28x28
        self.conv_3 = Mix_Residual(64, num_block=9, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5], split_out_channels=[96,32])
        self.conv_34 = Mix_Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, kernel_size=[3,5,7], split_out_channels=[128,64,64])

        # 14x14
        self.conv_4 = Mix_Residual(128, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5], split_out_channels=[192,64])
        self.conv_45 = Mix_Depth_Wise(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512*2, kernel_size=[3,5,7,9], split_out_channels=[128*2,128*2,128*2,128*2])

        # 7x7
        self.conv_5 = Mix_Residual(256, num_block=6, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5,7], split_out_channels=[86*2,85*2,85*2])
        self.conv_6_sep = Conv_block(256, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        """
        Forward pass through the entire network.
        Returns L2-normalized embedding vector.
        """
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)

        return l2_norm(out)
