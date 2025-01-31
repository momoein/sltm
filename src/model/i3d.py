import torch
import torch.nn as nn
from typing import Any, Optional, Union

class ConvBlock3D(nn.Module):
    """
    A 3D Convolutional block consisting of Conv3D -> BatchNorm3D -> ReLU.

    Args:
        in_channels (int): Number of input channels for the convolutional layer.
        out_channels (int): Number of output channels for the convolutional layer.
        kernel_size (Union[int, tuple]): Size of the convolutional kernel.
        stride (Union[int, tuple], optional): Stride of the convolution. Defaults to 1.
        padding (Union[int, tuple, str], optional): Padding added to the input. Defaults to 0.
        dilation (Union[int, tuple], optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        padding_mode (str, optional): Padding mode. Defaults to 'zeros'.
        device (Any, optional): Device where the tensor is stored. Defaults to None.
        dtype (Any, optional): Data type of the tensor. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple, str] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        super(ConvBlock3D, self).__init__()

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        self.batchnorm3d = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D Convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.relu(self.batchnorm3d(self.conv3d(x)))



class InceptionBlock3D(nn.Module):
    """
    3D Inception Block with four branches:
        (a) Branch 1: 1x1 Convolution
        (b) Branch 2: 1x1 Convolution -> 3x3 Convolution
        (c) Branch 3: 1x1 Convolution -> 5x5 Convolution
        (d) Branch 4: 3x3 MaxPool -> 1x1 Convolution

    Args:
        in_channels (int): Number of input channels.
        out_1x1 (int): Output channels for branch 1 (1x1 Conv).
        red_3x3 (int): Reduced channels for branch 2 (1x1 Conv before 3x3 Conv).
        out_3x3 (int): Output channels for branch 2 (3x3 Conv).
        red_5x5 (int): Reduced channels for branch 3 (1x1 Conv before 5x5 Conv).
        out_5x5 (int): Output channels for branch 3 (5x5 Conv).
        out_1x1_pooling (int): Output channels for branch 4 (1x1 Conv after MaxPool).
    """

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        out_1x1_pooling: int
    ):
        super(InceptionBlock3D, self).__init__()

        self.branch1 = ConvBlock3D(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            ConvBlock3D(in_channels, red_3x3, kernel_size=1, stride=1, padding=0),
            ConvBlock3D(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock3D(in_channels, red_5x5, kernel_size=1, stride=1, padding=0),
            ConvBlock3D(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            ConvBlock3D(in_channels, out_1x1_pooling, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Inception block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenated output from all branches.
        """
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1
        )



class Inception3D(nn.Module):
    """
    3D Inception Network based on the Inception-V1 architecture.

    Args:
        in_channels (int): Number of input channels.
        num_classes (Optional[int]): Number of classes for classification. If None, no classification head is added.
        p_dropout (float, optional): probability of dropout. Defaults to 0.0
    """

    def __init__(
        self, 
        in_channels: int, 
        num_classes: Optional[int] = None,
        p_dropout: float = 0.0,
    ):
        super(Inception3D, self).__init__()

        self.classification = num_classes is not None

        # Initial layers
        self.conv1 = ConvBlock3D(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        self.conv2 = nn.Sequential(
            ConvBlock3D(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock3D(64, 192, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inception blocks
        self.inception3a = InceptionBlock3D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock3D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock3D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock3D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock3D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock3D(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock3D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock3D(832, 384, 192, 384, 48, 128, 128)

        # Final layers
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
        self.dropout = nn.Dropout(p_dropout)

        if self.classification:
            self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Inception3D network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (logits if classification is True).
        """
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)

        if self.classification:
            x = self.fc1(x)

        return x
