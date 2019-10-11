import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Separable Conv2d
    """
    def __init__(self, in_channels, out_channels, dw_kernel_size, dw_stride,
                 dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
                                          kernel_size=dw_kernel_size,
                                          stride=dw_stride, padding=dw_padding,
                                          groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class MaxPool(nn.Module):
    """
    Max Pooling
    """
    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:]
        return x
