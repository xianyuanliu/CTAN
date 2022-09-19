# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Python implementation of Squeeze-and-Excitation Layers (SELayer)
Initial implementation: channel-wise (SELayerC)
Improved implementation: temporal-wise (SELayerT), convolution-based channel-wise (SELayerCoC), max-pooling-based
channel-wise (SELayerMC), multi-pooling-based channel-wise (SELayerMAC)

[Redundancy and repeat of code will be reduced in the future.]

References:
    Hu Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In CVPR, pp. 7132-7141. 2018.
    For initial implementation, please go to https://github.com/hujie-frank/SENet
"""

import torch
import torch.nn as nn


def get_selayer(attention):
    """Get SELayers referring to attention.

    Args:
        attention (string): the name of the SELayer.
            (Options: ["SELayerC", "SELayerT", "SRMLayerVideo", "CSAMLayer", "STAMLayer",
            "SELayerCoC", "SELayerMC", "SELayerMAC"])

    Returns:
        se_layer (SELayer, optional): the SELayer.
    """

    if attention == "SELayerC":
        se_layer = SELayerC
    elif attention == "SELayerT":
        se_layer = SELayerT
    elif attention == "SRMLayerVideo":
        se_layer = SRMLayerVideo
    elif attention == "CSAMLayer":
        se_layer = CSAMLayer
    elif attention == "STAMLayer":
        se_layer = STAMLayer
    elif attention == "SELayerCoC":
        se_layer = SELayerCoC
    elif attention == "SELayerMC":
        se_layer = SELayerMC
    elif attention == "SELayerMAC":
        se_layer = SELayerMAC
    elif attention == "SELayerCoC":
        se_layer = SELayerCoC

    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))
    return se_layer


class SELayer(nn.Module):
    """Helper class for SELayer design."""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = reduction

    def forward(self, x):
        return NotImplementedError()


class SRMLayer(SELayer):
    """Construct Style-based Recalibration Module for images.

    References:
        Lee, HyunJae, Hyo-Eun Kim, and Hyeonseob Nam. "Srm: A style-based recalibration module for convolutional neural
        networks." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1854-1862. 2019.
    """

    def __init__(self, channel, reduction=16):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__(channel, reduction)

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(self.channel, self.channel, kernel_size=2, bias=False, groups=self.channel)
        self.bn = nn.BatchNorm1d(self.channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.view(b, c, 1, 1)
        # out = x * g.expand_as(x)
        out = x + x * g.expand_as(x)
        return out


class SRMLayerVideo(SELayer):
    def __init__(self, channel, reduction=16):
        super(SRMLayerVideo, self).__init__(channel, reduction)
        self.cfc = nn.Conv1d(self.channel, self.channel, kernel_size=2, bias=False, groups=self.channel)
        self.bn = nn.BatchNorm1d(self.channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.view(b, c, 1, 1, 1)
        out = x + x * g.expand_as(x)
        return out


class SELayerC(SELayer):
    """Construct channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerC, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerT(SELayer):
    """Construct temporal-wise SELayer."""

    def __init__(self, channel, reduction=2):
        super(SELayerT, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, _, t, _, _ = x.size()
        output = x.transpose(1, 2).contiguous()
        y = self.avg_pool(output).view(b, t)
        y = self.fc(y).view(b, t, 1, 1, 1)
        y = y.transpose(1, 2).contiguous()
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class CSAMLayer(nn.Module):
    """Construct Channel-Spatial Attention Module. This module [2] extends CBAM [1] by apply 3D layers.

    References:
        [1] Woo, Sanghyun, Jongchan Park, Joon-Young Lee, and In So Kweon. "Cbam: Convolutional block attention
        module." In Proceedings of the European conference on computer vision (ECCV), pp. 3-19. 2018.
        [2] Yi, Ziwen, Zhonghua Sun, Jinchao Feng, and Kebin Jia. "3D Residual Networks with Channel-Spatial Attention
        Module for Action Recognition." In 2020 Chinese Automation Congress (CAC), pp. 5171-5174. IEEE, 2020.
    """

    def __init__(self, channel, reduction=16):
        super(CSAMLayer, self).__init__()
        self.CAM = CSAMChannelModule(channel, reduction)
        self.SAM = CSAMSpatialModule()

    def forward(self, x):
        y = self.CAM(x)
        y = self.SAM(y)
        return y


class CSAMChannelModule(SELayer):
    def __init__(self, channel, reduction=16):
        super(CSAMChannelModule, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)
        y = torch.add(y_avg, y_max)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class CSAMSpatialModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(CSAMSpatialModule, self).__init__()
        self.kernel_size = kernel_size
        self.compress = CSAMChannelPool()
        self.conv = nn.Conv3d(2, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        y = self.conv(x_compress)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class CSAMChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class STAMLayer(SELayer):
    """Construct Spatial-temporal Attention Module.

    References:
        Zhou, Shengwei, Liang Bai, Haoran Wang, Zhihong Deng, Xiaoming Zhu, and Cheng Gong. "A Spatial-temporal
        Attention Module for 3D Convolution Network in Action Recognition." DEStech Transactions on Computer
        Science and Engineering cisnrc (2019).
    """

    def __init__(self, channel, reduction=16):
        super(STAMLayer, self).__init__(channel, reduction)
        self.kernel_size = 7
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
        )
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        y = y.mean(1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class SELayerCoC(SELayer):
    """Construct convolution-based channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerCoC, self).__init__(channel, reduction)
        self.conv1 = nn.Conv3d(
            in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(num_features=self.channel // self.reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(
            in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(num_features=self.channel)

    def forward(self, x):
        b, c, t, _, _ = x.size()  # n, c, t, h, w
        y = self.conv1(x)  # n, c/r, t, h, w
        y = self.bn1(y)  # n, c/r, t, h, w
        y = self.avg_pool(y)  # n, c/r, 1, 1, 1
        y = self.conv2(y)  # n, c, 1, 1, 1
        y = self.bn2(y)  # n, c, 1, 1, 1
        y = self.sigmoid(y)  # n, c, 1, 1, 1
        # out = x * y.expand_as(x)  # n, c, t, h, w
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMC(SELayer):
    """Construct channel-wise SELayer with max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMC, self).__init__(channel, reduction)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMAC(SELayer):
    """Construct channel-wise SELayer with the mix of average pooling and max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMAC, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat((y_avg, y_max), dim=2).squeeze().unsqueeze(dim=1)
        y = self.conv(y).squeeze()
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out
