# This architecture code is implemented based on following paper
# https://arxiv.org/abs/1807.06521
# This code is referred from the official CBAM repository https://github.com/Jongchan/attention-module
# Convolution Block Attention Module
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        # self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01, affine=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        channel_attention = F.sigmoid(channel_att_sum)
        # channel_attention_expanded = channel_attention.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x, channel_attention
        # return  x * channel_attention_expanded, channel_attention


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 1
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, padding=0)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        spatial_attention = F.sigmoid(x_out)
        return x, spatial_attention
        # return x * spatial_attention, spatial_attention


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, ["avg", "max"])
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_c, channel_attention = self.ChannelGate(x)
        x_s, spatial_attention = self.SpatialGate(x)
        channel_attention_expanded = (
            channel_attention.unsqueeze(2).unsqueeze(3).expand_as(x)
        )
        x_out = x * spatial_attention * channel_attention_expanded
        return x, channel_attention, spatial_attention, x_out
