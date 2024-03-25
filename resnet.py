# RESNET ARCHITECTURE IMPLEMENTATION
# This model code is implemented based on following paper
# https://arxiv.org/abs/1807.06521
# This code is referred from the official CBAM repository https://github.com/Jongchan/attention-module
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from cbam import CBAM


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            f_map, channel, spatial, out = self.cbam(out)
            out += residual
            out = self.relu(out)
            return f_map, channel, spatial, out
        else:
            out += residual
            out = self.relu(out)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out, channel, spatial = self.cbam(out)
            out += residual
            out = self.relu(out)
            return out, channel, spatial
        else:
            out += residual
            out = self.relu(out)
            return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        network_type,
        num_classes,
        att_type=None,
        blocks_list=[64, 128, 256, 512],
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1_blocks = nn.ModuleList()
        self.layer1_blocks.append(
            block(self.inplanes, blocks_list[0], 1, None, use_cbam=att_type == "CBAM")
        )
        self.inplanes = blocks_list[0] * block.expansion
        for i in range(1, layers[0]):
            self.layer1_blocks.append(
                block(self.inplanes, blocks_list[0], use_cbam=att_type == "CBAM")
            )

        self.layer2_blocks = nn.ModuleList()
        self.layer2_blocks.append(
            block(
                self.inplanes,
                blocks_list[1],
                2,
                self.get_downsample_layer(block, blocks_list[1]),
                use_cbam=att_type == "CBAM",
            )
        )
        self.inplanes = blocks_list[1] * block.expansion
        for i in range(1, layers[1]):
            self.layer2_blocks.append(
                block(self.inplanes, blocks_list[1], use_cbam=att_type == "CBAM")
            )

        self.layer3_blocks = nn.ModuleList()
        self.layer3_blocks.append(
            block(
                self.inplanes,
                blocks_list[2],
                2,
                self.get_downsample_layer(block, blocks_list[2]),
                use_cbam=att_type == "CBAM",
            )
        )
        self.inplanes = blocks_list[2] * block.expansion
        for i in range(1, layers[2]):
            self.layer3_blocks.append(
                block(self.inplanes, blocks_list[2], use_cbam=att_type == "CBAM")
            )

        self.layer4_blocks = nn.ModuleList()
        self.layer4_blocks.append(
            block(
                self.inplanes,
                blocks_list[3],
                2,
                self.get_downsample_layer(block, blocks_list[3]),
                use_cbam=att_type == "CBAM",
            )
        )
        self.inplanes = blocks_list[3] * block.expansion
        for i in range(1, layers[3]):
            self.layer4_blocks.append(
                block(self.inplanes, blocks_list[3], use_cbam=att_type == "CBAM")
            )

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)

    def get_downsample_layer(self, block, planes):
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion),
        )
        return downsample

    def forward(self, x):
        channel_attention_maps = []
        spatial_attention_maps = []
        channel_feature_maps = []
        spatial_feature_maps = []
        feature_maps = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layer1_blocks:
            f_map, channel, spatial, x = layer(x)
            feature_maps.append(f_map)
            channel_attention_maps.append(channel)
            spatial_attention_maps.append(spatial)

        for layer in self.layer2_blocks:
            f_map, channel, spatial, x = layer(x)
            feature_maps.append(f_map)
            channel_attention_maps.append(channel)
            spatial_attention_maps.append(spatial)

        for layer in self.layer3_blocks:
            f_map, channel, spatial, x = layer(x)
            feature_maps.append(f_map)
            channel_attention_maps.append(channel)
            spatial_attention_maps.append(spatial)

        for layer in self.layer4_blocks:
            f_map, channel, spatial, x = layer(x)
            feature_maps.append(f_map)
            channel_attention_maps.append(channel)
            spatial_attention_maps.append(spatial)

        x = F.avg_pool2d(x, 28)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, feature_maps, channel_attention_maps, spatial_attention_maps


def ResidualNet(network_type, depth, num_classes):
    att_type = "CBAM"
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model
