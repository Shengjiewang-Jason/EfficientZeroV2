# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Post Activated Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out

# Residual block
class FCResidualBlock(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(FCResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.bn1 = nn.BatchNorm1d(hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)
        self.bn2 = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += identity
        out = nn.functional.relu(out)
        return out


def mlp(
    input_size,
    hidden_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ELU,
    init_zero=False,
):
    """
    MLP layers
    :param input_size:
    :param hidden_sizes:
    :param output_size:
    :param output_activation:
    :param activation:
    :param init_zero:   bool, zero initialization for the last layer (including w and b).
                        This can provide stable zero outputs in the beginning.
    :return:
    """
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1]),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
