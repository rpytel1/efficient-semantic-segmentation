from __future__ import division

import torch.nn as nn

from torch.nn.functional import interpolate

from .FastFCN_parts import FCNHead, JPU
from ..DilatedResNet import ResNet

class BaseNet(nn.Module):
    def __init__(self, n_classes, backbone, aux, se_loss, jpu=True, dilated=False, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], **kwargs):
        super(BaseNet, self).__init__()
        self.n_classes = n_classes
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = ResNet.resnet50(dilated=dilated)
        elif backbone == 'resnet101':
            self.pretrained = ResNet.resnet101(dilated=dilated)
        elif backbone == 'resnet152':
            self.pretrained = ResNet.resnet152(dilated=dilated)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self.backbone = backbone
        self.jpu = None
        if jpu:
            self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer)

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4


class FCN(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation"""

    def __init__(self, n_classes, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN, self).__init__(n_classes, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = FCNHead(2048, n_classes, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, n_classes, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = interpolate(x, imsize)
        output = x
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize)
            output = auxout
        return output
