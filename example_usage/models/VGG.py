'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn

from kernels.DualConv import DualConv
from kernels.GroupConv import GroupConv
from kernels.HetConv import HetConv

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, kernel, divide):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], kernel, divide)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, kernel, divide):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if in_channels == 3:
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                else:
                    if kernel == 'dualconv':
                        conv2d = DualConv(in_channels, x, 1, divide)
                    elif kernel == 'groupconv':
                        conv2d = GroupConv(in_channels, x, 1, divide)
                    elif kernel == 'groupconv':
                        conv2d = HetConv(in_channels, x, 1, divide)
                layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

