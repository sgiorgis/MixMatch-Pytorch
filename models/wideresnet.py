import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(4, 16, 32, BasicBlock, 1, True)
        self.block2 = NetworkBlock(4, 32, 64, BasicBlock, 2)
        self.block3 = NetworkBlock(4, 64, 128, BasicBlock, 2)

        self.batch_norm = nn.BatchNorm2d(128, momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.linear = nn.Linear(128, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.batch_norm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, 128)

        return self.linear(out)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            out = self.relu1(self.batch_norm(x))
        else:
            out = self.relu1(self.batch_norm(x))

        out = self.relu2(self.bn2(self.conv1(out)))
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                activate_before_residual))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
