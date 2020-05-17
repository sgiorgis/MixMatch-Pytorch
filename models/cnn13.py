import torch
import torch.nn as nn


class CNN13(nn.Module):

    def __init__(self, kernel_size=3, filters=32, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=kernel_size, padding=1)
        self.conv1_bn = nn.BatchNorm2d(filters, momentum=0.999)

        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=1)
        self.conv2_bn = nn.BatchNorm2d(filters, momentum=0.999)

        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=1)
        self.conv3_bn = nn.BatchNorm2d(filters, momentum=0.999)

        self.conv4 = nn.Conv2d(in_channels=filters, out_channels=2 * filters, kernel_size=kernel_size, padding=1)
        self.conv4_bn = nn.BatchNorm2d(2 * filters, momentum=0.999)

        self.conv5 = nn.Conv2d(in_channels=2 * filters, out_channels=2 * filters, kernel_size=kernel_size, padding=1)
        self.conv5_bn = nn.BatchNorm2d(2 * filters, momentum=0.999)

        self.conv6 = nn.Conv2d(in_channels=2 * filters, out_channels=2 * filters, kernel_size=kernel_size, padding=1)
        self.conv6_bn = nn.BatchNorm2d(2 * filters, momentum=0.999)

        self.conv7 = nn.Conv2d(in_channels=2 * filters, out_channels=4 * filters, kernel_size=kernel_size)
        self.conv7_bn = nn.BatchNorm2d(4 * filters, momentum=0.999)

        self.conv8 = nn.Conv2d(in_channels=4 * filters, out_channels=2 * filters, kernel_size=1)
        self.conv8_bn = nn.BatchNorm2d(2 * filters, momentum=0.999)

        self.conv9 = nn.Conv2d(in_channels=2 * filters, out_channels=filters, kernel_size=1)
        self.conv9_bn = nn.BatchNorm2d(filters, momentum=0.999)

        self.leaky_relu = nn.LeakyReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(filters, num_classes)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.conv1_bn(self.leaky_relu(x))
        x = self.conv2(x)
        x = self.conv2_bn(self.leaky_relu(x))
        x = self.conv3(x)
        x = self.conv3_bn(self.leaky_relu(x))
        x = self.max_pooling(x)
        x = self.conv4(x)
        x = self.conv4_bn(self.leaky_relu(x))
        x = self.conv5(x)
        x = self.conv5_bn(self.leaky_relu(x))
        x = self.conv6(x)
        x = self.conv6_bn(self.leaky_relu(x))
        x = self.max_pooling(x)
        x = self.conv7(x)
        x = self.conv7_bn(self.leaky_relu(x))
        x = self.conv8(x)
        x = self.conv8_bn(self.leaky_relu(x))
        x = self.conv9(x)
        x = self.conv9_bn(self.leaky_relu(x))
        x = self.linear(torch.mean(x, [2, 3]))

        return x