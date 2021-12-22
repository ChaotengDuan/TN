import random
import torch
import torch.nn as nn
from models.layers import *

use_TTBN = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, use_TTBN=False):
        super(BasicBlock, self).__init__()
        self.use_TTBN = use_TTBN
        #         self.conv1 = TTBNLayer(in_ch, out_ch, 3, stride, 1) if self.use_TTBN else tdBNLayer(in_ch, out_ch, 3, stride, 1)
        self.conv1 = TNLayer(in_ch, out_ch, 3, stride, 1)
        self.sn1 = LIFSpike()
        #         self.conv2 = TTBNLayer(out_ch, out_ch, 3, 1, 1) if self.use_TTBN else tdBNLayer(out_ch, out_ch, 3, 1, 1)
        self.conv2 = TNLayer(out_ch, out_ch, 3, 1, 1)
        self.sn2 = LIFSpike()
        self.stride = stride
        self.downsample = downsample
        #         self.bn = TTBN(out_ch) if self.use_TTBN else  tdBN(out_ch)
        self.bn = TN(out_ch)
    #         self.bn = SeqToANNContainer(nn.BatchNorm2d(out_ch))

    def forward(self, x):
        right = x
        y = self.conv1(x)
        y = self.sn1(y)
        y = self.conv2(y)
        if self.downsample is not None:
            right = self.downsample(x)
        else:
            right = self.bn(x)
        y += right
        y = self.sn2(y)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = BNLayer(in_ch, out_ch, 1, stride, 1)
        self.sn1 = LIFSpike()
        self.conv2 = BNLayer(out_ch, out_ch, 3, 1, 1)
        self.sn2 = LIFSpike()
        self.conv3 = BNLayer(out_ch, out_ch, 1, 1, 1)
        self.sn3 = LIFSpike()
        self.stride = stride
        self.downsample = downsample
        self.bn = TTBN(out_ch)

    def forward(self, x):
        right = x
        y = self.conv1(x)
        y = self.sn1(y)
        y = self.conv2(y)
        y = self.sn2(y)
        y = self.conv3(y)

        if self.downsample is not None:
            right = self.downsample(x)
        else:
            right = self.bn(x)
        y += right
        y = self.sn3(y)

        return y


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 3, 2], use_TTBN=False):
        super(ResNet, self).__init__()
        self.T = 10
        self.in_ch = 128
        self.use_TTBN = use_TTBN
        #         self.conv1 = TTBNLayer(3, self.in_ch, 3, 1, 1)   if self.use_TTBN else tdBNLayer(3, self.in_ch, 3, 1, 1)
        self.conv1 = TNLayer(2, self.in_ch, 3, 1, 1)
        self.sn1 = LIFSpike()
        self.pool1 = SeqToANNContainer(nn.AvgPool2d(2))
        
        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 512, layers[2], stride=2)
        #         self.layer4 = self.make_layer(block, 512, layers[3], stride=2,)    # ImageNet
        self.pool2 = SeqToANNContainer(nn.AvgPool2d(2))
        
        W = int(128 / 2 / 2 / 2 / 2)
        self.fc1 = SeqToANNContainer(nn.Linear(512 * W * W, 256), )
        self.fc2 = SeqToANNContainer(nn.Linear(256, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, in_ch, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != in_ch * block.expansion:
        # downsample = TTBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0) if self.use_TTBN else tdBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
            downsample = TNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
        layers = []
        layers.append(block(self.in_ch, in_ch, stride, downsample, use_TTBN=self.use_TTBN))
        self.in_ch = in_ch * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_ch, in_ch, use_TTBN=self.use_TTBN))
        return nn.Sequential(*layers)

    def forward_imp(self, input):
        #x = add_dimention(input, self.T)

        x = self.conv1(input)
        x = self.sn1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #         x = self.layer4(x)
        x = self.pool2(x)

        x = torch.flatten(x, 2)

        x = self.fc2(self.fc1(x))
        return x

    def forward(self, input):
        return self.forward_imp(input)


if __name__ == '__main__':
    model = ResNet()