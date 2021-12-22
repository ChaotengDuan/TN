import random
import torch
import torch.nn as nn
from models.layers import *




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, method='base'):
        super(BasicBlock, self).__init__()
        if method == 'TN':
            self.conv1 = TNLayer(in_ch, out_ch, 3, stride, 1)
        elif method == 'TBN':
            self.conv1 = TBNLayer(in_ch, out_ch, 3, stride, 1)
        elif method == 'tdBN':
            self.conv1 = tdBNLayer(in_ch, out_ch, 3, stride, 1)
        elif method == 'TTBN':
            self.conv1 = TTBNLayer(in_ch, out_ch, 3, stride, 1)
        elif method == 'base':
            self.conv1 = Layer(in_ch, out_ch, 3, stride, 1)
        self.sn1 = LIFSpike()

        if method == 'TN':
            self.conv2 = TNLayer(out_ch, out_ch, 3, 1, 1)
        elif method == 'TBN':
            self.conv2 = TBNLayer(out_ch, out_ch, 3, 1, 1)
        elif method == 'tdBN':
            self.conv2 = tdBNLayer(out_ch, out_ch, 3, 1, 1)
        elif method == 'TTBN':
            self.conv2 = TTBNLayer(out_ch, out_ch, 3, 1, 1)
        elif method == 'base':
            self.conv2 = Layer(out_ch, out_ch, 3, 1, 1)
        self.sn2 = LIFSpike()

        self.stride = stride
        self.downsample = downsample

        if method == 'TN':
            self.bn = TN(out_ch)
        elif method == 'TBN':
            self.bn = TBN(out_ch)
        elif method == 'tdBN':
            self.bn = tdBN(out_ch)
        elif method == 'TTBN':
            self.bn = TTBN(out_ch)
        elif method == 'base':
            self.bn = SeqToANNContainer(nn.BatchNorm2d(out_ch))

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
    def __init__(self, block=BasicBlock, layers=[3, 3, 2], method='base'):
        super(ResNet, self).__init__()
        self.T = 6
        self.in_ch = 128
        self.method = method
        if self.method == 'TN':
            self.conv1 = TNLayer(3, self.in_ch, 3, 1, 1)
        elif self.method == 'TBN':
            self.conv1 = TBNLayer(3, self.in_ch, 3, 1, 1)
        elif self.method == 'tdBN':
            self.conv1 = tdBNLayer(3, self.in_ch, 3, 1, 1)
        elif self.method == 'TTBN':
            self.conv1 = TTBNLayer(3, self.in_ch, 3, 1, 1)
        elif self.method == 'base':
            self.conv1 = Layer(3, self.in_ch, 3, 1, 1)
        self.sn1 = LIFSpike()

        self.pool = SeqToANNContainer(nn.AvgPool2d(2))

        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 512, layers[2], stride=2)
        #         self.layer4 = self.make_layer(block, 512, layers[3], stride=2,)    # ImageNet

        W = int(32 / 2 / 2 / 2)
        self.fc1 = nn.Sequential(Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 256)))
        self.fc2 = nn.Sequential(Dropout(0.25), SeqToANNContainer(nn.Linear(256, 100)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, in_ch, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != in_ch * block.expansion:
            if self.method == 'TN':
                downsample = TNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
            elif self.method == 'TBN':
                downsample = TBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
            elif self.method == 'tdBN':
                downsample = tdBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
            elif self.method == 'TTBN':
                downsample = TTBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
            elif self.method == 'base':
                downsample = Layer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
        layers = []
        layers.append(block(self.in_ch, in_ch, stride, downsample, method=self.method))
        self.in_ch = in_ch * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_ch, in_ch, method=self.method))

        return nn.Sequential(*layers)

    def forward_imp(self, input):
        x = add_dimention(input, self.T)

        x = self.conv1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #         x = self.layer4(x)
        x = self.pool(x)

        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x

    def forward(self, input):
        return self.forward_imp(input)


if __name__ == '__main__':
    model = ResNet()