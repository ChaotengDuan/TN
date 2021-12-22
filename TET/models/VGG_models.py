import random
from models.layers import *


class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            TBNLayer(2, 64, 3, 1, 1),
            TBNLayer(64, 128, 3, 1, 1),
            pool,
            TBNLayer(128, 256, 3, 1, 1),
            TBNLayer(256, 256, 3, 1, 1),
            pool,
            TBNLayer(256, 512, 3, 1, 1),
            TBNLayer(512, 512, 3, 1, 1),
            pool,
            TBNLayer(512, 512, 3, 1, 1),
            TBNLayer(512, 512, 3, 1, 1),
            pool,
        )
        W = int(128 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGbase(nn.Module):
    def __init__(self):
        super(VGGbase, self).__init__()
        self.features = nn.Sequential(
            Layer(3, 64, 3, 1, 1),
            LIFSpike(),
            Layer(64, 128, 3, 2, 1),
            LIFSpike(),
            Layer(128, 256, 3, 1, 1),
            LIFSpike(),
            Layer(256, 256, 3, 2, 1),
            LIFSpike(),
            Layer(256, 512, 3, 1, 1),
            LIFSpike(),
            Layer(512, 512, 3, 2, 1),
            LIFSpike(),
            Layer(512, 512, 3, 1, 1),
            LIFSpike(),
            Layer(512, 512, 3, 2, 1),
            LIFSpike(),
        )
        W = int(32 / 2 / 2 / 2 / 2)
        self.T = 6
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGTN(nn.Module):
    def __init__(self):
        super(VGGTN, self).__init__()
        self.features = nn.Sequential(
            TNLayer(3, 64, 3, 1, 1),
            LIFSpike(),
            TNLayer(64, 128, 3, 2, 1),
            LIFSpike(),
            TNLayer(128, 256, 3, 1, 1),
            LIFSpike(),
            TNLayer(256, 256, 3, 2, 1),
            LIFSpike(),
            TNLayer(256, 512, 3, 1, 1),
            LIFSpike(),
            TNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
            TNLayer(512, 512, 3, 1, 1),
            LIFSpike(),
            TNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
        )
        W = int(32 / 2 / 2 / 2 / 2)
        self.T = 6
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGTBN(nn.Module):
    def __init__(self):
        super(VGGTBN, self).__init__()
        self.features = nn.Sequential(
            TBNLayer(3, 64, 3, 1, 1),
            LIFSpike(),
            TBNLayer(64, 128, 3, 2, 1),
            LIFSpike(),
            TBNLayer(128, 256, 3, 1, 1),
            LIFSpike(),
            TBNLayer(256, 256, 3, 2, 1),
            LIFSpike(),
            TBNLayer(256, 512, 3, 1, 1),
            LIFSpike(),
            TBNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
            TBNLayer(512, 512, 3, 1, 1),
            LIFSpike(),
            TBNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
        )
        W = int(32 / 2 / 2 / 2 / 2)
        self.T = 6
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGtdBN(nn.Module):
    def __init__(self):
        super(VGGtdBN, self).__init__()
        self.features = nn.Sequential(
            tdBNLayer(3, 64, 3, 1, 1),
            LIFSpike(),
            tdBNLayer(64, 128, 3, 2, 1),
            LIFSpike(),
            tdBNLayer(128, 256, 3, 1, 1),
            LIFSpike(),
            tdBNLayer(256, 256, 3, 2, 1),
            LIFSpike(),
            tdBNLayer(256, 512, 3, 1, 1),
            LIFSpike(),
            tdBNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
            tdBNLayer(512, 512, 3, 1, 1),
            LIFSpike(),
            tdBNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
        )
        W = int(32 / 2 / 2 / 2 / 2)
        self.T = 6
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGTTBN(nn.Module):
    def __init__(self):
        super(VGGTTBN, self).__init__()
        self.features = nn.Sequential(
            TTBNLayer(3, 64, 3, 1, 1),
            LIFSpike(),
            TTBNLayer(64, 128, 3, 2, 1),
            LIFSpike(),
            TTBNLayer(128, 256, 3, 1, 1),
            LIFSpike(),
            TTBNLayer(256, 256, 3, 2, 1),
            LIFSpike(),
            TTBNLayer(256, 512, 3, 1, 1),
            LIFSpike(),
            TTBNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
            TTBNLayer(512, 512, 3, 1, 1),
            LIFSpike(),
            TTBNLayer(512, 512, 3, 2, 1),
            LIFSpike(),
        )
        W = int(32 / 2 / 2 / 2 / 2)
        self.T = 6
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGG()
