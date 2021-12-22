import random
from models.layers import *

class VGGSNNPN(nn.Module):
    def __init__(self):
        super(VGGSNNPN, self).__init__()
        self.features = nn.Sequential(
            TNLayer(2,64,3,1,1),
            LIFSpike(),
            TNLayer(64,128,3,2,1),
            LIFSpike(),
            TNLayer(128,256,3,1,1),
            LIFSpike(),
            TNLayer(256,256,3,2,1),
            LIFSpike(),
            TNLayer(256,512,3,1,1),
            LIFSpike(),
            TNLayer(512,512,3,2,1),
            LIFSpike(),
            TNLayer(512,512,3,1,1),
            LIFSpike(),
            TNLayer(512,512,3,2,1),
            LIFSpike(),
        )
        W = int(128/2/2/2/2)
        #self.T = 10
        self.classifier = nn.Sequential(Dropout(0.25),SeqToANNContainer(nn.Linear(512*W*W,10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x
#     def reset_(self):
#         for item in self.modules():
#             if hasattr(item, 'reset'):
#                 item.reset()

class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            TNLayer(2,64,3,1,1),
            LIFSpike(),
            TNLayer(64,128,3,1,1),
            LIFSpike(),
            pool,
            TNLayer(128,256,3,1,1),
            LIFSpike(),
            TNLayer(256,256,3,1,1),
            LIFSpike(),
            pool,
            TNLayer(256,512,3,1,1),
            LIFSpike(),
            TNLayer(512,512,3,1,1),
            LIFSpike(),
            pool,
            TNLayer(512,512,3,1,1),
            LIFSpike(),
            TNLayer(512,512,3,1,1),
            LIFSpike(),
            pool,
        )
        W = int(128/2/2/2/2)
        # self.T = 10
#         self.classifier = nn.Sequential(Dropout(0.25),SeqToANNContainer(nn.Linear(512*W*W,10)))
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x
#     def reset_(self):
#         for item in self.modules():
#             if hasattr(item, 'reset'):
#                 item.reset()
class VGGSNNwoAP(nn.Module):
    def __init__(self):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            BNLayer(3,64,3,1,1),
            BNLayer(64,128,3,2,1),
            BNLayer(128,256,3,1,1),
            BNLayer(256,256,3,2,1),
            BNLayer(256,512,3,1,1),
            BNLayer(512,512,3,2,1),
            BNLayer(512,512,3,1,1),
            BNLayer(512,512,3,2,1),
        )
        W = int(32/2/2/2/2)
        self.T = 6
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

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
    model = VGGSNNwoAP()
    