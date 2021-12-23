import random
from models.layers import *



class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            TBNLayer(2,64,3,1,1),
            TBNLayer(64,128,3,1,1),
            pool,
            TBNLayer(128,256,3,1,1),
            TBNLayer(256,256,3,1,1),
            pool,
            TBNLayer(256,512,3,1,1),
            TBNLayer(512,512,3,1,1),
            pool,
            TBNLayer(512,512,3,1,1),
            TBNLayer(512,512,3,1,1),
            pool,
        )
        W = int(128/2/2/2/2)
        # self.T = 4
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


class VGGbase(nn.Module):
    def __init__(self):
        super(VGGbase, self).__init__()
        self.features = nn.Sequential(
            Layer(3,64,3,1,1),
            LIFSpike(),
            Layer(64,128,3,2,1),
            LIFSpike(),
            Layer(128,256,3,1,1),
            LIFSpike(),
            Layer(256,256,3,2,1),
            LIFSpike(),
            Layer(256,512,3,1,1),
            LIFSpike(),
            Layer(512,512,3,2,1),
            LIFSpike(),
            Layer(512,512,3,1,1),
            LIFSpike(),
            Layer(512,512,3,2,1),
            LIFSpike(),
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
    
class VGGBNTT(nn.Module):
    def __init__(self):
        super(VGGBNTT, self).__init__()
        self.features = nn.Sequential(
            BNTTLayer(3,64,3,1,1),
            LIFSpike(),
            BNTTLayer(64,128,3,2,1),
            LIFSpike(),
            BNTTLayer(128,256,3,1,1),
            LIFSpike(),
            BNTTLayer(256,256,3,2,1),
            LIFSpike(),
            BNTTLayer(256,512,3,1,1),
            LIFSpike(),
            BNTTLayer(512,512,3,2,1),
            LIFSpike(),
            BNTTLayer(512,512,3,1,1),
            LIFSpike(),
            BNTTLayer(512,512,3,2,1),
            LIFSpike(),
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
    
# class VGGBNTT(nn.Module):
#     def __init__(self):
#         super(VGGBNTT, self).__init__()
#         self.T = 6
#         self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
#         self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64) for i in range(self.T)])
#         self.sn1 = LIFSpike()
#         self.conv2 = nn.Conv2d(64,128,3,2,1)
#         self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128) for i in range(self.T)])
#         self.sn2 = LIFSpike()
#         self.conv3 = nn.Conv2d(128,256,3,1,1)
#         self.bntt3 = nn.ModuleList([nn.BatchNorm2d(256) for i in range(self.T)])
#         self.sn3 = LIFSpike()
#         self.conv4 = nn.Conv2d(256,256,3,2,1)
#         self.bntt4 = nn.ModuleList([nn.BatchNorm2d(256) for i in range(self.T)])
#         self.sn4 = LIFSpike()
#         self.conv5 = nn.Conv2d(256,512,3,1,1)
#         self.bntt5 = nn.ModuleList([nn.BatchNorm2d(512) for i in range(self.T)])
#         self.sn5 = LIFSpike()
#         self.conv6 = nn.Conv2d(512,512,3,2,1)
#         self.bntt6 = nn.ModuleList([nn.BatchNorm2d(512) for i in range(self.T)])
#         self.sn6 = LIFSpike()
#         self.conv7 = nn.Conv2d(512,512,3,1,1)
#         self.bntt7 = nn.ModuleList([nn.BatchNorm2d(512) for i in range(self.T)])
#         self.sn7 = LIFSpike()
#         self.conv8 = nn.Conv2d(512,512,3,2,1)
#         self.bntt8 = nn.ModuleList([nn.BatchNorm2d(512) for i in range(self.T)])
#         self.sn8 = LIFSpike()
#         W = int(32/2/2/2/2)
#         self.classifier = nn.Linear(512*W*W,10)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         x = self.sn1(self.bntt1[0](self.conv1(input)))
#         x = self.sn2(self.bntt2[0](self.conv2(x)))
#         x = self.sn3(self.bntt3[0](self.conv3(x)))
#         x = self.sn4(self.bntt4[0](self.conv4(x)))
#         x = self.sn5(self.bntt5[0](self.conv5(x)))
#         x = self.sn6(self.bntt6[0](self.conv6(x)))
#         x = self.sn7(self.bntt7[0](self.conv7(x)))
#         x = self.sn8(self.bntt8[0](self.conv8(x)))
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         output = x.unsqueeze(1)
#         for i in range(1, self.T):
#             x = self.sn1(self.bntt1[i](self.conv1(input)))
#             x = self.sn2(self.bntt2[i](self.conv2(x)))
#             x = self.sn3(self.bntt3[i](self.conv3(x)))
#             x = self.sn4(self.bntt4[i](self.conv4(x)))
#             x = self.sn5(self.bntt5[i](self.conv5(x)))
#             x = self.sn6(self.bntt6[i](self.conv6(x)))
#             x = self.sn7(self.bntt7[i](self.conv7(x)))
#             x = self.sn8(self.bntt8[i](self.conv8(x)))
#             x = torch.flatten(x, 1)
#             x = self.classifier(x)
#             output = torch.cat((output, x.unsqueeze(1)),1)
#         return output
    
class VGGTN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGTN, self).__init__()
        self.tau = tau
        self.features = nn.Sequential(
            TNLayer(3,64,3,1,1),
            LIFSpike(tau=self.tau),
            TNLayer(64,128,3,2,1),
            LIFSpike(tau=self.tau),
            TNLayer(128,256,3,1,1),
            LIFSpike(tau=self.tau),
            TNLayer(256,256,3,2,1),
            LIFSpike(tau=self.tau),
            TNLayer(256,512,3,1,1),
            LIFSpike(tau=self.tau),
            TNLayer(512,512,3,2,1),
            LIFSpike(tau=self.tau),
            TNLayer(512,512,3,1,1),
            LIFSpike(tau=self.tau),
            TNLayer(512,512,3,2,1),
            LIFSpike(tau=self.tau),
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

class VGGTBN(nn.Module):
    def __init__(self):
        super(VGGTBN, self).__init__()
        self.features = nn.Sequential(
            TBNLayer(3,64,3,1,1),
            LIFSpike(),
            TBNLayer(64,128,3,2,1),
            LIFSpike(),
            TBNLayer(128,256,3,1,1),
            LIFSpike(),
            TBNLayer(256,256,3,2,1),
            LIFSpike(),
            TBNLayer(256,512,3,1,1),
            LIFSpike(),
            TBNLayer(512,512,3,2,1),
            LIFSpike(),
            TBNLayer(512,512,3,1,1),
            LIFSpike(),
            TBNLayer(512,512,3,2,1),
            LIFSpike(),
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

class VGGtdBN(nn.Module):
    def __init__(self):
        super(VGGtdBN, self).__init__()
        self.features = nn.Sequential(
            tdBNLayer(3,64,3,1,1),
            LIFSpike(),
            tdBNLayer(64,128,3,2,1),
            LIFSpike(),
            tdBNLayer(128,256,3,1,1),
            LIFSpike(),
            tdBNLayer(256,256,3,2,1),
            LIFSpike(),
            tdBNLayer(256,512,3,1,1),
            LIFSpike(),
            tdBNLayer(512,512,3,2,1),
            LIFSpike(),
            tdBNLayer(512,512,3,1,1),
            LIFSpike(),
            tdBNLayer(512,512,3,2,1),
            LIFSpike(),
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

class VGGTTBN(nn.Module):
    def __init__(self):
        super(VGGTTBN, self).__init__()
        self.features = nn.Sequential(
            TTBNLayer(3,64,3,1,1),
            LIFSpike(),
            TTBNLayer(64,128,3,2,1),
            LIFSpike(),
            TTBNLayer(128,256,3,1,1),
            LIFSpike(),
            TTBNLayer(256,256,3,2,1),
            LIFSpike(),
            TTBNLayer(256,512,3,1,1),
            LIFSpike(),
            TTBNLayer(512,512,3,2,1),
            LIFSpike(),
            TTBNLayer(512,512,3,1,1),
            LIFSpike(),
            TTBNLayer(512,512,3,2,1),
            LIFSpike(),
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
    model = VGG()
    