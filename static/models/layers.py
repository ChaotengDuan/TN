import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(6, 1, 1, 1, 1, device=device))

    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y


class TBN(nn.Module):  # tdBN+PN
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(TBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(6, 1, 1, 1, 1, device=device))

    def forward(self, input):
        y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        y = y.contiguous().transpose(1, 2)

        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW

        return y


class tdBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(tdBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)

    def forward(self, input):
        y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        return y.contiguous().transpose(1, 2)


class TTBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(TTBN, self).__init__()
        self.bn1 = nn.BatchNorm3d(6)
        self.bn2 = nn.BatchNorm3d(num_features)
        self.bn1.bias = None
        self.bn2.bias = None

    def forward(self, input):
        # N T C H W
        y = self.bn1(input)
        # return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)
        y = y.contiguous().transpose(1, 2)  # N T C H W -> N C T H W
        y = self.bn2(y)
        return y.contiguous().transpose(1, 2)  # N C T H W -> N T C H W


class TensorNormalization(nn.Module):
    def __init__(self, mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def forward(self, X):
        return normalizex(X, self.mean, self.std)


def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class TTBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=1, padding=1):
        super(TTBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TTBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        x = self.bn(x)
        # x = self.act(x)
        return x


class tdBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=1, padding=1):
        super(tdBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = tdBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        x = self.bn(x)
        # x = self.act(x)
        return x


class TBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=1, padding=1):
        super(TBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TBN(out_plane)

    def forward(self, input):
        y = self.fwd(input)
        y = self.bn(y)
        return y


class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x


class TNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y


class APLayer(nn.Module):
    def __init__(self, kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def extra_repr(self):
        return f'p={self.p}'

    def reset(self):
        self.mask = None

    def create_mask(self, x: torch.Tensor):
        self.mask = (F.dropout(torch.ones_like(x), p=self.p, training=True) > 0).float()

    #     def forward(self, x: torch.Tensor):
    #         self.init()
    #         print(self.mask)
    #         if self.training:
    #             if self.mask is None:
    #                 self.create_mask(x)
    #             print(self.mask)
    #             return x * self.mask
    #         else:
    #             return x
    def forward(self, x_seq: torch.Tensor):
        self.reset()
        if self.training:
            y = x_seq.transpose(0, 1).contiguous()  # N T C H W -> T N C H W
            if self.mask is None:
                self.create_mask(y[0])
            y = y * self.mask
            return y.contiguous().transpose(0, 1)
        else:
            return x_seq


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y
