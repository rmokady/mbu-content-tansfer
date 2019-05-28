import torch.nn as nn
import torch

class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
            nn.BatchNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
            nn.BatchNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.BatchNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class D_B(nn.Module):
    def __init__(self, size):
        super(D_B, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),
        )

    def forward(self, net, my_input):
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)
        mask = torch.sigmoid(output[:, :1])
        oimg = torch.tanh(output[:, 1:])
        mask = mask.repeat(1, 3, 1, 1)
        oimg = oimg * mask + my_input * (1 - mask)
        return oimg, mask

class D_A(nn.Module):
    def __init__(self, size):
        super(D_A, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, net):
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net

class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net
