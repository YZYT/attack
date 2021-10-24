import torch.nn as nn
import torch
from torchvision.models import alexnet

from models.layers.conv2d import ConvBlock

import torch.optim as optim


class LeNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.net1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)

        self.net2 = nn.Sequential(nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                    nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes))

        # self.net2 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=28, stride=28),
        #     nn.Flatten(),
        #     nn.Linear(6, num_classes))
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)


    def forward(self, X):
        return self.net2(self.net1(X))

class ToyNet(nn.Module):

    def __init__(self, n_features=2, num_classes=10):
        super(ToyNet, self).__init__()
        self.net1 = nn.Linear(n_features, 2)


        self.net2 = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
        # self.net2 = nn.Sequential(nn.Linear(2, 1))
        
        # def init_weights(m):
        #     if type(m) == nn.Linear:
        #         # nn.init.normal_(m.weight, std=0.01)
        #         # nn.init.xavier_uniform_(m.weight, gain=1)
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # self.net1.apply(init_weights)
        # self.net2.apply(init_weights)


    def forward(self, X):
        return self.net2(self.net1(X))
        # return (self.net1(X))


class LeNet_passport(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet_passport, self).__init__()
        self.step = 0

        self.net1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)

        self.passport_gamma = torch.randn((1, 1, 28, 28), device='cuda')
        self.passport_beta = torch.randn((1, 1, 28, 28), device='cuda')
    
        self.encoder = nn.Sequential(nn.Flatten(), 
                                nn.Linear(6 * 28 * 28, 10), 
                                # nn.Linear(128, 6 * 28 * 28)
                                )

        self.net2 = nn.Sequential(nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                    nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes))
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                # nn.init.normal_(m.weight, std=0.01)
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)


    def forward(self, X):
        y = self.net1(X)
        
        gamma = self.encoder(self.net1(self.passport_gamma)).mean()
        beta = self.encoder(self.net1(self.passport_beta)).mean()

        gamma = 1. + (gamma - 1) * 0.01
        beta /= 100

        y.mul_(gamma).add_(beta)
        return self.net2(y)