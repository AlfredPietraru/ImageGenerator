import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.addapt_linear = nn.Linear(100, 16 * 1024)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, padding=1, stride=2),      
            nn.Tanh(),
        )

    def forward(self, x : torch.Tensor):
        x = self.addapt_linear(x)
        x = x.reshape(shape=(x.shape[0], 1024, 4, 4))
        return self.generator(x)
        

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2), # primul lucru de scos daca nu merge
        )

        self.linear_goodby = nn.Linear(1024 * 4 * 4, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x : torch.Tensor):
        x = self.discriminator(x)
        x = x.flatten(start_dim=1)
        return self.sig(self.linear_goodby(x))