import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #osiguravanje izvrsavanja roditeljske klase
        self.flatten = nn.Flatten() #ako je slika 28x28 da bude 784
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 26),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

