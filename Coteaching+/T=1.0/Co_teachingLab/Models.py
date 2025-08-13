
import torch
import torch.nn as nn


class EEGNet_PT452_FC1(nn.Module):

    def __init__(self, nb_classes=2, input_dim=512, fc_dim=128):
        super().__init__()

        self.nb_classes = nb_classes
        self.input_dim = input_dim
        self.fc_dim = fc_dim

        self.fc = nn.Linear(self.input_dim, nb_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


