import torch.nn as nn


class Stage(nn.Module):
    def __init__(self, out_channels, layers):
        super(Stage, self).__init__()
        self.feature = nn.Sequential(*layers)
        self.out_channels = out_channels
        
    def forward(self, x):
        return self.feature(x)
