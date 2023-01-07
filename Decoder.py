from torch import nn
import torch


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_part = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 64 * 4 * 4 * 3),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4, 3))

        self.conv_part = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=(1, 1, 0), stride=(2, 2, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, output_padding=(1, 1, 0), stride=(2, 2, 1), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, output_padding=(1, 1, 0), stride=(2, 2, 1), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 3, output_padding=(1, 1, 0), stride=(2, 2, 1), padding=1),
        )

    def forward(self, x):
        x = self.lin_part(x)
        x = self.unflatten(x)
        x = self.conv_part(x)
        x = torch.sigmoid(x)
        x = torch.squeeze(x)
        return x
