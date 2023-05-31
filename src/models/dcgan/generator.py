import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int, stride: int, padding: int):
            return [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.model = nn.Sequential(
            *block(latent_dims, 512, stride=1, padding=0),
            *block(512, 256, stride=2, padding=1),
            *block(256, 128, stride=2, padding=1),
            *block(128, 64, stride=2, padding=1),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        # print(y_hat.shape)
        return y_hat


# latent_dims = 128
# z_dims = (latent_dims, 1, 1)
# generator = Generator(latent_dims=latent_dims)
# z = torch.randn((2, *z_dims))
# print(generator(z).shape)
