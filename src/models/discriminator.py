import torch
import torch.nn as nn
import numpy as np
from typing import Any, List


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(in_channel: int, out_channels: int, norm: bool = True):
            layers: List[Any] = [
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x).reshape(-1)
        # print(y_hat.shape)
        return y_hat
