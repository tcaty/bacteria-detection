import torch
import torch.nn as nn
from typing import Any, List
from utils import deepcopy_module


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(*self._block(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        # print(y_hat.shape)
        return y_hat

    def _grow(self):
        self.model.extend(self._block(3, 3))

    def _block(
        self,
        first_layer_in_channel: int = 3,
        first_layer_out_channels: int = 3,
        upsample: bool = True,
    ):
        layers = [
            nn.Conv2d(
                in_channels=first_layer_in_channel,
                out_channels=first_layer_out_channels,
                kernel_size=(3, 3),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=(3, 3),
            ),
            nn.LeakyReLU(0.2),
        ]
        if upsample:
            layers.append(nn.Upsample())
        return layers
