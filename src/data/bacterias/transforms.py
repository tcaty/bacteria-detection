import torch
from torchvision import transforms
from PIL.Image import Image


class Square(torch.nn.Module):
    """Makes image square by adding padding for min size (1/2 for both sides)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: Image) -> Image:
        size = image.size

        min_size_idx = size.index(min(size))
        max_size_idx = size.index(max(size))

        padding = [0, 0]
        padding[min_size_idx] = (size[max_size_idx] - size[min_size_idx]) // 2

        transform = transforms.Pad(padding)

        return transform(image)
