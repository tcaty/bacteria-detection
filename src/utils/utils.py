import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from constants import WORK_DIR_PATH


def show_images(
    image_with_title_tuples: list[tuple[np.ndarray, str]],
    grid=(2, 2),
    size_inches=(10, 10),
    cmap="gray",
):
    plt.gcf().set_size_inches(*size_inches)
    for i in range(len(image_with_title_tuples)):
        current_tuple = image_with_title_tuples[i]
        plt.subplot(*grid, i + 1)
        plt.imshow(current_tuple[0], cmap=cmap)
        plt.title(current_tuple[1])
        plt.xticks([])
        plt.yticks([])


def get_images_grid(images: np.ndarray, ncols: int = 3) -> np.ndarray:
    index, height, width = images.shape
    nrows = index // ncols

    img_grid = (
        images.reshape(nrows, ncols, height, width)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols)
    )

    return img_grid


def read_images(dir_path: str, limit=None) -> list[np.ndarray]:
    files = list(sorted(os.listdir(dir_path)))
    img_ext = ["png", "tif", "jpg"]
    imgs = list(filter(lambda f: f[-3:] in img_ext, files))
    return [cv.imread(f"{dir_path}/{img_name}") for img_name in imgs][:limit]


def get_model_checkpoint(model_name: str, version: int, epoch: int, step: int):
    return f"{WORK_DIR_PATH}/logs/{model_name}/lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt"


def get_bacterias_wgan_checkpoint(**kwargs):
    return get_model_checkpoint(model_name="bacterias_wgan", **kwargs)


def get_substrates_wgan_checkpoint(**kwargs):
    return get_model_checkpoint(model_name="substrates_wgan", **kwargs)


def get_bacterias_binary_segmentation_checkpoint(**kwargs):
    return get_model_checkpoint(model_name="bacterias_binary_segmentation", **kwargs)


def deepcopy_module(module):
    new_module = nn.Sequential()
    for _, module in module.named_children():
        new_module.append(module)
        new_module[-1].load_state_dict(module.state_dict())
    return new_module


def delete_description(image: np.ndarray):
    black = 0
    white = 255
    description_rows = [
        set([white, black]),
        set([black]),
        set([white])
    ]
    image_with_deleted_descr = np.array(
        list(filter(lambda row: set(row) not in description_rows, image.tolist()))
    )
    return image_with_deleted_descr